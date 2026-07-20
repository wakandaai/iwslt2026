"""
Stage 1: Pretrain speech encoder with CTC loss.

Step-based training with resume support. The saved checkpoint includes
the full model state + optimizer + scheduler + vocab, so Stage 2/3
can load it directly via load_encoder_from_checkpoint().

Usage:
    python -m st.training.pretrain_ctc --config <your_conformer_config.yaml>
    python -m st.training.pretrain_ctc --config <your_conformer_config.yaml> \
        --resume_from checkpoints/encoder/encoder_step50000.pt

No Conformer config currently ships in this repo (configs/experiment/ only
has the omniASR path) — this script and configs/encoder/'s architecture
configs would need to be written from scratch to use it again.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from st.data.dataset import SpeechDataset
from st.data.sampler import DurationBucketSampler
from st.data.vocab import build_vocab_from_index, save_vocab
from st.models.encoder import SpeechEncoder, load_encoder_from_checkpoint
from st.utils.config import load_config
from st.utils.metrics import compute_wer
from st.utils.schedulers import build_scheduler

log = logging.getLogger(__name__)


# ============================================================================
# ASR collator (CTC format — no Aura tokens, just mel + label sequences)
# ============================================================================

def ctc_collate(batch: list[dict], vocab: dict[str, int]) -> dict[str, torch.Tensor]:
    """Pad mel features and encode transcripts as CTC integer sequences."""
    mel_lens = torch.tensor([b["mel_len"] for b in batch], dtype=torch.long)
    max_mel  = int(mel_lens.max().item())
    mel_pad  = torch.zeros(len(batch), max_mel, 80)
    for i, b in enumerate(batch):
        mel_pad[i, : b["mel_len"]] = b["mel"]

    labels:        list[torch.Tensor] = []
    label_lengths: list[int]           = []
    for b in batch:
        # Substitute out-of-vocab characters with space (when available) rather
        # than dropping them — matches AuraCollator's CTC-label encoding
        # (src/st/data/collator.py), so Stage 1 and the Stage 2/3 auxiliary CTC
        # loss use the same label semantics for unseen characters.
        encoded = []
        for c in b["text"]:
            if c in vocab:
                encoded.append(vocab[c])
            elif " " in vocab:
                encoded.append(vocab[" "])
        labels.append(torch.tensor(encoded, dtype=torch.long))
        label_lengths.append(len(encoded))

    max_lab  = max(label_lengths) if label_lengths else 1
    lab_pad  = torch.zeros(len(batch), max_lab, dtype=torch.long)
    for i, lab in enumerate(labels):
        lab_pad[i, : lab.size(0)] = lab

    return {
        "features":       mel_pad,
        "feature_lengths": mel_lens,
        "labels":          lab_pad,
        "label_lengths":   torch.tensor(label_lengths, dtype=torch.long),
    }


# ============================================================================
# Validation
# ============================================================================

@torch.no_grad()
def validate(
    model: SpeechEncoder,
    loader: DataLoader,
    device: torch.device,
    vocab: dict[str, int],
    step: int = 0,
    output_dir: Path | None = None,
) -> dict[str, float]:
    model.eval()
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    idx_to_char = {v: k for k, v in vocab.items()}

    total_loss, n_batches = 0.0, 0
    preds: list[str] = []
    refs:  list[str] = []

    for batch in loader:
        features       = batch["features"].to(device)
        feature_lengths = batch["feature_lengths"].to(device)
        labels         = batch["labels"].to(device)
        label_lengths  = batch["label_lengths"].to(device)

        out      = model(features, feature_lengths)
        log_probs = out["ctc_logits"].log_softmax(dim=-1).transpose(0, 1)
        loss     = ctc_loss_fn(log_probs, labels, out["lengths"], label_lengths)

        total_loss += loss.item()
        n_batches  += 1

        # Greedy decode
        pred_ids = out["ctc_logits"].argmax(dim=-1)
        for i in range(pred_ids.size(0)):
            seq = pred_ids[i, : out["lengths"][i]].tolist()
            decoded, prev = [], -1
            for tid in seq:
                if tid != 0 and tid != prev:
                    decoded.append(idx_to_char.get(tid, ""))
                prev = tid
            preds.append("".join(decoded))
            ref_seq = labels[i, : label_lengths[i]].tolist()
            refs.append("".join(idx_to_char.get(t, "") for t in ref_seq))

    avg_loss = total_loss / max(n_batches, 1)
    wer      = compute_wer(preds, refs) if refs else 0.0

    # Save predictions CSV
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"val_preds_step{step}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["reference", "prediction"])
            for r, p in zip(refs, preds):
                writer.writerow([r, p])
        n_empty = sum(1 for p in preds if not p.strip())
        log.info(f"Val preds saved → {csv_path} ({n_empty} empty)")

    model.train()
    return {"val/ctc_loss": avg_loss, "val/wer": wer}


# ============================================================================
# Training
# ============================================================================

def infinite(loader: DataLoader):
    while True:
        yield from loader


def train(cfg: dict, resume_from: str | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc_cfg   = cfg["encoder"]
    train_cfg = cfg["training"]
    data_cfg  = cfg["data"]
    output_dir = Path(train_cfg.get("output_dir", "checkpoints/encoder"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Vocab ---
    vocab_path = data_cfg.get("vocab_path")
    if vocab_path and os.path.exists(vocab_path):
        from st.data.vocab import load_vocab
        vocab = load_vocab(vocab_path)
    else:
        vocab = build_vocab_from_index(
            data_cfg["train_index"],
            text_column="transcript",
            split=data_cfg.get("train_split", "train"),
            languages=data_cfg.get("languages"),
            lowercase=data_cfg.get("lowercase", False),
        )
        if vocab_path:
            save_vocab(vocab, vocab_path)

    # --- Dataset ---
    train_ds = SpeechDataset(
        index_path=data_cfg["train_index"],
        split=data_cfg.get("train_split", "train"),
        languages=data_cfg.get("languages"),
        text_column="transcript",
        max_duration=data_cfg.get("max_duration", 30.0),
        min_duration=data_cfg.get("min_duration", 0.1),
        lowercase=data_cfg.get("lowercase", False),
    )
    val_ds = SpeechDataset(
        index_path=data_cfg.get("val_index", data_cfg["train_index"]),
        split=data_cfg.get("val_split", "dev"),
        languages=data_cfg.get("languages"),
        text_column="transcript",
        max_duration=data_cfg.get("max_duration", 30.0),
        lowercase=data_cfg.get("lowercase", False),
    )

    def collate(batch):
        return ctc_collate(batch, vocab)

    train_sampler = DurationBucketSampler(
        dataset=train_ds,
        target_duration=train_cfg.get("max_batch_duration", 240.0),
        max_batch_size=train_cfg.get("max_batch_size", 32),
        shuffle=True,
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler,
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.get("val_batch_size", 16),
        shuffle=False, num_workers=train_cfg.get("num_workers", 4),
        collate_fn=collate, pin_memory=True,
    )

    # --- Model ---
    model = SpeechEncoder(
        input_dim=enc_cfg.get("input_dim", 80),
        encoder_dim=enc_cfg.get("encoder_dim", 512),
        num_heads=enc_cfg.get("num_heads", 8),
        ffn_dim=enc_cfg.get("ffn_dim", 2048),
        num_layers=enc_cfg.get("num_layers", 12),
        depthwise_conv_kernel_size=enc_cfg.get("depthwise_conv_kernel_size", 31),
        dropout=enc_cfg.get("dropout", 0.1),
        vocab_size=len(vocab),
    ).to(device)
    log.info(f"Encoder: {sum(p.numel() for p in model.parameters()):,} params")

    # --- Optimizer + Scheduler ---
    total_steps = train_cfg["total_steps"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    sched_cfg = train_cfg.get("scheduler", {})
    scheduler = build_scheduler(
        name=sched_cfg.get("name", "cosine_warmup_restarts"),
        optimizer=optimizer,
        total_steps=total_steps,
        max_lr=train_cfg.get("lr", 1e-4),
        min_lr=train_cfg.get("min_lr", 1e-6),
        warmup_steps=sched_cfg.get("warmup_steps", 2000),
        first_cycle_steps=sched_cfg.get("first_cycle_steps", total_steps),
    )

    # --- Resume ---
    start_step = 0
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt.get("step", 0)
        log.info(f"Resumed from {resume_from} at step {start_step}")

    # --- W&B ---
    use_wandb = cfg.get("wandb", {}).get("enabled", False)
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg["wandb"].get("project", "iwslt2026"),
            name=cfg["wandb"].get("name", "pretrain-ctc"),
            config=cfg,
            resume="allow" if start_step > 0 else None,
        )
    else:
        try:
            import wandb; wandb.init(mode="disabled")
        except ImportError:
            pass

    # --- Loop ---
    ctc_loss_fn  = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    log_every    = train_cfg.get("log_every_steps", 100)
    save_every   = train_cfg.get("save_every_steps", 10000)
    eval_every   = train_cfg.get("eval_every_steps", 5000)

    model.train()
    running_loss, log_steps = 0.0, 0
    pbar = tqdm(total=total_steps - start_step, desc="CTC pretrain", unit="step")

    for step_offset, batch in enumerate(infinite(train_loader), start=1):
        step = start_step + step_offset

        features        = batch["features"].to(device)
        feature_lengths = batch["feature_lengths"].to(device)
        labels          = batch["labels"].to(device)
        label_lengths   = batch["label_lengths"].to(device)

        out      = model(features, feature_lengths)
        log_probs = out["ctc_logits"].log_softmax(dim=-1).transpose(0, 1)
        loss     = ctc_loss_fn(log_probs, labels, out["lengths"], label_lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        running_loss += loss.item()
        log_steps    += 1
        pbar.update(1)

        if step % log_every == 0:
            avg  = running_loss / log_steps
            lr   = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=avg, lr=lr)
            log.info(f"step {step:,}/{total_steps:,} | ctc_loss={avg:.4f} | lr={lr:.2e}")
            try:
                import wandb
                wandb.log({"train/ctc_loss": avg, "train/lr": lr}, step=step)
            except Exception:
                pass
            running_loss, log_steps = 0.0, 0

        if step % eval_every == 0:
            metrics = validate(model, val_loader, device, vocab, step, output_dir)
            log.info(f"step {step:,} | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            try:
                import wandb; wandb.log(metrics, step=step)
            except Exception:
                pass

        if step % save_every == 0:
            ckpt_path = output_dir / f"encoder_step{step}.pt"
            torch.save({
                "step":                step,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "vocab":               vocab,
                "encoder_config":      enc_cfg,
            }, ckpt_path)
            log.info(f"Checkpoint → {ckpt_path}")

        if step >= total_steps:
            break

    pbar.close()

    # Final checkpoint
    final_path = output_dir / "encoder_final.pt"
    torch.save({
        "step":                total_steps,
        "model_state_dict":    model.state_dict(),
        "vocab":               vocab,
        "encoder_config":      enc_cfg,
    }, final_path)
    log.info(f"Final checkpoint → {final_path}")

    try:
        import wandb; wandb.finish()
    except Exception:
        pass


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: CTC encoder pretraining")
    parser.add_argument("--config",      required=True)
    parser.add_argument("--resume_from", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
