"""
Stage 1: Pretrain speech encoder with CTC loss.

Step-based training with DDP support and resume. Saved checkpoints include
full model state + optimizer + scheduler + vocab, so Stage 2/3 can load
directly via load_encoder_from_checkpoint().

Single GPU:
    python -m st.training.pretrain_ctc \
        --config configs/experiment/pretrain_ctc.yaml

Multi-GPU (torchrun):
    torchrun --standalone --nproc_per_node=4 \
        -m st.training.pretrain_ctc \
        --config configs/experiment/pretrain_ctc.yaml

Resume:
    torchrun --standalone --nproc_per_node=4 \
        -m st.training.pretrain_ctc \
        --config configs/experiment/pretrain_ctc.yaml \
        --resume_from checkpoints/encoder/encoder_step50000.pt
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from st.data.dataset import SpeechDataset
from st.data.sampler import DurationBucketSampler
from st.data.vocab import build_vocab_from_index, save_vocab, load_vocab
from st.models.encoder import SpeechEncoder, load_encoder_from_checkpoint
from st.utils.config import load_config
from st.utils.metrics import compute_wer
from st.utils.schedulers import build_scheduler
from st.utils.ddp_utils import setup_ddp, teardown_ddp, reduce_tensor, barrier

log = logging.getLogger(__name__)


# ============================================================================
# CTC collator
# ============================================================================

def ctc_collate(batch: list[dict], vocab: dict[str, int]) -> dict[str, torch.Tensor]:
    """Pad mel features and encode transcripts as CTC integer sequences."""
    mel_lens = torch.tensor([b["mel_len"] for b in batch], dtype=torch.long)
    max_mel  = int(mel_lens.max().item())
    mel_pad  = torch.zeros(len(batch), max_mel, 80)
    for i, b in enumerate(batch):
        mel_pad[i, : b["mel_len"]] = b["mel"]

    labels:        list[torch.Tensor] = []
    label_lengths: list[int]          = []
    for b in batch:
        encoded = [vocab[c] for c in b["transcript"] if c in vocab]
        labels.append(torch.tensor(encoded, dtype=torch.long))
        label_lengths.append(len(encoded))

    max_lab = max(label_lengths) if label_lengths else 1
    lab_pad = torch.zeros(len(batch), max_lab, dtype=torch.long)
    for i, lab in enumerate(labels):
        lab_pad[i, : lab.size(0)] = lab

    return {
        "features":        mel_pad,
        "feature_lengths": mel_lens,
        "labels":          lab_pad,
        "label_lengths":   torch.tensor(label_lengths, dtype=torch.long),
    }


# ============================================================================
# Validation  (master rank only)
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
        features        = batch["features"].to(device)
        feature_lengths = batch["feature_lengths"].to(device)
        labels          = batch["labels"].to(device)
        label_lengths   = batch["label_lengths"].to(device)

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

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"val_preds_step{step}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["reference", "prediction"])
            for r, p in zip(refs, preds):
                writer.writerow([r, p])
        n_empty = sum(1 for p in preds if not p.strip())
        log.info(f"Val preds saved → {csv_path} ({n_empty} empty)")

    model.train()
    return {"val/ctc_loss": avg_loss, "val/wer": wer}


# ============================================================================
# Checkpoint helpers
# ============================================================================

def save_checkpoint(
    model: SpeechEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    vocab: dict[str, int],
    enc_cfg: dict,
    output_dir: Path,
) -> Path:
    """Save a full training checkpoint. Call on master rank only."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"encoder_step{step}.pt"
    torch.save(
        {
            "step":                 step,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "vocab":                vocab,
            "encoder_config":       enc_cfg,
        },
        ckpt_path,
    )
    log.info(f"Checkpoint → {ckpt_path}")
    return ckpt_path


def load_checkpoint(
    model: SpeechEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler,
    path: str,
    device: torch.device,
) -> int:
    """Load checkpoint into raw (unwrapped) model. Returns step."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    step = ckpt.get("step", 0)
    log.info(f"Resumed from {path} at step {step}")
    return step


# ============================================================================
# Training
# ============================================================================

def train(cfg: dict, resume_from: str | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- DDP setup ----
    is_ddp, rank, local_rank, world_size, device_str = setup_ddp()
    master = rank == 0
    device = torch.device(device_str)

    if master:
        log.info(
            f"DDP: {'enabled' if is_ddp else 'disabled'} | "
            f"rank={rank} | world_size={world_size} | device={device_str}"
        )

    enc_cfg   = cfg["encoder"]
    train_cfg = cfg["training"]
    data_cfg  = cfg["data"]
    output_dir = Path(train_cfg.get("output_dir", "checkpoints/encoder"))

    if master:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Vocab (all ranks need it for the collator) ----
    vocab_path = data_cfg.get("vocab_path")
    if vocab_path and os.path.exists(vocab_path):
        vocab = load_vocab(vocab_path)
    else:
        # Only master builds and saves; then all ranks load
        if master:
            vocab = build_vocab_from_index(
                data_cfg["train_index"],
                text_column="transcript",
                split=data_cfg.get("train_split", "train"),
                languages=data_cfg.get("languages"),
                lowercase=data_cfg.get("lowercase", False),
            )
            if vocab_path:
                save_vocab(vocab, vocab_path)
        # Non-master ranks wait until master has written the file
        barrier()
        if not master:
            vocab = load_vocab(vocab_path) if vocab_path else build_vocab_from_index(
                data_cfg["train_index"],
                text_column="transcript",
                split=data_cfg.get("train_split", "train"),
                languages=data_cfg.get("languages"),
                lowercase=data_cfg.get("lowercase", False),
            )

    if master:
        log.info(f"Vocab: {len(vocab)} tokens")

    # ---- Datasets ----
    train_ds = SpeechDataset(
        index_path=data_cfg["train_index"],
        split=data_cfg.get("train_split", "train"),
        task="asr",
        languages=data_cfg.get("languages"),
        max_duration=data_cfg.get("max_duration", 30.0),
        min_duration=data_cfg.get("min_duration", 0.1),
        lowercase=data_cfg.get("lowercase", False),
    )

    # Validation dataset and loader: master rank only
    val_loader = None
    if master and data_cfg.get("val_index"):
        val_ds = SpeechDataset(
            index_path=data_cfg["val_index"],
            split=data_cfg.get("val_split", "dev"),
            task="asr",
            languages=data_cfg.get("languages"),
            max_duration=data_cfg.get("max_duration", 30.0),
            lowercase=data_cfg.get("lowercase", False),
        )

        def collate_fn(batch):
            return ctc_collate(batch, vocab)

        val_sampler = DurationBucketSampler(
            dataset=val_ds,
            target_duration=train_cfg.get("max_batch_duration", 240.0),
            max_batch_size=train_cfg.get("val_batch_size", 16),
            shuffle=False,
            shuffle_buckets=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=val_sampler,
            num_workers=train_cfg.get("num_workers", 4),
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # ---- Model ----
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

    if master:
        log.info(f"Encoder: {sum(p.numel() for p in model.parameters()):,} params")

    # ---- Optimizer + Scheduler ----
    total_steps = train_cfg["total_steps"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    # Flat scheduler keys — same layout as stage 2/3/4/5 configs.
    # `scheduler` key is just the name string (e.g. "cosine_warmup_restarts").
    scheduler = build_scheduler(
        name=train_cfg.get("scheduler", "cosine_warmup_restarts"),
        optimizer=optimizer,
        total_steps=total_steps,
        max_lr=float(train_cfg.get("lr", 1e-4)),
        min_lr=float(train_cfg.get("min_lr", 1e-6)),
        warmup_steps=train_cfg.get("warmup_steps", 2000),
        first_cycle_steps=train_cfg.get("first_cycle_steps", total_steps),
        gamma=train_cfg.get("gamma", 1.0),
    )

    # ---- Resume (all ranks load so state is identical) ----
    start_step = 0
    if resume_from:
        start_step = load_checkpoint(model, optimizer, scheduler, resume_from, device)

    # ---- Wrap with DDP after loading weights ----
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    raw_model: SpeechEncoder = model.module if is_ddp else model

    # ---- Train sampler + loader ----
    def collate_fn(batch):
        return ctc_collate(batch, vocab)

    train_sampler = DurationBucketSampler(
        dataset=train_ds,
        target_duration=train_cfg.get("max_batch_duration", 240.0),
        max_batch_size=train_cfg.get("max_batch_size", 32),
        shuffle=True,
        shuffle_buckets=True,
        rank=rank,
        world_size=world_size,
        seed=42,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    if master:
        log.info(
            f"Train: {len(train_ds)} samples, {len(train_sampler)} batches/epoch "
            f"(world_size={world_size}, effective_batch ≈ "
            f"{train_cfg.get('max_batch_duration', 240.0) * world_size:.0f}s/step)"
        )

    # ---- W&B (master only) ----
    use_wandb = cfg.get("wandb", {}).get("enabled", False) and master
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg["wandb"].get("project", "iwslt2026"),
                name=cfg["wandb"].get("name", "pretrain-ctc"),
                config=cfg,
                resume="allow" if start_step > 0 else None,
            )
            log.info(f"W&B: {wandb.run.url}")
        except ImportError:
            use_wandb = False

    # ---- Training loop ----
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    log_every   = train_cfg.get("log_every_steps", 200)
    save_every  = train_cfg.get("save_every_steps", 10000)
    eval_every  = train_cfg.get("eval_every_steps", 5000)

    model.train()
    global_step   = start_step
    epoch         = 0
    running_loss  = 0.0
    log_steps     = 0

    pbar = tqdm(
        total=total_steps - start_step,
        desc="CTC pretrain",
        unit="step",
        dynamic_ncols=True,
        disable=not master,
    )

    if master:
        log.info(f"Training for {total_steps} steps (resuming from {start_step})")

    while global_step < total_steps:
        epoch += 1
        train_sampler.set_epoch(epoch)

        for batch in train_loader:
            if global_step >= total_steps:
                break

            features        = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            labels          = batch["labels"].to(device)
            label_lengths   = batch["label_lengths"].to(device)

            out      = model(features, feature_lengths)
            log_probs = out["ctc_logits"].log_softmax(dim=-1).transpose(0, 1)
            loss     = ctc_loss_fn(log_probs, labels, out["lengths"], label_lengths)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            global_step   += 1
            cur_bs         = features.size(0)
            cur_dur        = feature_lengths.sum().item() * 0.01  # frames → seconds (10ms hop)
            running_loss  += loss.item()
            log_steps     += 1

            if master:
                pbar.update(1)
                pbar.set_postfix(
                    loss=f"{loss.item():.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                    gnorm=f"{grad_norm:.2f}",
                    ep=epoch,
                )

            # ---- Logging ----
            if global_step % log_every == 0:
                # Reduce loss across ranks so the logged value is the true mean
                loss_tensor = loss.detach().clone()
                reduce_tensor(loss_tensor)   # all ranks participate; non-master discards

                if master:
                    avg = running_loss / log_steps
                    lr  = optimizer.param_groups[0]["lr"]

                    log.info(
                        f"step {global_step:,}/{total_steps:,} | "
                        f"ctc_loss={avg:.4f} | lr={lr:.2e} | "
                        f"gnorm={grad_norm:.2f} | bs={cur_bs} | dur={cur_dur:.0f}s"
                    )
                    if use_wandb:
                        import wandb
                        wandb.log(
                            {
                                "train/ctc_loss":          avg,
                                "train/lr":                lr,
                                "train/epoch":             epoch,
                                "train/grad_norm":         grad_norm,
                                "train/batch_size":        cur_bs * world_size,
                                "train/batch_dur_s":       cur_dur * world_size,
                            },
                            step=global_step,
                        )
                    running_loss = 0.0
                    log_steps    = 0

            # ---- Checkpoint (master only) ----
            if master and global_step % save_every == 0:
                save_checkpoint(
                    raw_model, optimizer, scheduler,
                    global_step, vocab, enc_cfg, output_dir,
                )

            # ---- Validation (master only) ----
            if master and val_loader is not None and global_step % eval_every == 0:
                metrics = validate(
                    raw_model, val_loader, device, vocab,
                    step=global_step, output_dir=output_dir,
                )
                log.info(
                    f"step {global_step:,} | "
                    + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                )
                if use_wandb:
                    import wandb
                    wandb.log(metrics, step=global_step)
                model.train()

    pbar.close()

    # ---- Final checkpoint ----
    if master:
        final_path = output_dir / "encoder_final.pt"
        torch.save(
            {
                "step":             total_steps,
                "model_state_dict": raw_model.state_dict(),
                "vocab":            vocab,
                "encoder_config":   enc_cfg,
            },
            final_path,
        )
        log.info(f"Final checkpoint → {final_path}")

        if val_loader is not None:
            metrics = validate(
                raw_model, val_loader, device, vocab,
                step=global_step, output_dir=output_dir,
            )
            log.info("Final val | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if use_wandb:
                import wandb
                wandb.log(metrics, step=global_step)

        if use_wandb:
            import wandb
            wandb.finish()

    barrier()
    teardown_ddp()

    if master:
        log.info("Training complete.")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: CTC encoder pretraining")
    parser.add_argument("--config",      required=True,  help="Experiment YAML config")
    parser.add_argument("--resume_from", default=None,   help="Checkpoint .pt to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume_from=args.resume_from)


if __name__ == "__main__":
    main()