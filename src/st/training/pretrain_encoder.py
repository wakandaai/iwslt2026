"""
Stage 1: Pretrain speech encoder with CTC loss.

Step-based training with DDP support and deterministic resume.

Resume semantics
----------------
Each checkpoint stores three pieces of "position" state:
  - step               : global optimizer step count
  - epoch              : which sampler epoch we were in when saved
  - batches_into_epoch : how many per-rank batches of that epoch had been
                         consumed at save time

On resume, the sampler is set to `epoch` (rebuilding the identical shuffled
batch list) and the first `batches_into_epoch` batches are skipped. The
remainder of the resumed epoch plays out exactly as it would have without
the restart. Subsequent epochs are full and reseeded by `epoch + 1, epoch +
2, ...` as normal.

Single GPU:
    python -m st.training.pretrain_encoder \
        --config configs/experiment/pretrain_ctc.yaml

Multi-GPU (torchrun):
    torchrun --standalone --nproc_per_node=4 \
        -m st.training.pretrain_encoder \
        --config configs/experiment/pretrain_ctc.yaml

Resume:
    torchrun --standalone --nproc_per_node=4 \
        -m st.training.pretrain_encoder \
        --config configs/experiment/pretrain_ctc.yaml \
        --resume_from runs/stage1_23_lang/encoder_step18000.pt
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from st.data.dataset import SpeechDataset
from core.sampler import DurationBucketSampler
from st.data.vocab import build_vocab_from_index, save_vocab, load_vocab
from st.models.encoder import SpeechEncoder
from core.utils.config import load_config
from st.utils.metrics import compute_wer
from core.utils.schedulers import build_scheduler
from core.utils.ddp_utils import setup_ddp, teardown_ddp, reduce_tensor, barrier

log = logging.getLogger(__name__)


# ============================================================================
# CTC collator
# ============================================================================

def ctc_collate(batch: list[dict], vocab: dict[str, int]) -> dict:
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
        "src_languages":   [b["src_language"] for b in batch],
        "transcripts":     [b["transcript"] for b in batch],
    }


# ============================================================================
# CTC greedy decode helper
# ============================================================================

def ctc_greedy_decode(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    idx_to_char: dict[int, str],
    blank_id: int = 0,
) -> list[str]:
    """Greedy CTC decoding: argmax → collapse repeats → drop blanks."""
    pred_ids = logits.argmax(dim=-1)   # (B, T)
    out: list[str] = []
    for i in range(pred_ids.size(0)):
        seq = pred_ids[i, : lengths[i]].tolist()
        decoded, prev = [], -1
        for tid in seq:
            if tid != blank_id and tid != prev:
                decoded.append(idx_to_char.get(tid, ""))
            prev = tid
        out.append("".join(decoded))
    return out


# ============================================================================
# Val index selection
# ============================================================================

def build_val_generate_indices(
    val_ds: SpeechDataset,
    samples_per_lang: int = 100,
) -> list[int]:
    """Return the first `samples_per_lang` indices per language. Deterministic."""
    lang_indices: dict[str, list[int]] = defaultdict(list)
    for idx in range(len(val_ds)):
        lang = val_ds._src_languages[idx] or "?"
        if len(lang_indices[lang]) < samples_per_lang:
            lang_indices[lang].append(idx)

    indices: list[int] = []
    for lang in sorted(lang_indices):
        n = len(lang_indices[lang])
        log.info(f"  Val decode: {n} samples for language '{lang}'")
        indices.extend(lang_indices[lang])

    log.info(f"Val decode indices: {len(indices)} total ({len(lang_indices)} languages)")
    return indices


# ============================================================================
# Validation  (master rank only)
# ============================================================================

@torch.no_grad()
def validate(
    model: SpeechEncoder,
    val_loader: DataLoader,
    device: torch.device,
    vocab: dict[str, int],
    step: int = 0,
    output_dir: Path | None = None,
) -> dict[str, float]:
    """Single-pass eval on a bounded subset (~val_samples_per_lang × N_langs)."""
    model.eval()
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    idx_to_char = {v: k for k, v in vocab.items()}

    total_loss, n_batches = 0.0, 0
    lang_hyp: dict[str, list[str]] = defaultdict(list)
    lang_ref: dict[str, list[str]] = defaultdict(list)
    all_rows: list[tuple[str, str, str]] = []   # (lang, ref, hyp)

    for batch in tqdm(val_loader, desc=f"val step {step}",
                      unit="batch", dynamic_ncols=True, leave=False):
        features        = batch["features"].to(device)
        feature_lengths = batch["feature_lengths"].to(device)
        labels          = batch["labels"].to(device)
        label_lengths   = batch["label_lengths"].to(device)

        out       = model(features, feature_lengths)
        log_probs = out["ctc_logits"].log_softmax(dim=-1).transpose(0, 1)
        loss      = ctc_loss_fn(log_probs, labels, out["lengths"], label_lengths)
        total_loss += loss.item()
        n_batches  += 1

        hyps = ctc_greedy_decode(
            out["ctc_logits"], out["lengths"], idx_to_char, blank_id=0,
        )
        for hyp, ref, lang in zip(hyps, batch["transcripts"], batch["src_languages"]):
            lang_hyp[lang].append(hyp)
            lang_ref[lang].append(ref)
            all_rows.append((lang, ref, hyp))

    metrics: dict[str, float] = {"val/ctc_loss": total_loss / max(n_batches, 1)}

    if all_rows:
        all_hyp = [h for _, _, h in all_rows]
        all_ref = [r for _, r, _ in all_rows]
        metrics["val/wer"] = compute_wer(all_hyp, all_ref)

        for lang in sorted(lang_hyp):
            lang_wer = compute_wer(lang_hyp[lang], lang_ref[lang])
            metrics[f"val/wer_{lang}"] = lang_wer
            log.info(f"  val WER [{lang}]: {lang_wer:.4f} "
                     f"({len(lang_hyp[lang])} samples)")

        logged: dict[str, int] = defaultdict(int)
        for lang, ref, hyp in all_rows:
            if logged[lang] < 2:
                log.info(f"  [val {lang}] ref: {ref[:80]}")
                log.info(f"  [val {lang}] hyp: {hyp[:80]}")
                logged[lang] += 1

    if output_dir is not None and all_rows:
        from jiwer import wer as _sample_wer
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"val_preds_step{step}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "src_lang", "ref", "hyp", "wer"])
            for i, (lang, ref, hyp) in enumerate(all_rows):
                try:
                    w = _sample_wer(ref, hyp) if ref.strip() else 0.0
                except Exception:
                    w = 1.0
                writer.writerow([i, lang, ref, hyp, f"{w:.4f}"])
        n_empty = sum(1 for _, _, h in all_rows if not h.strip())
        log.info(f"Val preds saved → {csv_path} "
                 f"({len(all_rows)} samples, {n_empty} empty)")

    model.train()
    return metrics


# ============================================================================
# Checkpoint helpers
# ============================================================================

def save_checkpoint(
    model: SpeechEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    epoch: int,
    batches_into_epoch: int,
    vocab: dict[str, int],
    enc_cfg: dict,
    output_dir: Path,
) -> Path:
    """Save a full training checkpoint. Call on master rank only.

    Persists (epoch, batches_into_epoch) alongside (step, model, optim, sched)
    so resume can rebuild the exact sampler position.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"encoder_step{step}.pt"
    torch.save(
        {
            "step":                 step,
            "epoch":                epoch,
            "batches_into_epoch":   batches_into_epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "vocab":                vocab,
            "encoder_config":       enc_cfg,
        },
        ckpt_path,
    )
    log.info(
        f"Checkpoint → {ckpt_path} "
        f"(step={step}, epoch={epoch}, batches_into_epoch={batches_into_epoch})"
    )
    return ckpt_path


def load_checkpoint(
    model: SpeechEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler,
    path: str,
    device: torch.device,
) -> tuple[int, int, int]:
    """Load checkpoint into raw (unwrapped) model.

    Returns:
        (step, epoch, batches_into_epoch). Old checkpoints without epoch /
        batches_into_epoch fields default to (step, 0, 0), which means resume
        will reshuffle from epoch 0 — fine for forward progress, but not
        bit-exact to what the original run would have done.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    step               = ckpt.get("step", 0)
    epoch              = ckpt.get("epoch", 0)
    batches_into_epoch = ckpt.get("batches_into_epoch", 0)

    if "epoch" not in ckpt:
        log.warning(
            f"Checkpoint {path} predates epoch/batches_into_epoch tracking; "
            f"resume will restart sampler at epoch=0 (forward progress only, "
            f"not bit-exact replay)."
        )

    log.info(
        f"Resumed from {path} at step {step} "
        f"(epoch={epoch}, batches_into_epoch={batches_into_epoch})"
    )
    return step, epoch, batches_into_epoch


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

    # ---- Vocab ----
    vocab_path = data_cfg.get("vocab_path")
    if vocab_path and os.path.exists(vocab_path):
        vocab = load_vocab(vocab_path)
    else:
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

        samples_per_lang = train_cfg.get("val_samples_per_lang", 100)
        val_indices = build_val_generate_indices(val_ds, samples_per_lang)
        val_subset = Subset(val_ds, val_indices)
        val_subset.durations = [float(val_ds.durations[i]) for i in val_indices]

        val_sampler = DurationBucketSampler(
            dataset=val_subset,
            target_duration=train_cfg.get("max_batch_duration", 240.0),
            max_batch_size=train_cfg.get("val_batch_size", 16),
            shuffle=False,
            shuffle_buckets=False,
        )
        val_loader = DataLoader(
            val_subset,
            batch_sampler=val_sampler,
            num_workers=train_cfg.get("num_workers", 4),
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=train_cfg.get("num_workers", 4) > 0,
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
    start_step           = 0
    start_epoch          = 0
    start_batch_in_epoch = 0
    if resume_from:
        start_step, start_epoch, start_batch_in_epoch = load_checkpoint(
            model, optimizer, scheduler, resume_from, device,
        )

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
    num_workers = train_cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
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
    global_step      = start_step
    epoch            = start_epoch
    batches_in_epoch = start_batch_in_epoch
    running_loss     = 0.0
    log_steps        = 0

    pbar = tqdm(
        total=total_steps - start_step,
        desc="CTC pretrain",
        unit="step",
        dynamic_ncols=True,
        disable=not master,
    )

    if master:
        log.info(
            f"Training for {total_steps} steps (resuming from step={start_step}, "
            f"epoch={start_epoch}, batches_into_epoch={start_batch_in_epoch})"
        )

    # Outer loop: each iteration consumes one (possibly partial-on-resume) epoch.
    # We do NOT pre-increment `epoch` — the first pass through this loop
    # replays the same epoch number we were saved in, then bumps at the bottom.
    while global_step < total_steps:
        train_sampler.set_epoch(epoch)
        if batches_in_epoch > 0:
            train_sampler.skip(batches_in_epoch)
            if master:
                log.info(
                    f"Resuming epoch {epoch}: skipping {batches_in_epoch} batches "
                    f"(out of {len(train_sampler)} per-rank batches this epoch)"
                )

        for batch in train_loader:
            if global_step >= total_steps:
                break

            features        = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            labels          = batch["labels"].to(device)
            label_lengths   = batch["label_lengths"].to(device)

            out       = model(features, feature_lengths)
            log_probs = out["ctc_logits"].log_softmax(dim=-1).transpose(0, 1)
            loss      = ctc_loss_fn(log_probs, labels, out["lengths"], label_lengths)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            global_step      += 1
            batches_in_epoch += 1
            cur_bs            = features.size(0)
            cur_dur           = feature_lengths.sum().item() * 0.01
            running_loss     += loss.item()
            log_steps        += 1

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
                loss_tensor = loss.detach().clone()
                reduce_tensor(loss_tensor)

                if master:
                    avg = running_loss / log_steps
                    lr  = optimizer.param_groups[0]["lr"]

                    log.info(
                        f"step {global_step:,}/{total_steps:,} | "
                        f"ctc_loss={avg:.4f} | lr={lr:.2e} | "
                        f"gnorm={grad_norm:.2f} | bs={cur_bs} | dur={cur_dur:.0f}s | "
                        f"ep={epoch} | bie={batches_in_epoch}"
                    )
                    if use_wandb:
                        import wandb
                        wandb.log(
                            {
                                "train/ctc_loss":    avg,
                                "train/lr":          lr,
                                "train/epoch":       epoch,
                                "train/grad_norm":   grad_norm,
                                "train/batch_size":  cur_bs * world_size,
                                "train/batch_dur_s": cur_dur * world_size,
                            },
                            step=global_step,
                        )
                    running_loss = 0.0
                    log_steps    = 0

            # ---- Checkpoint (master only) ----
            if master and global_step % save_every == 0:
                save_checkpoint(
                    raw_model, optimizer, scheduler,
                    global_step, epoch, batches_in_epoch,
                    vocab, enc_cfg, output_dir,
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

        # For-loop exhausted naturally → epoch complete. Bump and reset.
        epoch += 1
        batches_in_epoch = 0

    pbar.close()

    # ---- Final checkpoint ----
    if master:
        save_checkpoint(
            raw_model, optimizer, scheduler,
            global_step, epoch, batches_in_epoch,
            vocab, enc_cfg, output_dir,
        )

        final_path = output_dir / "encoder_final.pt"
        torch.save(
            {
                "step":               total_steps,
                "epoch":              epoch,
                "batches_into_epoch": batches_in_epoch,
                "model_state_dict":   raw_model.state_dict(),
                "vocab":              vocab,
                "encoder_config":     enc_cfg,
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