"""
Stage 1: Fine-tune omniASR_CTC_1B on our 22 languages via standalone CTC loss.

Unlike st.training.pretrain_ctc (which trains the old Conformer-18 from a
random/small init against a char-level CTC vocab built from the training
data), this fine-tunes Meta's *already ASR-pretrained* omniASR_CTC_1B
encoder — encoder body + its own 9812-piece SentencePiece CTC head are both
trainable, everything else (projector, CTC-compressor, Aura-1B LLM) simply
never gets loaded in this script at all, so there's nothing to freeze.

MUST run under the isolated torch==2.8.0+cu128 env (.envs/omniasr_extract) —
fairseq2n requires torch 2.8, incompatible with the main training env
(Aura_base/env, torch==2.6.0+cu124) used by st.training.train_st.

Usage (single GPU):
    .envs/omniasr_extract/bin/python -m st.training.pretrain_omniasr_ctc \
        --config configs/experiment/stage1_omniasr_ctc_22lang.yaml
    .envs/omniasr_extract/bin/python -m st.training.pretrain_omniasr_ctc \
        --config configs/experiment/stage1_omniasr_ctc_22lang.yaml \
        --resume_from runs/stage1_omniasr_ctc/checkpoint_step5000.pt

Usage (multi-GPU, DDP via torchrun):
    torchrun --standalone --nproc_per_node=2 -m st.training.pretrain_omniasr_ctc \
        --config configs/experiment/stage1_omniasr_ctc_22lang_ddp.yaml
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from st.data import (
    RawAudioDataset, CTCRawAudioCollator,
    DurationBucketSampler, WeightedLanguageSampler,
)
from st.models.omniasr_encoder import build_omniasr_encoder_from_config
from st.utils.config import load_config
from st.utils.ddp_utils import setup_ddp, teardown_ddp, reduce_tensor, barrier
from st.utils.metrics import compute_wer
from st.utils.schedulers import build_scheduler

log = logging.getLogger(__name__)

CTC_BLANK_ID = 0  # matches CTCCompressor's default and OmniASREncoder's ctc_logits convention


def load_sp_tokenizer(path: str):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp


def build_val_indices(val_ds, samples_per_lang: int) -> list[int]:
    """First `samples_per_lang` indices per language, in dataset order —
    deterministic and stable across resumes/eval calls (mirrors
    train_st.py's build_val_generate_indices)."""
    lang_indices: dict[str, list[int]] = defaultdict(list)
    for idx, entry in enumerate(val_ds.entries):
        lang = entry.get("language") or entry.get("src_language") or "?"
        if len(lang_indices[lang]) < samples_per_lang:
            lang_indices[lang].append(idx)

    indices: list[int] = []
    for lang in sorted(lang_indices):
        indices.extend(lang_indices[lang])
    log.info(f"Val indices: {len(indices)} total across {len(lang_indices)} languages")
    return indices


def greedy_ctc_decode(sp, ids: list[int]) -> str:
    """Collapse repeats, drop blanks, decode remaining piece ids to text."""
    decoded, prev = [], -1
    for tid in ids:
        if tid != CTC_BLANK_ID and tid != prev:
            decoded.append(tid)
        prev = tid
    if not decoded:
        return ""
    return sp.decode(decoded)


# ============================================================================
# Validation (master rank only — CTC greedy decode is cheap enough that
# sharding it across ranks, like train_st.py does for expensive autoregressive
# generation, isn't worth the extra gather/broadcast complexity here)
# ============================================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    sp,
    step: int = 0,
    output_dir: Path | None = None,
) -> dict[str, float]:
    model.eval()
    ctc_loss_fn = nn.CTCLoss(blank=CTC_BLANK_ID, reduction="mean", zero_infinity=True)

    total_loss, n_batches = 0.0, 0
    preds: list[str] = []
    refs:  list[str] = []
    langs: list[str] = []

    for batch in loader:
        if batch is None:
            continue
        features       = batch["audio_features"].to(device)
        feature_lengths = batch["audio_lengths"].to(device)
        labels         = batch["ctc_labels"].to(device)
        label_lengths  = batch["ctc_label_lengths"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = model(features, feature_lengths)
            log_probs = out["ctc_logits"].float().log_softmax(dim=-1).transpose(0, 1)
            loss = ctc_loss_fn(log_probs, labels, out["lengths"], label_lengths)

        total_loss += loss.item()
        n_batches  += 1

        pred_ids = out["ctc_logits"].argmax(dim=-1)
        for i in range(pred_ids.size(0)):
            seq = pred_ids[i, : out["lengths"][i]].tolist()
            preds.append(greedy_ctc_decode(sp, seq))
            ref_seq = labels[i, : label_lengths[i]].tolist()
            refs.append(sp.decode(ref_seq))
        langs.extend(batch["language"])

    avg_loss = total_loss / max(n_batches, 1)
    results = {"val/ctc_loss": avg_loss, "val/wer": compute_wer(preds, refs) if refs else 0.0}

    # Per-language WER
    lang_preds: dict[str, list[str]] = defaultdict(list)
    lang_refs:  dict[str, list[str]] = defaultdict(list)
    for r, p, lang in zip(refs, preds, langs):
        lang_preds[lang].append(p)
        lang_refs[lang].append(r)
    for lang in sorted(lang_preds):
        lang_wer = compute_wer(lang_preds[lang], lang_refs[lang])
        results[f"val/wer_{lang}"] = lang_wer
        log.info(f"  val WER [{lang}]: {lang_wer:.4f} ({len(lang_preds[lang])} samples)")

    if output_dir is not None and preds:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"val_preds_step{step}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["language", "reference", "prediction"])
            for lang, r, p in zip(langs, refs, preds):
                writer.writerow([lang, r, p])
        n_empty = sum(1 for p in preds if not p.strip())
        log.info(f"Val preds saved → {csv_path} ({n_empty} empty)")

    model.train()
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ============================================================================
# Training
# ============================================================================

def save_checkpoint(model, optimizer, scheduler, step: int, output_dir: Path) -> Path:
    """Must only be called on master rank. `model` must be the raw (unwrapped)
    module — DDP-wrapped state_dicts have a "module." prefix that a
    single-GPU resume wouldn't recognize."""
    ckpt_path = output_dir / f"encoder_step{step}.pt"
    torch.save({
        "step":                step,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }, ckpt_path)
    log.info(f"Checkpoint → {ckpt_path}")
    return ckpt_path


def train(cfg: dict, resume_from: str | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- DDP setup ---
    is_ddp, rank, local_rank, world_size, device_str = setup_ddp()
    master = (rank == 0)
    device = torch.device(device_str)

    if master:
        log.info(f"DDP: {'enabled' if is_ddp else 'disabled'} | "
                 f"rank={rank} | world_size={world_size} | device={device_str}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if master:
            log.info(f"CUDA device: {torch.cuda.get_device_name(local_rank)} — TF32 enabled")

    enc_cfg   = cfg["encoder"]
    train_cfg = cfg["training"]
    data_cfg  = cfg["data"]
    output_dir = Path(train_cfg.get("output_dir", "runs/stage1_omniasr_ctc"))
    if master:
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- Tokenizer (CTC target space — omniASR's own 9812-piece SentencePiece vocab) ---
    sp = load_sp_tokenizer(data_cfg["sp_tokenizer_path"])
    if master:
        log.info(f"Loaded SentencePiece tokenizer: {sp.get_piece_size()} pieces")

    # --- Data (identical on every rank — same index, same filters) ---
    languages = data_cfg.get("languages")  # None = all languages in the index
    lowercase = data_cfg.get("lowercase", True)

    train_ds = RawAudioDataset(
        index_path=data_cfg["train_index"],
        split=data_cfg.get("train_split", "train"),
        languages=languages,
        max_duration=data_cfg.get("max_duration", 20.0),
        min_duration=data_cfg.get("min_duration", 0.5),
        lowercase=lowercase,
    )
    val_ds = RawAudioDataset(
        index_path=data_cfg.get("val_index", data_cfg["train_index"]),
        split=data_cfg.get("val_split", "dev"),
        languages=languages,
        max_duration=data_cfg.get("max_duration", 20.0),
        min_duration=data_cfg.get("min_duration", 0.5),
        lowercase=lowercase,
    )

    collate = CTCRawAudioCollator(sp_tokenizer=sp, max_target_tokens=data_cfg.get("max_target_tokens", 400))

    # Synchronized samplers: shared seed across ranks builds an identical
    # global batch sequence, then each rank slices its own [rank::world_size]
    # portion — no inter-rank communication needed for this to stay in sync.
    sampling_cfg = train_cfg.get("sampling", {})
    if sampling_cfg.get("strategy") == "weighted_language":
        train_sampler = WeightedLanguageSampler(
            dataset=train_ds,
            beta_language=sampling_cfg["beta_language"],
            target_duration=train_cfg.get("max_batch_duration", 40.0),
            max_batch_size=train_cfg.get("max_batch_size", 4),
            num_batches=sampling_cfg.get("num_batches", 200_000),
            rank=rank, world_size=world_size,
            seed=sampling_cfg.get("seed", 42),
        )
        if master:
            log.info(
                f"Train: {len(train_ds)} samples, WeightedLanguageSampler "
                f"(beta_language={sampling_cfg['beta_language']}), "
                f"{len(train_sampler.partitions)} languages, "
                f"{len(train_sampler)} per-rank batches (world_size={world_size})"
            )
            for lang in sorted(train_sampler.partition_weight, key=lambda l: -train_sampler.partition_weight[l]):
                log.info(f"  sampling weight [{lang}]: {train_sampler.partition_weight[lang]:.4f}")
    else:
        train_sampler = DurationBucketSampler(
            dataset=train_ds,
            target_duration=train_cfg.get("max_batch_duration", 40.0),
            max_batch_size=train_cfg.get("max_batch_size", 4),
            shuffle=True,
            shuffle_buckets=True,
            rank=rank, world_size=world_size,
            seed=42,
        )
        if master:
            log.info(f"Train: {len(train_ds)} samples, {len(train_sampler)} per-rank "
                     f"batches/epoch (world_size={world_size})")

    num_workers = train_cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate, pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # Cap validation to N samples per language — the full dev split would take
    # hours to iterate every eval_every steps, and since index rows are
    # grouped by language/source (not shuffled), naively truncating the
    # loader would only ever cover the 1-2 languages that happen to sit first
    # in file order. Build a deterministic, language-balanced subset instead.
    # Validation itself only runs on master (see note on `validate()`).
    val_samples_per_lang = train_cfg.get("val_samples_per_lang", 30)
    val_indices = build_val_indices(val_ds, val_samples_per_lang)
    val_loader = None
    if master:
        val_subset = torch.utils.data.Subset(val_ds, val_indices)
        val_loader = DataLoader(
            val_subset, batch_size=train_cfg.get("val_batch_size", 4),
            shuffle=False, num_workers=num_workers,
            collate_fn=collate, pin_memory=True,
        )
        log.info(f"Val: {len(val_indices)} samples ({val_samples_per_lang}/lang cap), "
                 f"{len(val_loader)} batches")

    # --- Model — omniASR_CTC_1B, trainable, nothing else loaded ---
    # Build + unfreeze identically on every rank BEFORE DDP-wrapping — DDP
    # doesn't proxy arbitrary custom methods like .unfreeze(), only forward().
    raw_model = build_omniasr_encoder_from_config(enc_cfg).to(device)
    raw_model.unfreeze()
    n_trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in raw_model.parameters())
    if master:
        log.info(f"OmniASREncoder: {n_trainable:,} / {n_total:,} params trainable")

    model = DDP(raw_model, device_ids=[local_rank]) if is_ddp else raw_model

    # --- Optimizer + Scheduler ---
    trainable = [p for p in model.parameters() if p.requires_grad]
    lr     = float(train_cfg.get("lr", 1e-5))
    min_lr = float(train_cfg.get("min_lr", 1e-7))
    total_steps = train_cfg["max_steps"]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=train_cfg.get("weight_decay", 0.01))
    scheduler = build_scheduler(
        name=train_cfg.get("scheduler", "cosine_warmup_restarts"),
        optimizer=optimizer,
        total_steps=total_steps,
        max_lr=lr,
        min_lr=min_lr,
        warmup_steps=train_cfg.get("warmup_steps", 500),
        first_cycle_steps=train_cfg.get("first_cycle_steps", total_steps),
        cycle_mult=train_cfg.get("cycle_mult", 1.0),
        gamma=train_cfg.get("gamma", 1.0),
    )

    # --- Resume (load on all ranks so weights + optimizer state are identical
    # everywhere — resume_from checkpoints always hold a raw, unwrapped
    # state_dict, so this loads into raw_model regardless of is_ddp). ---
    start_step = 0
    if resume_from:
        # map_location="cpu" (not `device`): loading straight to GPU put the
        # whole checkpoint (weights + AdamW's m/v buffers, ~11.7GB) on-device
        # at once, on every rank, alongside the already-constructed model/
        # optimizer -- a transient ~27GB peak before the first training step
        # even ran. Fine on H100-80 (never caught there); OOM'd immediately
        # on V100-32. load_state_dict() copies CPU tensors to each param's
        # existing device itself, so staging on CPU changes nothing else.
        ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt.get("step", 0)
        if master:
            log.info(f"Resumed from {resume_from} at step {start_step}")
        del ckpt
        torch.cuda.empty_cache()

    # --- W&B — master only ---
    use_wandb = not train_cfg.get("no_wandb", False)
    if master and use_wandb:
        try:
            import wandb
            wandb.init(
                project=train_cfg.get("wandb_project", "iwslt2026"),
                entity=train_cfg.get("wandb_entity"),
                name=train_cfg.get("wandb_run_name", output_dir.name),
                config=cfg,
                resume="allow" if start_step > 0 else None,
            )
            log.info(f"W&B: {wandb.run.url}")
        except ImportError:
            use_wandb = False
    elif not master:
        use_wandb = False

    # --- Training loop ---
    ctc_loss_fn  = nn.CTCLoss(blank=CTC_BLANK_ID, reduction="mean", zero_infinity=True)
    grad_accum   = train_cfg.get("grad_accum", 8)
    log_every    = train_cfg.get("log_every", 100)
    save_every   = train_cfg.get("save_every", 2000)
    eval_every   = train_cfg.get("eval_every", 2000)
    oom_cooldown = 0

    model.train()
    global_step = start_step
    epoch = 0
    running_loss, run_n, micro_step = 0.0, 0, 0
    pbar = tqdm(total=total_steps - start_step, desc="Stage1 CTC", unit="step",
                dynamic_ncols=True, disable=not master)
    optimizer.zero_grad()

    if master:
        log.info(f"Training for {total_steps} steps (resuming from {start_step})")

    while global_step < total_steps:
        epoch += 1
        for batch in train_loader:
            # A batch was drawn from the sampler — the skip decision (None
            # batch, or OOM cooldown) is collective: it must be identical on
            # every rank, since disagreeing here would desync micro_step
            # across ranks and break the no_sync() grad-accum agreement below,
            # hanging the next all-reduce.
            skip = int(batch is None or oom_cooldown > 0)
            if is_ddp:
                flags = torch.tensor([oom_cooldown, skip], dtype=torch.int32, device=device)
                dist.all_reduce(flags, op=dist.ReduceOp.MAX)
                oom_cooldown, skip = int(flags[0].item()), int(flags[1].item())

            if skip:
                oom_cooldown = max(0, oom_cooldown - 1)
                continue

            features        = batch["audio_features"].to(device)
            feature_lengths = batch["audio_lengths"].to(device)
            labels          = batch["ctc_labels"].to(device)
            label_lengths   = batch["ctc_label_lengths"].to(device)
            cur_bs  = features.size(0)
            cur_dur = feature_lengths.sum().item() / 16000

            oom_this_step = False
            out = log_probs = loss = None
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    out = model(features, feature_lengths)
                    log_probs = out["ctc_logits"].float().log_softmax(dim=-1).transpose(0, 1)
                    loss = ctc_loss_fn(log_probs, labels, out["lengths"], label_lengths)
            except torch.cuda.OutOfMemoryError:
                oom_this_step = True
                # See the long comment historically here: out/log_probs/loss
                # must be reassigned (not just left to the except's implicit
                # scope) or the leaked forward graph prevents empty_cache()
                # from reclaiming memory, causing every retry at this step to
                # OOM again regardless of batch size.
                out = log_probs = loss = None
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()

            # Sync OOM across ranks BEFORE backward — same reasoning as the
            # skip/cooldown sync above.
            if is_ddp:
                oom_tensor = torch.tensor(int(oom_this_step), dtype=torch.int32, device=device)
                dist.all_reduce(oom_tensor, op=dist.ReduceOp.MAX)
                oom_this_step = bool(oom_tensor.item())

            if oom_this_step:
                if master:
                    log.warning(f"OOM at step {global_step}: bs={cur_bs} dur={cur_dur:.0f}s — all ranks skipping")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                oom_cooldown = 3
                micro_step = (micro_step // grad_accum) * grad_accum
                running_loss, run_n = 0.0, 0
                continue

            # Under DDP, only the micro-batch that CLOSES an accumulation
            # window should all-reduce gradients — no_sync() suppresses the
            # reduction on the others, so we pay 1 all-reduce per optimizer
            # step instead of grad_accum of them. All ranks agree on
            # micro_step (skip/OOM are collective above), so they agree here.
            closes_window = (micro_step + 1) % grad_accum == 0
            sync_ctx = model.no_sync() if (is_ddp and not closes_window) else nullcontext()
            with sync_ctx:
                (loss / grad_accum).backward()

            running_loss += loss.item()
            run_n += 1
            micro_step += 1

            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                if scheduler:
                    scheduler.step()
                global_step += 1
                if master:
                    pbar.update(1)

            if master:
                pbar.set_postfix(
                    loss=f"{loss.item():.3f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                    bs=cur_bs, dur=f"{cur_dur:.0f}s", ep=epoch,
                )

            # ---- Logging (master only, loss averaged across ranks) ----
            if global_step % log_every == 0 and run_n > 0 and micro_step % grad_accum == 0:
                loss_for_log = loss.detach().clone()
                reduce_tensor(loss_for_log)  # all ranks participate in the all-reduce

                if master:
                    avg = running_loss / run_n
                    cur_lr = optimizer.param_groups[0]["lr"]
                    log.info(f"step {global_step}/{total_steps} | ctc_loss={avg:.4f} | "
                             f"lr={cur_lr:.2e} | bs={cur_bs} | dur={cur_dur:.0f}s | ep={epoch}")
                    if use_wandb:
                        import wandb
                        wandb.log({"train/ctc_loss": avg, "train/lr": cur_lr, "train/epoch": epoch,
                                   "train/batch_size": cur_bs * world_size,
                                   "train/batch_dur": cur_dur * world_size}, step=global_step)
                running_loss, run_n = 0.0, 0

            # ---- Checkpoint (master only; all ranks wait so no one races ahead) ----
            if global_step % save_every == 0 and micro_step % grad_accum == 0:
                if master:
                    save_checkpoint(raw_model, optimizer, scheduler, global_step, output_dir)
                barrier()

            # ---- Validation (master only — see note on validate()) ----
            if global_step % eval_every == 0 and micro_step % grad_accum == 0:
                if master:
                    torch.cuda.empty_cache()
                    metrics = validate(raw_model, val_loader, device, sp, global_step, output_dir)
                    log.info(f"step {global_step} val | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                    if use_wandb:
                        import wandb
                        wandb.log(metrics, step=global_step)
                barrier()
                model.train()

            if global_step >= total_steps:
                break

    pbar.close()
    if master:
        save_checkpoint(raw_model, optimizer, scheduler, global_step, output_dir)
    barrier()

    if master:
        metrics = validate(raw_model, val_loader, device, sp, global_step, output_dir)
        log.info("Final val | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if use_wandb:
            import wandb
            wandb.log(metrics, step=global_step)
            wandb.finish()

    barrier()
    teardown_ddp()

    if master:
        log.info("Training complete.")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: omniASR_CTC_1B encoder fine-tune (CTC-only)")
    parser.add_argument("--config",      required=True)
    parser.add_argument("--resume_from", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
