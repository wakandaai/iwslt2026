"""
TTS Stage 1 — train the codec embeddings, depth transformer, speaker projector
and stop head on top of the Aura backbone (frozen by default; LoRA / full
fine-tune available via config).

Mirrors st/training/train_st.py: DDP, DurationBucketSampler (keyed on DAC frame
count, so `max_batch_duration` is a per-batch *frame* budget here), cosine
scheduler, OOM-safe micro-batching, and mid-epoch resume.

Validation is a sharded LOSS pass (depth CE + stop BCE). Audio-quality eval
needs DAC.decode and is done offline via tts/inference/tts_generate.py — we
don't pull the DAC decoder into the training loop.

Single GPU:
    PYTHONPATH=src python -m tts.training.train_tts \
        --config configs/experiment/tts_stage1.yaml

Multi-GPU:
    torchrun --standalone --nproc_per_node=4 \
        -m tts.training.train_tts \
        --config configs/experiment/tts_stage1.yaml

Resume:
    torchrun --standalone --nproc_per_node=4 \
        -m tts.training.train_tts \
        --config configs/experiment/tts_stage1.yaml \
        --resume_from runs/tts_stage1/checkpoint_step10000
"""

from __future__ import annotations

import argparse
import gc
import logging
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from core.aura import AuraLLM
from core.sampler import DurationBucketSampler
from core.utils.config import load_config
from core.utils.ddp_utils import setup_ddp, teardown_ddp, reduce_tensor, barrier
from core.utils.schedulers import build_scheduler
from tts.data import TTSDataset, TTSCollator
from tts.models import SpeechAuraTTS

log = logging.getLogger(__name__)


# ============================================================================
# Build model from config (codec dims come from the cache, via the dataset)
# ============================================================================

def build_model(
    cfg: dict, n_codebooks: int, cardinality: int, speaker_dim: int,
) -> SpeechAuraTTS:
    aura_cfg  = cfg["aura"]
    tts_cfg   = cfg.get("tts", {})
    train_cfg = cfg["training"]

    freeze_llm = not train_cfg.get("unfreeze_llm", False)
    lora_rank  = train_cfg.get("lora_rank", 0)
    if lora_rank > 0 and not freeze_llm:
        log.warning("lora_rank > 0 requires freeze_llm=True — overriding unfreeze_llm.")
        freeze_llm = True

    aura = AuraLLM(
        ckpt_path=aura_cfg["checkpoint"],
        tokenizer_path=aura_cfg["tokenizer"],
        size=aura_cfg.get("size", "1b"),
        freeze=freeze_llm,
        lora_rank=lora_rank,
        lora_alpha=train_cfg.get("lora_alpha", 32),
        lora_targets=train_cfg.get("lora_targets", ["q_proj", "v_proj"]),
    )

    # Config may override the codec dims, but they must match the cache.
    cfg_K    = tts_cfg.get("n_codebooks", n_codebooks)
    cfg_card = tts_cfg.get("cardinality", cardinality)
    if cfg_K != n_codebooks or cfg_card != cardinality:
        raise ValueError(
            f"Config tts.n_codebooks/cardinality ({cfg_K}/{cfg_card}) disagree "
            f"with the DAC cache ({n_codebooks}/{cardinality}).")

    model = SpeechAuraTTS(
        aura=aura,
        n_codebooks=n_codebooks,
        cardinality=cardinality,
        speaker_dim=speaker_dim,
        depth_dim=tts_cfg.get("depth_dim", 1024),
        depth_layers=tts_cfg.get("depth_layers", 6),
        depth_heads=tts_cfg.get("depth_heads", 8),
        depth_dropout=tts_cfg.get("depth_dropout", 0.0),
        codebook_weights=tts_cfg.get("codebook_weights"),
        freeze_llm=freeze_llm,
    )

    if train_cfg.get("gradient_checkpointing", False):
        aura.model.gradient_checkpointing_enable()
        log.info("Gradient checkpointing enabled on Aura LLM")

    return model


# ============================================================================
# Checkpoint load / save  (always operate on raw_model)
# ============================================================================

def load_checkpoint(model, optimizer, scheduler, path: str) -> tuple[int, int, int]:
    import json
    model.load_checkpoint(path)

    opt_path = f"{path}/optimizer.pt"
    if os.path.exists(opt_path):
        optimizer.load_state_dict(
            torch.load(opt_path, map_location="cpu", weights_only=False))
        log.info(f"Loaded optimizer state ← {opt_path}")

    sch_path = f"{path}/scheduler.pt"
    if os.path.exists(sch_path) and scheduler is not None:
        scheduler.load_state_dict(
            torch.load(sch_path, map_location="cpu", weights_only=False))
        log.info(f"Loaded scheduler state ← {sch_path}")

    step = epoch = batches_into_epoch = 0
    meta_path = f"{path}/meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        step               = meta.get("step", 0)
        epoch              = meta.get("epoch", 0)
        batches_into_epoch = meta.get("batches_into_epoch", 0)
        if "epoch" not in meta:
            log.warning(
                f"Checkpoint {path} predates epoch tracking; resume restarts the "
                f"sampler at epoch=0 (forward progress only, not bit-exact replay).")

    log.info(f"Resumed from {path} at step {step} "
             f"(epoch={epoch}, batches_into_epoch={batches_into_epoch})")
    return step, epoch, batches_into_epoch


def save_checkpoint(model, optimizer, scheduler, step, epoch,
                    batches_into_epoch, output_dir: str) -> str:
    import json
    ckpt_dir = os.path.join(output_dir, f"checkpoint_step{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    model.save_checkpoint(ckpt_dir)
    torch.save(optimizer.state_dict(), f"{ckpt_dir}/optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), f"{ckpt_dir}/scheduler.pt")

    meta_path = f"{ckpt_dir}/meta.json"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    meta["step"]               = step
    meta["epoch"]              = epoch
    meta["batches_into_epoch"] = batches_into_epoch
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Checkpoint saved → {ckpt_dir} "
             f"(step={step}, epoch={epoch}, batches_into_epoch={batches_into_epoch})")
    return ckpt_dir


# ============================================================================
# Validation — sharded loss pass (all ranks participate)
# ============================================================================

@torch.no_grad()
def evaluate(model, val_loader, device, rank, world_size, is_ddp) -> dict[str, float]:
    model.eval()
    sums = {"loss": 0.0, "depth_loss": 0.0}
    n = 0
    if val_loader is not None:
        for batch in val_loader:
            if batch is None:
                continue
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                out = model(**batch)
            for k in sums:
                sums[k] += out[k].item()
            n += 1

    # Reduce mean across ranks (sum of per-rank sums / sum of per-rank counts).
    stats = torch.tensor([sums["loss"], sums["depth_loss"], n],
                         dtype=torch.float64, device=device)
    if is_ddp:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_n = max(float(stats[2].item()), 1.0)
    model.train()
    return {
        "loss":       float(stats[0].item()) / total_n,
        "depth_loss": float(stats[1].item()) / total_n,
    }


# ============================================================================
# Training loop
# ============================================================================

def train(cfg: dict, resume_from: str | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    is_ddp, rank, local_rank, world_size, device_str = setup_ddp()
    master = (rank == 0)
    device = torch.device(device_str)

    train_cfg  = cfg["training"]
    data_cfg   = cfg["data"]
    output_dir = train_cfg.get("output_dir", "runs/tts_stage1")
    if master:
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"DDP: {'enabled' if is_ddp else 'disabled'} | "
                 f"rank={rank} | world_size={world_size} | device={device_str}")

    # ---- Data first: the DAC cache is the source of truth for codec dims ----
    languages = data_cfg.get("languages")
    sources   = data_cfg.get("sources")
    seed      = train_cfg.get("seed", 42)

    train_ds = TTSDataset(
        index_path=data_cfg["train_index"],
        dac_dir=data_cfg["dac_cache"],
        spk_dir=data_cfg["spk_cache"],
        split=data_cfg.get("train_split", "train"),
        languages=languages,
        sources=sources,
        min_frames=data_cfg.get("min_frames", 1),
        max_frames=data_cfg.get("max_frames"),
        lowercase=data_cfg.get("lowercase", False),
        seed=seed,
    )
    val_ds = None
    if data_cfg.get("val_index"):
        val_ds = TTSDataset(
            index_path=data_cfg["val_index"],
            dac_dir=data_cfg["dac_cache"],
            spk_dir=data_cfg["spk_cache"],
            split=data_cfg.get("val_split", "dev"),
            languages=languages,
            sources=sources,
            min_frames=data_cfg.get("min_frames", 1),
            max_frames=data_cfg.get("max_frames"),
            lowercase=data_cfg.get("lowercase", False),
            seed=seed,
        )

    # ---- Model ----
    model = build_model(
        cfg, train_ds.n_codebooks, train_ds.cardinality, train_ds.speaker_dim,
    ).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
    raw_model: SpeechAuraTTS = model.module if is_ddp else model

    collator = TTSCollator(
        tokenizer=raw_model.aura.tokenizer,
        max_text_tokens=train_cfg.get("max_text_tokens", 256),
        max_frames=data_cfg.get("max_frames"),
    )

    frame_budget = train_cfg.get("max_batch_frames", 6000)
    train_sampler = DurationBucketSampler(
        dataset=train_ds,
        target_duration=frame_budget,     # frames/batch (durations = frame counts)
        max_batch_size=train_cfg.get("max_batch_size", 64),
        shuffle=True, shuffle_buckets=True,
        rank=rank, world_size=world_size, seed=seed,
    )
    if master:
        log.info(f"Train: {len(train_ds)} samples, {len(train_sampler)} batches/epoch "
                 f"(frame_budget={frame_budget}/rank)")

    num_workers = train_cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, num_workers=num_workers,
        collate_fn=collator, pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = None
    if val_ds is not None:
        val_sampler = DurationBucketSampler(
            dataset=val_ds,
            target_duration=frame_budget,
            max_batch_size=train_cfg.get("max_batch_size", 64),
            shuffle=False, shuffle_buckets=False,
            rank=rank, world_size=world_size, seed=seed,
        )
        val_loader = DataLoader(
            val_ds, batch_sampler=val_sampler, num_workers=num_workers,
            collate_fn=collator, pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    # ---- Optimizer / scheduler ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    lr     = float(train_cfg.get("lr", 2e-4))
    min_lr = float(train_cfg.get("min_lr", 1e-6))
    optimizer = torch.optim.AdamW(
        trainable, lr=lr, weight_decay=train_cfg.get("weight_decay", 0.01))

    max_steps = train_cfg["max_steps"]
    scheduler = build_scheduler(
        name=train_cfg.get("scheduler", "cosine_warmup_restarts"),
        optimizer=optimizer, total_steps=max_steps, max_lr=lr, min_lr=min_lr,
        warmup_steps=train_cfg.get("warmup_steps", 1000),
        first_cycle_steps=train_cfg.get("first_cycle_steps", max_steps),
        gamma=train_cfg.get("gamma", 1.0),
    )

    # ---- Resume ----
    start_step = start_epoch = start_batch_in_epoch = 0
    if resume_from:
        start_step, start_epoch, start_batch_in_epoch = load_checkpoint(
            raw_model, optimizer, scheduler, resume_from)

    # ---- W&B (master only) ----
    wandb_cfg = cfg.get("wandb", {})
    use_wandb = master and wandb_cfg.get("enabled", True)
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=wandb_cfg.get("project", "iwslt2026"),
                entity=wandb_cfg.get("entity"),
                name=wandb_cfg.get("name", os.path.basename(output_dir)),
                config=cfg,
                resume="allow" if start_step > 0 else None,
            )
            log.info(f"W&B: {wandb.run.url}")
        except Exception as e:
            log.warning(f"W&B disabled — wandb.init failed: {type(e).__name__}: {e}")
            use_wandb = False

    # ---- Loop ----
    model.train()
    global_step      = start_step
    epoch            = start_epoch
    batches_in_epoch = start_batch_in_epoch

    grad_accum = train_cfg.get("grad_accum", 1)
    log_every  = train_cfg.get("log_every", 100)
    save_every = train_cfg.get("save_every", 5000)
    eval_every = train_cfg.get("eval_every", 5000)
    oom_cooldown = 0

    running = {"loss": 0.0, "depth_loss": 0.0}
    run_n = 0
    micro_step = 0

    from tqdm import tqdm
    pbar = tqdm(total=max_steps - start_step, desc="Training", unit="step",
                dynamic_ncols=True, disable=not master)
    if master:
        log.info(f"Training for {max_steps} steps (resume step={start_step}, "
                 f"epoch={start_epoch}, batches_into_epoch={start_batch_in_epoch})")
    optimizer.zero_grad()

    # ---- Step-0 validation baseline ----
    if val_ds is not None and start_step == 0:
        if master:
            log.info("Running step-0 validation (untrained baseline)...")
        metrics = evaluate(raw_model, val_loader, device, rank, world_size, is_ddp)
        if master:
            log.info("step 0 val | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if use_wandb:
                import wandb
                wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=0)
        model.train()

    while global_step < max_steps:
        train_sampler.set_epoch(epoch)
        train_ds.set_epoch(epoch)          # vary sibling speaker references per epoch
        if batches_in_epoch > 0:
            train_sampler.skip(batches_in_epoch)
            if master:
                log.info(f"Resuming epoch {epoch}: skipping {batches_in_epoch} batches")

        for batch in train_loader:
            if is_ddp and oom_cooldown > 0:
                ct = torch.tensor(oom_cooldown, dtype=torch.int32, device=device)
                dist.all_reduce(ct, op=dist.ReduceOp.MAX)
                oom_cooldown = int(ct.item())

            if batch is None or oom_cooldown > 0:
                batches_in_epoch += 1
                oom_cooldown = max(0, oom_cooldown - 1)
                continue

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            cur_bs     = batch["codes"].size(0)
            cur_frames = int(batch["code_lengths"].sum().item())

            oom_this_step = False
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=(device.type == "cuda")):
                    out  = model(**batch)
                    loss = out["loss"] / grad_accum
            except torch.cuda.OutOfMemoryError:
                oom_this_step = True
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                gc.collect()

            if is_ddp:
                ot = torch.tensor(int(oom_this_step), dtype=torch.int32, device=device)
                dist.all_reduce(ot, op=dist.ReduceOp.MAX)
                oom_this_step = bool(ot.item())

            if oom_this_step:
                if master:
                    log.warning(f"OOM at step {global_step}: bs={cur_bs} — all ranks skipping")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                oom_cooldown = 3
                micro_step = (micro_step // grad_accum) * grad_accum
                running = {k: 0.0 for k in running}
                run_n = 0
                batches_in_epoch += 1
                continue

            loss.backward()

            for k in running:
                running[k] += out[k].item()
            run_n            += 1
            micro_step       += 1
            batches_in_epoch += 1

            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                if master:
                    pbar.update(1)

            if master:
                pbar.set_postfix(
                    loss=f"{out['loss'].item():.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                    bs=cur_bs, fr=cur_frames, ep=epoch)

            # ---- Logging ----
            if global_step % log_every == 0 and run_n > 0 and micro_step % grad_accum == 0:
                loss_for_log = out["loss"].detach().clone()
                reduce_tensor(loss_for_log)   # all ranks participate
                if master:
                    avg    = {k: v / run_n for k, v in running.items()}
                    cur_lr = optimizer.param_groups[0]["lr"]
                    log.info(
                        f"step {global_step}/{max_steps} | "
                        + " | ".join(f"{k}={v:.4f}" for k, v in avg.items())
                        + f" | lr={cur_lr:.2e} | bs={cur_bs} | frames={cur_frames} | "
                        + f"ep={epoch} | bie={batches_in_epoch}")
                    if use_wandb:
                        import wandb
                        wandb.log(
                            {f"train/{k}": v for k, v in avg.items()}
                            | {"train/lr": cur_lr, "train/epoch": epoch,
                               "train/batch_size": cur_bs * world_size,
                               "train/frames": cur_frames * world_size},
                            step=global_step)
                    running = {k: 0.0 for k in running}
                    run_n = 0

            # ---- Checkpoint ----
            if (global_step > 0 and global_step % save_every == 0
                    and micro_step % grad_accum == 0):
                if master:
                    save_checkpoint(raw_model, optimizer, scheduler,
                                    global_step, epoch, batches_in_epoch, output_dir)
                barrier()

            # ---- Validation ----
            if (val_ds is not None and global_step > 0
                    and global_step % eval_every == 0 and micro_step % grad_accum == 0):
                metrics = evaluate(raw_model, val_loader, device, rank, world_size, is_ddp)
                if master:
                    log.info(f"step {global_step} val | "
                             + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                    if use_wandb:
                        import wandb
                        wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=global_step)
                model.train()

            if global_step >= max_steps:
                break

        epoch += 1
        batches_in_epoch = 0

    pbar.close()

    if master:
        save_checkpoint(raw_model, optimizer, scheduler,
                        global_step, epoch, batches_in_epoch, output_dir)
    barrier()

    if val_ds is not None:
        metrics = evaluate(raw_model, val_loader, device, rank, world_size, is_ddp)
        if master:
            log.info(f"step {global_step} val | "
                     + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if use_wandb:
                import wandb
                wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=global_step)

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
    parser = argparse.ArgumentParser(description="Train SpeechAuraTTS (Stage 1)")
    parser.add_argument("--config",      required=True, help="Experiment YAML config")
    parser.add_argument("--resume_from", default=None,  help="Checkpoint dir to resume from")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
