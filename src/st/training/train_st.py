"""
Stage 2 / 3 / 4 — Speech Translation training with DDP support.

Stage 2: Freeze encoder + LLM, train projector only.
Stage 3: Freeze encoder + LLM, train projector + LoRA.
Stage 4: Train everything (encoder + projector + full LLM).

Controlled entirely by the experiment YAML — no code changes needed to
switch stages.

Resume semantics
----------------
meta.json now persists three position fields:
  - step               : global optimizer-step count
  - epoch              : which sampler epoch was active when the checkpoint
                         was saved
  - batches_into_epoch : per-rank micro-batches consumed in that epoch at
                         save time (counts micro-batches, not optimizer
                         steps — this is the granularity at which the
                         sampler advances)

On resume, set_epoch(epoch) rebuilds the identical shuffled batch list, then
skip(batches_into_epoch) advances past already-consumed batches. OOM-skipped
batches DO count toward batches_into_epoch (they were drawn from the sampler
even though their gradients were thrown away).

Single GPU:
    PYTHONPATH=src python -m st.training.train_st \
        --config configs/experiment/stage3.yaml

Multi-GPU (torchrun):
    torchrun --standalone --nproc_per_node=4 \
        -m st.training.train_st \
        --config configs/experiment/stage3.yaml

Resume:
    torchrun --standalone --nproc_per_node=4 \
        -m st.training.train_st \
        --config configs/experiment/stage3.yaml \
        --resume_from runs/stage3/checkpoint_step10000
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from st.data import SpeechDataset, AuraCollator, DurationBucketSampler
from st.models import (
    SpeechAura, AuraLLM,
    load_encoder_from_checkpoint,
    build_ctc_compressor,
)
from st.utils.config import load_config
from st.utils.schedulers import build_scheduler
from st.utils.metrics import compute_wer, compute_bleu, compute_chrf
from st.utils.ddp_utils import setup_ddp, teardown_ddp, reduce_tensor, barrier
import torch.distributed as dist

log = logging.getLogger(__name__)


# ============================================================================
# Build model from config
# ============================================================================

def build_model(cfg: dict) -> SpeechAura:
    enc_cfg      = cfg["encoder"]
    aura_cfg     = cfg["aura"]
    proj_cfg     = cfg.get("projector", {"type": "mlp"})
    ctc_comp_cfg = cfg.get("ctc_compress", None)
    train_cfg    = cfg["training"]

    # Encoder
    encoder = load_encoder_from_checkpoint(
        config=enc_cfg,
        checkpoint_path=enc_cfg.get("checkpoint"),
        vocab_size=enc_cfg.get("vocab_size"),
        strict=False,
    )

    # Aura LLM
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

    model = SpeechAura(
        encoder=encoder,
        aura=aura,
        projector_cfg=proj_cfg,
        ctc_compress_cfg=ctc_comp_cfg,
        ctc_weight=train_cfg.get("ctc_weight", 0.0),
        freeze_encoder=not train_cfg.get("unfreeze_encoder", False),
        freeze_llm=freeze_llm,
    )

    # Gradient checkpointing on the LLM — trades ~30% throughput for
    # ~30-50% lower activation memory. Useful for stage 4 (full fine-tune).
    if train_cfg.get("gradient_checkpointing", False):
        aura.model.gradient_checkpointing_enable()
        log.info("Gradient checkpointing enabled on Aura LLM")

    # Load projector checkpoint if specified
    proj_ckpt = train_cfg.get("projector_checkpoint")
    if proj_ckpt:
        proj_path = f"{proj_ckpt}/projector.pt"
        state = torch.load(proj_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.projector.load_state_dict(state, strict=True)
        if missing:
            log.warning(f"Projector checkpoint missing keys: {missing}")
        if unexpected:
            log.warning(f"Projector checkpoint unexpected keys: {unexpected}")
        log.info(f"Projector loaded ← {proj_path}")
    else:
        log.info("Projector: no checkpoint specified, training from scratch")

    return model


# ============================================================================
# Checkpoint load / save  (always operate on raw_model, never DDP wrapper)
# ============================================================================

def load_checkpoint(
    model: SpeechAura,
    optimizer: torch.optim.Optimizer,
    scheduler,
    path: str,
) -> tuple[int, int, int]:
    """Load checkpoint directory into raw (unwrapped) model.

    Returns:
        (step, epoch, batches_into_epoch). Old checkpoints without epoch /
        batches_into_epoch fields default to (step, 0, 0), which means resume
        will reshuffle from epoch 0 — forward progress is fine but not
        bit-exact to what the original run would have done.
    """
    import json

    model.load_checkpoint(path)

    opt_path = f"{path}/optimizer.pt"
    if os.path.exists(opt_path):
        optimizer.load_state_dict(
            torch.load(opt_path, map_location="cpu", weights_only=False)
        )
        log.info(f"Loaded optimizer state ← {opt_path}")

    sch_path = f"{path}/scheduler.pt"
    if os.path.exists(sch_path) and scheduler is not None:
        scheduler.load_state_dict(
            torch.load(sch_path, map_location="cpu", weights_only=False)
        )
        log.info(f"Loaded scheduler state ← {sch_path}")

    meta_path = f"{path}/meta.json"
    step               = 0
    epoch              = 0
    batches_into_epoch = 0
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        step               = meta.get("step", 0)
        epoch              = meta.get("epoch", 0)
        batches_into_epoch = meta.get("batches_into_epoch", 0)

        if "epoch" not in meta:
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


def save_checkpoint(
    model: SpeechAura,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    epoch: int,
    batches_into_epoch: int,
    output_dir: str,
) -> str:
    """Save checkpoint. Must only be called on master rank."""
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

    log.info(
        f"Checkpoint saved → {ckpt_dir} "
        f"(step={step}, epoch={epoch}, batches_into_epoch={batches_into_epoch})"
    )
    return ckpt_dir


# ============================================================================
# Val index selection
# ============================================================================

def build_val_generate_indices(
    val_ds: SpeechDataset,
    samples_per_lang: int = 100,
) -> list[int]:
    """Return the first `samples_per_lang` indices per language. Deterministic."""
    from collections import defaultdict
    lang_indices: dict[str, list[int]] = defaultdict(list)
    # Read directly from the columnar array — avoids materializing dicts via _EntriesView
    src_langs = val_ds._src_languages
    for idx in range(len(val_ds)):
        lang = src_langs[idx] or "?"
        if len(lang_indices[lang]) < samples_per_lang:
            lang_indices[lang].append(idx)

    indices: list[int] = []
    for lang in sorted(lang_indices):
        n = len(lang_indices[lang])
        log.info(f"  Val generate: {n} samples for language '{lang}'")
        indices.extend(lang_indices[lang])

    log.info(f"Val generate indices: {len(indices)} total ({len(lang_indices)} languages)")
    return indices


# ============================================================================
# Validation  (master rank only)
# ============================================================================

@torch.no_grad()
def evaluate(
    model: SpeechAura,
    val_loader: DataLoader | None,
    device: torch.device,
    task: str,
    val_generate_indices: list[int],
    val_ds: SpeechDataset,
    rank: int = 0,
    world_size: int = 1,
    is_ddp: bool = False,
    step: int = 0,
    output_dir: str | None = None,
) -> dict[str, float]:
    """Sharded validation. All ranks must call this together.

    Each rank generates its slice of val_generate_indices, then gathers
    hyps/refs to rank 0 for metric computation. Only rank 0 returns
    populated results; other ranks return an empty dict.

    The loss pass currently runs on rank 0 only — it iterates a Subset-backed
    DataLoader that exists only on rank 0. Generation is the expensive part
    and is the one being sharded.

    Args:
        val_loader:           Loss-pass loader. Built only on rank 0; pass
                              None on other ranks.
        val_generate_indices: Full list of indices into val_ds. Must be
                              identical on every rank — broadcast before
                              calling this if necessary.
        val_ds:               Underlying full validation dataset. Required
                              on every rank for per-sample generation.
    """
    from collections import defaultdict
    from tqdm import tqdm

    model.eval()
    master = (rank == 0)
    results: dict[str, float] = {}

    # ---- Loss pass (master only — uses Subset loader on rank 0) ----
    if master and val_loader is not None:
        total_loss, n = 0.0, 0
        for batch in val_loader:
            if batch is None:
                continue
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                out = model(**batch)
            total_loss += out["loss"].item()
            n += 1
        results["loss"] = total_loss / max(n, 1)
        torch.cuda.empty_cache()

    # ---- Generation pass (sharded across ranks) ----
    # Interleaved slice: rank 0 gets indices 0, W, 2W, ...; rank 1 gets 1, W+1, ...
    # This keeps each rank's per-language sample distribution roughly uniform.
    my_indices = val_generate_indices[rank::world_size]

    my_idx_seen:      list[int] = []
    my_src_langs:     list[str] = []
    my_hyp_t:         list[str] = []
    my_ref_t:         list[str] = []
    my_hyp_r:         list[str] = []
    my_ref_r:         list[str] = []

    for idx in tqdm(
        my_indices,
        desc=f"Generating val (rank {rank})",
        unit="sample",
        dynamic_ncols=True,
        disable=not master,  # only master shows the bar; all ranks do the work
    ):
        sample  = val_ds[idx]
        mel     = sample["mel"].unsqueeze(0).to(device)
        mel_len = torch.tensor([sample["mel_len"]], device=device)
        try:
            output = model.generate(
                mel, mel_len,
                src_lang=sample["src_language"],
                tgt_lang=sample["tgt_language"],
                task=task,
                max_new_tokens=256 if task == "cot" else 128,
            )
            if task == "asr":
                hyp_t = model._strip_special_tokens(output).strip()
                hyp_r = ""
            else:
                hyp_t, hyp_r = model.split_cot_output(output)
        except Exception as e:
            log.warning(f"[rank {rank}] generate() failed for sample {idx}: {e}")
            hyp_t = ""
            hyp_r = ""

        my_idx_seen.append(idx)
        my_src_langs.append(sample["src_language"])
        my_hyp_t.append(hyp_t)
        my_ref_t.append(sample["transcript"].strip())
        my_hyp_r.append(hyp_r)
        my_ref_r.append(sample.get("translation", "").strip())

        del mel, mel_len
        torch.cuda.empty_cache()

    # ---- Gather to rank 0 ----
    local_payload = {
        "idx":   my_idx_seen,
        "lang":  my_src_langs,
        "hyp_t": my_hyp_t,
        "ref_t": my_ref_t,
        "hyp_r": my_hyp_r,
        "ref_r": my_ref_r,
    }

    if is_ddp:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_payload)
    else:
        gathered = [local_payload]

    # ---- Metric computation (master only) ----
    if not master:
        model.train()
        return {}

    # Flatten gathered payloads and re-sort by original idx so the CSV and
    # bucketing match the deterministic order from build_val_generate_indices.
    all_idx:    list[int] = []
    all_lang:   list[str] = []
    all_hyp_t:  list[str] = []
    all_ref_t:  list[str] = []
    all_hyp_r:  list[str] = []
    all_ref_r:  list[str] = []
    for payload in gathered:
        all_idx   .extend(payload["idx"])
        all_lang  .extend(payload["lang"])
        all_hyp_t .extend(payload["hyp_t"])
        all_ref_t .extend(payload["ref_t"])
        all_hyp_r .extend(payload["hyp_r"])
        all_ref_r .extend(payload["ref_r"])

    order = sorted(range(len(all_idx)), key=lambda i: all_idx[i])
    src_langs_seen   = [all_lang [i] for i in order]
    hyp_transcripts  = [all_hyp_t[i] for i in order]
    ref_transcripts  = [all_ref_t[i] for i in order]
    hyp_translations = [all_hyp_r[i] for i in order]
    ref_translations = [all_ref_r[i] for i in order]
    sorted_idx       = [all_idx  [i] for i in order]

    log.info(
        f"Gathered {len(hyp_transcripts)} val samples from {world_size} rank(s) "
        f"(expected {len(val_generate_indices)})"
    )

    if hyp_transcripts:
        lang_hyp_t: dict[str, list[str]] = defaultdict(list)
        lang_ref_t: dict[str, list[str]] = defaultdict(list)
        lang_hyp_r: dict[str, list[str]] = defaultdict(list)
        lang_ref_r: dict[str, list[str]] = defaultdict(list)
        for h_t, r_t, h_r, r_r, lang in zip(
            hyp_transcripts, ref_transcripts,
            hyp_translations, ref_translations,
            src_langs_seen,
        ):
            lang_hyp_t[lang].append(h_t)
            lang_ref_t[lang].append(r_t)
            lang_hyp_r[lang].append(h_r)
            lang_ref_r[lang].append(r_r)

        from jiwer import wer as _sample_wer
        per_sample_wer = []
        for r, p in zip(ref_transcripts, hyp_transcripts):
            try:
                per_sample_wer.append(_sample_wer(r, p) if r.strip() else 0.0)
            except Exception:
                per_sample_wer.append(1.0)

        results["wer"] = compute_wer(hyp_transcripts, ref_transcripts)
        for lang in sorted(lang_hyp_t):
            lang_wer = compute_wer(lang_hyp_t[lang], lang_ref_t[lang])
            results[f"wer_{lang}"] = lang_wer
            log.info(f"  val WER [{lang}]: {lang_wer:.4f} ({len(lang_hyp_t[lang])} samples)")

        if task == "cot":
            results["bleu"] = compute_bleu(hyp_translations, ref_translations)["bleu"]
            results["chrf"] = compute_chrf(hyp_translations, ref_translations)["chrf"]
            for lang in sorted(lang_hyp_r):
                lang_bleu = compute_bleu(lang_hyp_r[lang], lang_ref_r[lang])["bleu"]
                lang_chrf = compute_chrf(lang_hyp_r[lang], lang_ref_r[lang])["chrf"]
                results[f"bleu_{lang}"] = lang_bleu
                results[f"chrf_{lang}"] = lang_chrf
                log.info(
                    f"  val [{lang}]: BLEU={lang_bleu:.2f} chrF={lang_chrf:.2f} "
                    f"({len(lang_hyp_r[lang])} samples)"
                )

        logged: dict[str, int] = defaultdict(int)
        for h_t, r_t, h_r, r_r, lang in zip(
            hyp_transcripts, ref_transcripts,
            hyp_translations, ref_translations,
            src_langs_seen,
        ):
            if logged[lang] < 2:
                log.info(f"  [val {lang}] ref_t: {r_t[:80]}")
                log.info(f"  [val {lang}] hyp_t: {h_t[:80]}")
                if task == "cot":
                    log.info(f"  [val {lang}] ref_r: {r_r[:80]}")
                    log.info(f"  [val {lang}] hyp_r: {h_r[:80]}")
                logged[lang] += 1

        if output_dir is not None:
            csv_path = os.path.join(output_dir, f"val_preds_step{step}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if task == "asr":
                    writer.writerow(["idx", "src_lang", "ref_transcript",
                                     "hyp_transcript", "wer"])
                    for orig_idx, lang, r, h, w in zip(
                        sorted_idx, src_langs_seen, ref_transcripts,
                        hyp_transcripts, per_sample_wer,
                    ):
                        writer.writerow([orig_idx, lang, r, h, f"{w:.4f}"])
                else:
                    writer.writerow([
                        "idx", "src_lang",
                        "ref_transcript", "hyp_transcript", "wer",
                        "ref_translation", "hyp_translation",
                    ])
                    for orig_idx, lang, r_t, h_t, w, r_r, h_r in zip(
                        sorted_idx, src_langs_seen, ref_transcripts,
                        hyp_transcripts, per_sample_wer,
                        ref_translations, hyp_translations,
                    ):
                        writer.writerow([orig_idx, lang, r_t, h_t, f"{w:.4f}", r_r, h_r])
            log.info(f"Val predictions saved → {csv_path} ({len(hyp_transcripts)} samples)")

    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    return results


# ============================================================================
# Training loop
# ============================================================================

def train(cfg: dict, resume_from: str | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- DDP setup ----
    is_ddp, rank, local_rank, world_size, device_str = setup_ddp()
    master = (rank == 0)
    device = torch.device(device_str)

    if master:
        log.info(f"DDP: {'enabled' if is_ddp else 'disabled'} | "
                 f"rank={rank} | world_size={world_size} | device={device_str}")

    train_cfg  = cfg["training"]
    data_cfg   = cfg["data"]
    output_dir = train_cfg.get("output_dir", "runs/speech_aura")

    # Only master creates directories
    if master:
        os.makedirs(output_dir, exist_ok=True)

    task = data_cfg.get("task") or train_cfg.get("task", "asr")
    if task not in ("asr", "cot"):
        raise ValueError(f"data.task must be 'asr' or 'cot', got '{task}'")
    if master:
        log.info(f"Training task: {task}")

    # ---- Model ----
    # All ranks build the model identically (same weights loaded from checkpoint)
    model = build_model(cfg).to(device)

    # Wrap with DDP — gradient all-reduce happens automatically during backward()
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    # raw_model is used for: save/load checkpoint, generate, evaluate
    # — these must bypass the DDP wrapper
    raw_model: SpeechAura = model.module if is_ddp else model

    # ---- Data ----
    lowercase     = data_cfg.get("lowercase", False)
    src_languages = data_cfg.get("src_languages") or data_cfg.get("languages")
    tgt_languages = data_cfg.get("tgt_languages")

    train_ds = SpeechDataset(
        index_path=data_cfg["train_index"],
        split=data_cfg.get("train_split", "train"),
        task=task,
        src_languages=src_languages,
        tgt_languages=tgt_languages,
        max_duration=data_cfg.get("max_duration", 20.0),
        lowercase=lowercase,
    )
    val_ds = None
    if data_cfg.get("val_index"):
        val_ds = SpeechDataset(
            index_path=data_cfg["val_index"],
            split=data_cfg.get("val_split", "dev"),
            task=task,
            src_languages=src_languages,
            tgt_languages=tgt_languages,
            max_duration=data_cfg.get("max_duration", 20.0),
            lowercase=lowercase,
        )

    vocab = None
    if train_cfg.get("ctc_weight", 0.0) > 0:
        from st.data.vocab import load_vocab
        vocab = load_vocab(data_cfg["vocab_path"])
        if master:
            log.info(f"CTC vocab loaded: {len(vocab)} tokens")

    collator = AuraCollator(
        tokenizer=raw_model.aura.tokenizer,
        vocab=vocab,
        max_target_tokens=train_cfg.get("max_target_tokens", 256),
    )

    # Synchronized bucket sampler: shared seed for bucket ORDER (all ranks),
    # per-rank slice for data parallelism
    train_sampler = DurationBucketSampler(
        dataset=train_ds,
        target_duration=train_cfg.get("max_batch_duration", 120.0),
        max_batch_size=train_cfg.get("max_batch_size", 64),
        shuffle=True,
        shuffle_buckets=True,
        rank=rank,
        world_size=world_size,
        seed=42,
    )
    if master:
        log.info(f"Train: {len(train_ds)} samples, {len(train_sampler)} batches/epoch "
                 f"(world_size={world_size}, effective_batch ≈ "
                 f"{train_cfg.get('max_batch_duration', 120.0) * world_size:.0f}s/step)")

    num_workers = train_cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = None
    val_generate_indices: list[int] = []
    if val_ds is not None:
        samples_per_lang = train_cfg.get("val_samples_per_lang", 100)

        if master:
            val_generate_indices = build_val_generate_indices(val_ds, samples_per_lang)

        # Broadcast the indices list so every rank has the same shard input.
        if is_ddp:
            obj_list = [val_generate_indices if master else None]
            dist.broadcast_object_list(obj_list, src=0)
            val_generate_indices = obj_list[0]

        if master:
            from torch.utils.data import Subset
            val_subset = Subset(val_ds, val_generate_indices)
            val_subset.durations = [float(val_ds.durations[i]) for i in val_generate_indices]
            val_sampler = DurationBucketSampler(
                dataset=val_subset,
                target_duration=train_cfg.get("max_batch_duration", 120.0),
                max_batch_size=train_cfg.get("max_batch_size", 64),
                shuffle=False,
                shuffle_buckets=False,
            )
            val_loader = DataLoader(
                val_subset,
                batch_sampler=val_sampler,
                num_workers=num_workers,
                collate_fn=collator,
                pin_memory=True,
                persistent_workers=num_workers > 0,
            )

    # ---- Optimizer ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    lr     = float(train_cfg.get("lr", 2e-4))
    min_lr = float(train_cfg.get("min_lr", 1e-6))
    optimizer = torch.optim.AdamW(
        trainable,
        lr=lr,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    # ---- Scheduler ----
    max_steps = train_cfg["max_steps"]
    scheduler = build_scheduler(
        name=train_cfg.get("scheduler", "cosine_warmup_restarts"),
        optimizer=optimizer,
        total_steps=max_steps,
        max_lr=lr,
        min_lr=min_lr,
        warmup_steps=train_cfg.get("warmup_steps", 1000),
        first_cycle_steps=train_cfg.get("first_cycle_steps", max_steps),
        gamma=train_cfg.get("gamma", 1.0),
    )

    # ---- Resume ----
    # Load on all ranks so weights + optimizer state are identical everywhere.
    # raw_model is the unwrapped model — DDP wrapping happened above, but
    # load_checkpoint operates on the underlying SpeechAura via raw_model.
    start_step           = 0
    start_epoch          = 0
    start_batch_in_epoch = 0
    if resume_from:
        start_step, start_epoch, start_batch_in_epoch = load_checkpoint(
            raw_model, optimizer, scheduler, resume_from,
        )

    # ---- W&B — master only ----
    use_wandb = not train_cfg.get("no_wandb", False)
    if master and use_wandb:
        try:
            import wandb
            wandb.init(
                project=train_cfg.get("wandb_project", "iwslt2026"),
                entity=train_cfg.get("wandb_entity"),
                name=train_cfg.get("wandb_run_name", os.path.basename(output_dir)),
                config=cfg,
                resume="allow" if start_step > 0 else None,
            )
            log.info(f"W&B: {wandb.run.url}")
        except ImportError:
            use_wandb = False
    elif not master:
        use_wandb = False  # non-master ranks never log to W&B

    # ---- Training loop ----
    model.train()
    global_step      = start_step
    epoch            = start_epoch
    batches_in_epoch = start_batch_in_epoch

    grad_accum   = train_cfg.get("grad_accum", 1)
    log_every    = train_cfg.get("log_every", 100)
    save_every   = train_cfg.get("save_every", 5000)
    eval_every   = train_cfg.get("eval_every", 5000)
    oom_cooldown = 0

    running: dict[str, float] = {"loss": 0.0, "ce_loss": 0.0, "ctc_loss": 0.0}
    run_n      = 0
    micro_step = 0

    from tqdm import tqdm
    pbar = tqdm(
        total=max_steps - start_step,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
        disable=not master,
    )

    if master:
        log.info(
            f"Training for {max_steps} steps (resuming from step={start_step}, "
            f"epoch={start_epoch}, batches_into_epoch={start_batch_in_epoch})"
        )
    optimizer.zero_grad()

    # Outer loop: each iteration consumes one (possibly partial-on-resume) epoch.
    # We do NOT pre-increment `epoch` — the first pass replays the same epoch
    # number we were saved in, then bumps at the bottom after the for-loop
    # exhausts naturally.
    while global_step < max_steps:
        train_sampler.set_epoch(epoch)
        if batches_in_epoch > 0:
            train_sampler.skip(batches_in_epoch)
            if master:
                log.info(
                    f"Resuming epoch {epoch}: skipping {batches_in_epoch} batches "
                    f"(out of {len(train_sampler)} per-rank batches this epoch)"
                )

        for batch in train_loader:
            # Synchronize cooldown across ranks
            if is_ddp and oom_cooldown > 0:
                cooldown_tensor = torch.tensor(oom_cooldown, dtype=torch.int32, device=device)
                dist.all_reduce(cooldown_tensor, op=dist.ReduceOp.MAX)
                oom_cooldown = int(cooldown_tensor.item())

            # A batch was drawn from the sampler — advance position regardless
            # of whether we end up processing it. Two cases where we skip the
            # body but still count the batch as consumed:
            #   1. collator returned None (all samples dropped by max_target_tokens)
            #   2. we're in OOM cooldown
            if batch is None or oom_cooldown > 0:
                batches_in_epoch += 1
                oom_cooldown = max(0, oom_cooldown - 1)
                continue

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            cur_bs  = batch["audio_features"].size(0)
            cur_dur = batch["audio_lengths"].sum().item() * 0.01

            oom_this_step = False
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    out  = model(**batch)
                    loss = out["loss"] / grad_accum
            except torch.cuda.OutOfMemoryError:
                oom_this_step = True
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                gc.collect()

            # Sync OOM BEFORE backward
            if is_ddp:
                oom_tensor = torch.tensor(int(oom_this_step), dtype=torch.int32, device=device)
                dist.all_reduce(oom_tensor, op=dist.ReduceOp.MAX)
                oom_this_step = bool(oom_tensor.item())

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

            # Only reach backward if no rank OOM'd
            loss.backward()

            # Accumulate unscaled metrics
            for k in ("loss", "ce_loss", "ctc_loss"):
                running[k] += out[k].item()
            run_n            += 1
            micro_step       += 1
            batches_in_epoch += 1

            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

                if scheduler is not None:
                    scheduler.step()

                global_step += 1
                if master:
                    pbar.update(1)

            if master:
                pbar.set_postfix(
                    loss=f"{out['loss'].item():.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                    bs=cur_bs, dur=f"{cur_dur:.0f}s", ep=epoch,
                )

            # ---- Logging (master only) ----
            if global_step % log_every == 0 \
                    and run_n > 0 and micro_step % grad_accum == 0:
                
                loss_for_log = out["loss"].detach().clone()
                reduce_tensor(loss_for_log)  # all ranks participate

                if master:
                    avg    = {k: v / run_n for k, v in running.items()}
                    cur_lr = optimizer.param_groups[0]["lr"]
                    log.info(
                        f"step {global_step}/{max_steps} | "
                        + " | ".join(f"{k}={v:.4f}" for k, v in avg.items())
                        + f" | lr={cur_lr:.2e} | bs={cur_bs} | dur={cur_dur:.0f}s | "
                        + f"ep={epoch} | bie={batches_in_epoch}"
                    )
                    if use_wandb:
                        import wandb
                        wandb.log(
                            {f"train/{k}": v for k, v in avg.items()}
                            | {"train/lr": cur_lr, "train/epoch": epoch,
                            "train/batch_size": cur_bs * world_size,
                            "train/batch_dur": cur_dur * world_size},
                            step=global_step,
                        )
                    running = {k: 0.0 for k in running}
                    run_n   = 0

            # ---- Checkpoint (master only) ----
            # All ranks evaluate the predicate together
            should_save = (
                global_step > 0
                and global_step % save_every == 0
                and micro_step % grad_accum == 0
            )
            if should_save:
                if master:
                    save_checkpoint(
                        raw_model, optimizer, scheduler,
                        global_step, epoch, batches_in_epoch, output_dir,
                    )
                barrier()  # all ranks wait so non-master doesn't race ahead

            # ---- Validation (ALL ranks must enter for the sharded generate to work) ----
            if val_ds is not None \
                and global_step > 0 \
                and global_step % eval_every == 0 \
                and micro_step % grad_accum == 0:

                torch.cuda.empty_cache()
                metrics = evaluate(
                    raw_model, val_loader, device, task,
                    val_generate_indices=val_generate_indices,
                    val_ds=val_ds,
                    rank=rank, world_size=world_size, is_ddp=is_ddp,
                    step=global_step, output_dir=output_dir,
                )
                if master:
                    log.info(
                        f"step {global_step} val | "
                        + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                    )
                    if use_wandb:
                        import wandb
                        wandb.log({f"val/{k}": v for k, v in metrics.items()},
                                step=global_step)
                model.train()  # all ranks restore train mode

            if global_step >= max_steps:
                break

        # For-loop exhausted naturally → epoch complete. Bump and reset.
        epoch += 1
        batches_in_epoch = 0

    pbar.close()

    # ---- Final checkpoint and eval ----
    if master:
        save_checkpoint(
            raw_model, optimizer, scheduler,
            global_step, epoch, batches_in_epoch, output_dir,
        )
    barrier()  # all ranks

    # ---- Validation (ALL ranks must enter for the sharded generate to work) ----
    if val_ds is not None:
        torch.cuda.empty_cache()
        metrics = evaluate(
            raw_model, val_loader, device, task,
            val_generate_indices=val_generate_indices,
            val_ds=val_ds,
            rank=rank, world_size=world_size, is_ddp=is_ddp,
            step=global_step, output_dir=output_dir,
        )
        if master:
            log.info(
                f"step {global_step} val | "
                + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            )
            if use_wandb:
                import wandb
                wandb.log({f"val/{k}": v for k, v in metrics.items()},
                        step=global_step)

    if use_wandb:
        import wandb
        wandb.finish()

    # All ranks wait here before tearing down
    barrier()
    teardown_ddp()

    if master:
        log.info("Training complete.")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train SpeechAura (stages 2/3/4)")
    parser.add_argument("--config",      required=True, help="Experiment YAML config")
    parser.add_argument("--resume_from", default=None,  help="Checkpoint directory to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume_from=args.resume_from)


if __name__ == "__main__":
    main()