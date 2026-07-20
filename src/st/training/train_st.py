"""
Stage 2 / 3 / 4 — Speech Translation training.

Stage 2: Freeze encoder + LLM, train projector only.
Stage 3: Freeze encoder + LLM, train projector + LoRA.
Stage 4: Train everything (encoder + projector + full LLM).

Controlled entirely by the experiment YAML — no code changes needed to
switch stages.

Usage:
    python -m st.training.train_st --config configs/experiment/stage2/stage2_omniasr_live_22lang.yaml
    python -m st.training.train_st --config configs/experiment/stage3/stage3_omniasr_live_22lang.yaml \
        --resume_from runs/stage2_omniasr_live_22lang/checkpoint_stepNNNN
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
from torch.utils.data import DataLoader

from st.data import (
    SpeechDataset, AuraCollator,
    CachedFeatureDataset, CachedFeatureCollator,
    RawAudioDataset, RawAudioCollator,
    DurationBucketSampler,
)
from st.models import (
    SpeechAura, AuraLLM,
    load_encoder_from_checkpoint,
    build_ctc_compressor,
)
from st.utils.config import load_config
from st.utils.schedulers import build_scheduler
from st.utils.metrics import compute_wer, compute_bleu, compute_chrf

log = logging.getLogger(__name__)


# ============================================================================
# Build model from config
# ============================================================================

def build_model(cfg: dict) -> SpeechAura:
    aura_cfg     = cfg["aura"]
    proj_cfg     = cfg.get("projector", {"type": "mlp"})
    ctc_comp_cfg = cfg.get("ctc_compress", None)
    train_cfg    = cfg["training"]
    cached_cfg   = cfg.get("cached_features", {})
    use_cached   = cached_cfg.get("enabled", False)

    # Encoder — either a live SpeechEncoder, or None for cached-features mode
    # (precomputed omniASR_CTC_1B features, see scripts/extract_omniasr_features.py)
    if use_cached:
        if train_cfg.get("ctc_weight", 0.0) > 0.0:
            raise ValueError(
                "cached_features mode requires training.ctc_weight == 0.0 — cached "
                "features only store argmax predicted_ids, not per-frame ctc_logits."
            )
        if train_cfg.get("unfreeze_encoder", False):
            raise ValueError(
                "cached_features mode has no live encoder to unfreeze — "
                "set training.unfreeze_encoder: false."
            )
        encoder = None
        encoder_output_dim = cached_cfg["encoder_output_dim"]
        log.info(
            f"Cached-features mode: encoder_output_dim={encoder_output_dim}, "
            f"cache_dir={cached_cfg.get('cache_dir')}"
        )
    else:
        enc_cfg = cfg["encoder"]
        enc_type = enc_cfg.get("type", "conformer")
        if enc_type == "conformer":
            encoder = load_encoder_from_checkpoint(
                config=enc_cfg,
                checkpoint_path=enc_cfg.get("checkpoint"),
                vocab_size=enc_cfg.get("vocab_size"),   # required if ctc_weight>0 or ctc_compress
                strict=False,
            )
        elif enc_type == "omniasr_live":
            if train_cfg.get("ctc_weight", 0.0) > 0.0:
                raise ValueError(
                    "encoder.type == 'omniasr_live' requires training.ctc_weight == 0.0 — "
                    "omniASR's own CTC head outputs over its 9812-piece SentencePiece vocab, "
                    "incompatible with this project's char-level CTC vocab."
                )
            from st.models.omniasr_encoder import build_omniasr_encoder_from_config
            encoder = build_omniasr_encoder_from_config(enc_cfg)
        else:
            raise ValueError(f"Unknown encoder.type: {enc_type!r}")
        encoder_output_dim = None

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
        task=train_cfg.get("task", "transcribe"),
    )

    model = SpeechAura(
        encoder=encoder,
        aura=aura,
        projector_cfg=proj_cfg,
        ctc_compress_cfg=ctc_comp_cfg,
        ctc_weight=train_cfg.get("ctc_weight", 0.0),
        freeze_encoder=not train_cfg.get("unfreeze_encoder", False),
        freeze_llm=freeze_llm,
        encoder_output_dim=encoder_output_dim,
    )

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
# Load a checkpoint into an existing model (for resume)
# ============================================================================

def load_checkpoint(
    model: SpeechAura,
    optimizer: torch.optim.Optimizer,
    scheduler,
    path: str,
) -> tuple[int, bool]:
    """Load checkpoint directory. Returns (step, optimizer_loaded)."""
    import json
    import os

    model.load_checkpoint(path)
    opt_loaded = False

    opt_path = f"{path}/optimizer.pt"
    if os.path.exists(opt_path):
        try:
            optimizer.load_state_dict(
                torch.load(opt_path, map_location="cpu", weights_only=False)
            )
            log.info(f"Loaded optimizer state ← {opt_path}")
            opt_loaded = True
        except ValueError as e:
            log.warning(f"Could not load optimizer state (likely stage transition): {e}")
            log.info("Starting with a fresh optimizer state for this stage.")

    sch_path = f"{path}/scheduler.pt"
    if os.path.exists(sch_path) and scheduler is not None and opt_loaded:
        try:
            scheduler.load_state_dict(
                torch.load(sch_path, map_location="cpu", weights_only=False)
            )
            log.info(f"Loaded scheduler state ← {sch_path}")
        except (ValueError, KeyError) as e:
            log.warning(f"Could not load scheduler state: {e}")

    meta_path = f"{path}/meta.json"
    step = 0
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        step = meta.get("step", 0)

    log.info(f"Resumed from {path} at step {step}")
    return step, opt_loaded


# ============================================================================
# Save checkpoint
# ============================================================================

def save_checkpoint(
    model: SpeechAura,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    output_dir: str,
) -> str:
    import json
    ckpt_dir = os.path.join(output_dir, f"checkpoint_step{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    model.save_checkpoint(ckpt_dir)
    torch.save(optimizer.state_dict(), f"{ckpt_dir}/optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), f"{ckpt_dir}/scheduler.pt")

    # Overwrite meta with step
    meta_path = f"{ckpt_dir}/meta.json"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    meta["step"] = step
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Checkpoint saved → {ckpt_dir}")
    return ckpt_dir


# ============================================================================
# Val index selection
# ============================================================================

def build_val_generate_indices(
    val_ds: SpeechDataset,
    samples_per_lang: int = 100,
) -> list[int]:
    """Return the first `samples_per_lang` indices per language from val_ds.

    Iterates entries in dataset order (no shuffling) so the result is
    fully deterministic and stable across resumes.
    """
    from collections import defaultdict
    lang_indices: dict[str, list[int]] = defaultdict(list)
    for idx, entry in enumerate(val_ds.entries):
        lang = entry.get("language") or entry.get("src_language") or "?"
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
# Model-forward dispatch (live encoder vs. cached-features mode)
# ============================================================================

def run_forward(model: SpeechAura, batch: dict, cached: bool) -> dict[str, torch.Tensor]:
    """Dispatch to SpeechAura.forward() or .forward_cached() depending on
    whether the batch came from AuraCollator (live encoder) or
    CachedFeatureCollator (precomputed omniASR features). Used at all three
    model-forward call sites (train loop, evaluate()'s loss loop) so the
    branch only needs to be written once.
    """
    if cached:
        return model.forward_cached(
            hidden_states=batch["encoder_hidden_states"],
            predicted_ids=batch["ctc_predicted_ids"],
            audio_lengths=batch["encoder_lengths"],
            target_ids=batch["target_ids"],
            target_lengths=batch["target_lengths"],
            languages=batch["language"],
        )
    return model(
        audio_features=batch["audio_features"],
        audio_lengths=batch["audio_lengths"],
        target_ids=batch["target_ids"],
        target_lengths=batch["target_lengths"],
        languages=batch["language"],
        ctc_labels=batch.get("ctc_labels"),
        ctc_label_lengths=batch.get("ctc_label_lengths"),
    )


# ============================================================================
# Validation
# ============================================================================

@torch.no_grad()
def evaluate(
    model: SpeechAura,
    val_loader: DataLoader,
    device: torch.device,
    task: str,
    val_generate_indices: list[int],
    step: int = 0,
    output_dir: str | None = None,
    mode: str = "mel") -> dict[str, float]:
    """mode: "mel" (live Conformer) | "cached" (frozen omniASR features) |
    "waveform" (live/unfrozen omniASR encoder)."""

    from collections import defaultdict
    from tqdm import tqdm

    cached = mode == "cached"

    model.eval()
    total_loss, n = 0.0, 0

    for batch in val_loader:
        if batch is None:
            continue
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
            out = run_forward(model, batch, cached)
        total_loss += out["loss"].item()
        n += 1

    results: dict[str, float] = {"loss": total_loss / max(n, 1)}
    torch.cuda.empty_cache()

    # Generation over fixed val indices (first N per language)
    val_ds = val_loader.dataset
    preds: list[str] = []
    refs:  list[str] = []
    languages_seen: list[str] = []

    for idx in tqdm(val_generate_indices, desc="Generating val", unit="sample", dynamic_ncols=True):
        sample = val_ds[idx]
        try:
            if mode == "cached":
                hidden_states = sample["hidden_states"].unsqueeze(0).to(device)
                predicted_ids = sample["predicted_ids"].unsqueeze(0).to(device)
                feature_len   = torch.tensor([sample["feature_len"]], device=device)
                pred = model.generate_cached(
                    hidden_states, predicted_ids, feature_len,
                    target_lang=sample["language"],
                    max_new_tokens=128,
                )
                del hidden_states, predicted_ids, feature_len
            elif mode == "waveform":
                waveform     = sample["waveform"].unsqueeze(0).to(device)
                waveform_len = torch.tensor([sample["waveform_len"]], device=device)
                pred = model.generate(
                    waveform, waveform_len,
                    target_lang=sample["language"],
                    max_new_tokens=128,
                )
                del waveform, waveform_len
            else:
                mel     = sample["mel"].unsqueeze(0).to(device)
                mel_len = torch.tensor([sample["mel_len"]], device=device)
                pred = model.generate(
                    mel, mel_len,
                    target_lang=sample["language"],
                    max_new_tokens=128,
                )
                del mel, mel_len
            preds.append(pred.strip())
        except Exception as e:
            log.warning(f"generate() failed for sample {idx}: {e}")
            preds.append("")
        refs.append(sample["text"].strip())
        languages_seen.append(sample["language"])

        torch.cuda.empty_cache()

    if preds:
        # Group by language for per-language metrics
        lang_preds: dict[str, list[str]] = defaultdict(list)
        lang_refs:  dict[str, list[str]] = defaultdict(list)
        for r, p, lang in zip(refs, preds, languages_seen):
            lang_preds[lang].append(p)
            lang_refs[lang].append(r)

        if task == "transcribe":
            from jiwer import wer as _sample_wer
            per_sample_wer = []
            for r, p in zip(refs, preds):
                try:
                    per_sample_wer.append(_sample_wer(r, p) if r.strip() else 0.0)
                except Exception:
                    per_sample_wer.append(1.0)

            # Overall WER
            results["wer"] = compute_wer(preds, refs)
            # Per-language WER
            for lang in sorted(lang_preds):
                lang_wer = compute_wer(lang_preds[lang], lang_refs[lang])
                results[f"wer_{lang}"] = lang_wer
                log.info(f"  val WER [{lang}]: {lang_wer:.4f} ({len(lang_preds[lang])} samples)")
        else:
            per_sample_wer = None
            # Overall BLEU / chrF
            results["bleu"] = compute_bleu(preds, refs)["bleu"]
            results["chrf"] = compute_chrf(preds, refs)["chrf"]
            # Per-language BLEU / chrF
            for lang in sorted(lang_preds):
                lang_bleu = compute_bleu(lang_preds[lang], lang_refs[lang])["bleu"]
                lang_chrf = compute_chrf(lang_preds[lang], lang_refs[lang])["chrf"]
                results[f"bleu_{lang}"] = lang_bleu
                results[f"chrf_{lang}"] = lang_chrf
                log.info(
                    f"  val [{lang}]: BLEU={lang_bleu:.2f} chrF={lang_chrf:.2f} "
                    f"({len(lang_preds[lang])} samples)"
                )

        # Log a few examples per language
        logged: dict[str, int] = defaultdict(int)
        for r, p, lang in zip(refs, preds, languages_seen):
            if logged[lang] < 2:
                log.info(f"  [val {lang}]  ref: {r[:80]}")
                log.info(f"  [val {lang}]  hyp: {p[:80]}")
                logged[lang] += 1

        # Write CSV
        if output_dir is not None:
            csv_path = os.path.join(output_dir, f"val_preds_step{step}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if task == "transcribe":
                    writer.writerow(["idx", "language", "reference", "hypothesis", "wer"])
                    for i, (r, p, lang, w) in enumerate(
                        zip(refs, preds, languages_seen, per_sample_wer)
                    ):
                        writer.writerow([i, lang, r, p, f"{w:.4f}"])
                else:
                    writer.writerow(["idx", "language", "reference", "hypothesis"])
                    for i, (r, p, lang) in enumerate(zip(refs, preds, languages_seen)):
                        writer.writerow([i, lang, r, p])
            log.info(f"Val predictions saved → {csv_path} ({len(preds)} samples)")

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TF32 on Tensor Cores gives ~10% speedup at no accuracy cost since we
    # train in bfloat16 anyway.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log.info(f"CUDA device: {torch.cuda.get_device_name(0)} — TF32 enabled")

    train_cfg  = cfg["training"]
    data_cfg   = cfg["data"]
    output_dir = train_cfg.get("output_dir", "runs/speech_aura")
    os.makedirs(output_dir, exist_ok=True)

    use_cached = cfg.get("cached_features", {}).get("enabled", False)
    enc_type   = cfg.get("encoder", {}).get("type", "conformer")
    if use_cached:
        data_mode = "cached"
    elif enc_type == "omniasr_live":
        data_mode = "waveform"
    else:
        data_mode = "mel"

    # --- Model ---
    model = build_model(cfg).to(device)

    # Enable gradient checkpointing for full fine-tuning (if LLM is unfrozen)
    if train_cfg.get("unfreeze_llm", False):
        model.aura.model.gradient_checkpointing_enable()
        log.info("Gradient checkpointing enabled for AuraLLM")

    # --- Data ---
    languages = data_cfg.get("languages")
    task      = train_cfg.get("task", "transcribe")

    lowercase = data_cfg.get("lowercase", False)

    if use_cached:
        cache_dir = cfg["cached_features"]["cache_dir"]
        train_ds = CachedFeatureDataset(
            index_path=data_cfg["train_index"],
            cache_dir=cache_dir,
            split=data_cfg.get("train_split", "train"),
            languages=languages,
            max_duration=data_cfg.get("max_duration", 20.0),
            lowercase=lowercase,
        )
        val_ds = None
        if data_cfg.get("val_index"):
            val_ds = CachedFeatureDataset(
                index_path=data_cfg["val_index"],
                cache_dir=cache_dir,
                split=data_cfg.get("val_split", "dev"),
                languages=languages,
                max_duration=data_cfg.get("max_duration", 20.0),
                lowercase=lowercase,
            )
        collator = CachedFeatureCollator(
            tokenizer=model.aura.tokenizer,
            max_target_tokens=train_cfg.get("max_target_tokens", 256),
        )
    elif enc_type == "omniasr_live":
        train_ds = RawAudioDataset(
            index_path=data_cfg["train_index"],
            split=data_cfg.get("train_split", "train"),
            languages=languages,
            max_duration=data_cfg.get("max_duration", 20.0),
            lowercase=lowercase,
        )
        val_ds = None
        if data_cfg.get("val_index"):
            val_ds = RawAudioDataset(
                index_path=data_cfg["val_index"],
                split=data_cfg.get("val_split", "dev"),
                languages=languages,
                max_duration=data_cfg.get("max_duration", 20.0),
                lowercase=lowercase,
            )
        # No CTC vocab — ctc_weight is guarded to 0.0 for omniasr_live in build_model()
        collator = RawAudioCollator(
            tokenizer=model.aura.tokenizer,
            max_target_tokens=train_cfg.get("max_target_tokens", 256),
        )
    else:
        train_ds = SpeechDataset(
            index_path=data_cfg["train_index"],
            split=data_cfg.get("train_split", "train"),
            languages=languages,
            max_duration=data_cfg.get("max_duration", 20.0),
            lowercase=lowercase,
        )
        val_ds = None
        if data_cfg.get("val_index"):
            val_ds = SpeechDataset(
                index_path=data_cfg["val_index"],
                split=data_cfg.get("val_split", "dev"),
                languages=languages,
                max_duration=data_cfg.get("max_duration", 20.0),
                lowercase=lowercase,
            )

        # Vocab for CTC labels (only needed when ctc_weight > 0)
        vocab = None
        if train_cfg.get("ctc_weight", 0.0) > 0:
            from st.data.vocab import load_vocab
            vocab = load_vocab(data_cfg["vocab_path"])
            log.info(f"CTC vocab loaded: {len(vocab)} tokens")

        collator = AuraCollator(
            tokenizer=model.aura.tokenizer,
            vocab=vocab,
            max_target_tokens=train_cfg.get("max_target_tokens", 256),
        )

    train_sampler = DurationBucketSampler(
        dataset=train_ds,
        target_duration=train_cfg.get("max_batch_duration", 120.0),
        max_batch_size=train_cfg.get("max_batch_size", 64),
        shuffle=True,
        shuffle_buckets=True,
    )
    log.info(f"Train: {len(train_ds)} samples, {len(train_sampler)} batches/epoch")

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=collator,
        pin_memory=True,
    )
    val_loader = None
    if val_ds:
        val_sampler = DurationBucketSampler(
            dataset=val_ds,
            target_duration=train_cfg.get("max_batch_duration", 120.0),
            max_batch_size=train_cfg.get("max_batch_size", 64),
            shuffle=False,
            shuffle_buckets=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=val_sampler,
            num_workers=train_cfg.get("num_workers", 4),
            collate_fn=collator,
            pin_memory=True,
        )

    # Build fixed val generation indices once — first N per language, deterministic
    val_generate_indices: list[int] = []
    if val_ds is not None:
        samples_per_lang = train_cfg.get("val_samples_per_lang", 100)
        val_generate_indices = build_val_generate_indices(val_ds, samples_per_lang)

    # --- Optimizer ---
    trainable = [p for p in model.parameters() if p.requires_grad]
    lr     = float(train_cfg.get("lr", 2e-4))
    min_lr = float(train_cfg.get("min_lr", 1e-6))
    optimizer = torch.optim.AdamW(
        trainable,
        lr=lr,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    # --- Scheduler ---
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

    # --- Resume ---
    start_step = 0
    if resume_from:
        step, opt_loaded = load_checkpoint(model, optimizer, scheduler, resume_from)
        if opt_loaded:
            start_step = step
        else:
            log.info("Transitioning stages: Resetting step counter to 0 for a fresh warmup.")
            start_step = 0

    # --- W&B ---
    use_wandb = not train_cfg.get("no_wandb", False)
    if use_wandb:
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

    # --- Training loop ---
    model.train()
    global_step   = start_step
    epoch         = 0
    grad_accum    = train_cfg.get("grad_accum", 8)
    log_every     = train_cfg.get("log_every", 100)
    save_every    = train_cfg.get("save_every", 5000)
    eval_every    = train_cfg.get("eval_every", 5000)
    oom_cooldown  = 0

    running: dict[str, float] = {"loss": 0.0, "ce_loss": 0.0, "ctc_loss": 0.0}
    run_n = 0
    micro_step = 0

    from tqdm import tqdm
    pbar = tqdm(total=max_steps - start_step, desc="Training", unit="step", dynamic_ncols=True)

    log.info(f"Training for {max_steps} steps (resuming from {start_step})")
    optimizer.zero_grad()

    while global_step < max_steps:
        epoch += 1
        for batch in train_loader:
            if batch is None or oom_cooldown > 0:
                oom_cooldown = max(0, oom_cooldown - 1)
                continue

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            lengths_key   = "encoder_lengths" if use_cached else "audio_lengths"
            features_key  = "encoder_hidden_states" if use_cached else "audio_features"
            # omniASR@50Hz (cached) vs. mel@10ms hop vs. raw 16kHz waveform (live omniASR)
            frame_seconds = {"cached": 0.02, "mel": 0.01, "waveform": 1 / 16000}[data_mode]
            cur_bs  = batch[features_key].size(0)
            cur_dur = batch[lengths_key].sum().item() * frame_seconds

            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=(device.type == "cuda")):
                    out = run_forward(model, batch, use_cached)
                    loss = out["loss"] / grad_accum

                loss.backward()

            except torch.cuda.OutOfMemoryError:
                log.warning(
                    f"OOM at step {global_step}: bs={cur_bs} — skipping"
                )
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                oom_cooldown = 3
                micro_step = (micro_step // grad_accum) * grad_accum
                running = {k: 0.0 for k in running}
                run_n = 0
                continue

            # Accumulate metrics (unscaled)
            for k in ("loss", "ce_loss", "ctc_loss"):
                running[k] += out[k].item()
            run_n += 1
            micro_step += 1

            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

                if scheduler is not None:
                    scheduler.step()

                global_step += 1
                pbar.update(1)

            pbar.set_postfix(
                loss=f"{out['loss'].item():.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                bs=cur_bs, dur=f"{cur_dur:.0f}s", ep=epoch,
            )

            if global_step % log_every == 0 and run_n > 0 and micro_step % grad_accum == 0:
                avg   = {k: v / run_n for k, v in running.items()}
                cur_lr = optimizer.param_groups[0]["lr"]
                log.info(
                    f"step {global_step}/{max_steps} | "
                    + " | ".join(f"{k}={v:.4f}" for k, v in avg.items())
                    + f" | lr={cur_lr:.2e} | bs={cur_bs} | dur={cur_dur:.0f}s"
                )
                if use_wandb:
                    import wandb
                    wandb.log(
                        {f"train/{k}": v for k, v in avg.items()}
                        | {"train/lr": cur_lr, "train/epoch": epoch,
                           "train/batch_size": cur_bs, "train/batch_dur": cur_dur},
                        step=global_step,
                    )
                running = {k: 0.0 for k in running}
                run_n = 0

            if global_step % save_every == 0 and micro_step % grad_accum == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, output_dir)

            if val_loader and global_step % eval_every == 0 and micro_step % grad_accum == 0:
                torch.cuda.empty_cache()
                metrics = evaluate(
                    model, val_loader, device, task,
                    val_generate_indices=val_generate_indices,
                    step=global_step, output_dir=output_dir,
                    mode=data_mode,
                )
                log.info(
                    f"step {global_step} val | "
                    + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                )
                if use_wandb:
                    import wandb
                    wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=global_step)

            if global_step >= max_steps:
                break

    pbar.close()
    save_checkpoint(model, optimizer, scheduler, global_step, output_dir)

    if val_loader:
        metrics = evaluate(
            model, val_loader, device, task,
            val_generate_indices=val_generate_indices,
            step=global_step, output_dir=output_dir,
            mode=data_mode,
        )
        log.info("Final val | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if use_wandb:
            import wandb
            wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=global_step)

    if use_wandb:
        import wandb
        wandb.finish()

    log.info("Training complete.")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train SpeechAura (stages 2/3/4)")
    parser.add_argument("--config",      required=True, help="Experiment YAML config")
    parser.add_argument("--resume_from", default=None,  help="Checkpoint directory to resume from")
    args = parser.parse_args()

    from st.utils.config import load_config
    cfg = load_config(args.config)
    train(cfg, resume_from=args.resume_from)


if __name__ == "__main__":
    main()