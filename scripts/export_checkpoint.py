"""
Export a trained checkpoint into a single self-contained directory for HF upload.

Three modes:

  encoder   Stage 1 CTC encoder. Packages:
              - encoder.pt          (model weights only)
              - encoder_config.yaml (architecture)
              - vocab.json          (CTC vocab)
              - README.md           (auto-generated stub)
            Optionally pass --config to enrich the README with the training
            language list + data provenance (architecture/vocab still come
            from the checkpoint, which is authoritative for the shipped weights).

  speech_aura  Stage 2/3/4 SpeechAura. Packages everything needed to rebuild
               the full model offline:
              - config.yaml             (experiment YAML, with paths rewritten
                                          to point inside the export dir)
              - encoder.pt              (Stage 1 encoder weights)
              - encoder_config.yaml
              - vocab.json              (CTC vocab — only if ctc_weight > 0 or
                                          ctc_compress is enabled)
              - projector.pt            (from checkpoint dir)
              - lora.pt                 (if Stage 3)
              - llm_full.pt             (if Stage 4 full FT)
              - meta.json               (training position + flags)
              - aura/                   (Aura base: model.safetensors/.pt + tokenizer.json)
              - README.md

  speech_nllb  Stage 2/3/4 SpeechNLLB (the NLLB-200 decoder line). Same idea,
               but the fine-tuned NLLB weights are *merged into* the base rather
               than shipped as a delta beside it — a `trainable: all` checkpoint
               is the whole 600M model, so shipping both would store every weight
               twice. Packages:
              - config.yaml             (paths rewritten, nllb.trainable → none)
              - encoder.pt / encoder_config.yaml
              - vocab.json              (the CTC compressor needs it)
              - projector.pt            (from checkpoint dir)
              - nllb/                   (merged NLLB-200 + tokenizer, via
                                          save_pretrained — no nllb_trainable.pt)
              - meta.json / training_meta.json / README.md

The exported directory is consumed by the installed `st` package — install the
repo first (`pip install git+...`), then run the bundled console commands
(`ctc-encoder`, `speech-aura`, `speech-aura-mic`) or `python -m st.inference.*`.
The exported config's relative paths are rewritten automatically on export.

Usage:
    # Stage 1 encoder
    python scripts/export_checkpoint.py encoder \
        --checkpoint runs/stage1_23_lang/encoder_step96000.pt \
        --output exports/ctc_encoder_23lang

    # Stage 1 encoder, richer model card (languages + provenance from YAML)
    python scripts/export_checkpoint.py encoder \
        --checkpoint runs/stage1_23_lang/encoder_step96000.pt \
        --config configs/experiment/stage1.yaml \
        --output exports/ctc_encoder_23lang

    # SpeechAura (any of stage 2/3/4)
    python scripts/export_checkpoint.py speech_aura \
        --config configs/experiment/stage3.yaml \
        --checkpoint runs/stage3_23_lang/checkpoint_step32407 \
        --output exports/speech_aura_stage3

    # Skip the Aura base (saves ~2GB if you'll point at it externally)
    python scripts/export_checkpoint.py speech_aura \
        --config configs/experiment/stage3.yaml \
        --checkpoint runs/stage3_23_lang/checkpoint_step32407 \
        --output exports/speech_aura_stage3 \
        --skip_aura_base

    # SpeechNLLB (any of stage 2/3/4)
    python scripts/export_checkpoint.py speech_nllb \
        --config configs/stt/experiment/stage4_st_nllb.yaml \
        --checkpoint runs/stage4_nllb_st_v2/checkpoint_step30000 \
        --output exports/speech_nllb_st_v2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import torch
import yaml

from core.utils.config import load_config

log = logging.getLogger(__name__)

# Default repo URL shown in generated READMEs. Override with --repo-url so the
# model card's pip-install line points at the right place.
DEFAULT_REPO_URL = "https://github.com/<you>/speechaura"


# ============================================================================
# Encoder export
# ============================================================================

def export_encoder(
    checkpoint: str,
    output_dir: str,
    config_path: str | None = None,
    repo_url: str = DEFAULT_REPO_URL,
    overwrite: bool = False,
) -> Path:
    """Package a Stage 1 encoder checkpoint into a self-contained directory.

    The checkpoint already bundles encoder_config + vocab; this just unpacks
    them into discrete files and strips optimizer/scheduler state.

    If `config_path` is given, the README is enriched with the training
    language list and data provenance. Architecture and vocab are always
    sourced from the checkpoint (authoritative for the shipped weights), never
    from the YAML, to avoid drift.
    """
    out = _prepare_output_dir(output_dir, overwrite)

    log.info(f"Loading encoder checkpoint ← {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    for required in ("model_state_dict", "encoder_config", "vocab"):
        if required not in ckpt:
            raise ValueError(
                f"Checkpoint {checkpoint} is missing '{required}'. "
                f"Use a Stage 1 checkpoint saved by pretrain_encoder.py."
            )

    # Weights only — drop optimizer / scheduler / step counters
    torch.save({"model_state_dict": ckpt["model_state_dict"]}, out / "encoder.pt")

    with open(out / "encoder_config.yaml", "w") as f:
        yaml.safe_dump(ckpt["encoder_config"], f, sort_keys=False)

    with open(out / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(ckpt["vocab"], f, ensure_ascii=False, indent=2)

    # Optional enrichment from the experiment YAML — languages + provenance only.
    languages: list[str] | None = None
    provenance: dict | None = None
    if config_path:
        cfg = load_config(config_path)
        data_cfg = cfg.get("data", {})
        languages = data_cfg.get("languages") or data_cfg.get("src_languages")
        provenance = {
            "train_index":  data_cfg.get("train_index"),
            "max_duration": data_cfg.get("max_duration"),
            "lowercase":    data_cfg.get("lowercase"),
            "source_config": str(config_path),
        }
        log.info(
            f"Enriching README from {config_path}: "
            f"{len(languages) if languages else 0} languages"
        )

    meta = {
        "kind":         "ctc_encoder",
        "step":         ckpt.get("step"),
        "epoch":        ckpt.get("epoch"),
        "vocab_size":   len(ckpt["vocab"]),
        "encoder_dim":  ckpt["encoder_config"].get("encoder_dim"),
        "num_layers":   ckpt["encoder_config"].get("num_layers"),
        "languages":    languages,
        "provenance":   provenance,
        "source_checkpoint": str(checkpoint),
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    _write_encoder_readme(out, meta, repo_url=repo_url)
    _log_export_summary(out)
    return out


# ============================================================================
# SpeechAura export
# ============================================================================

def export_speech_aura(
    config_path: str,
    checkpoint_dir: str,
    output_dir: str,
    skip_aura_base: bool = False,
    repo_url: str = DEFAULT_REPO_URL,
    overwrite: bool = False,
) -> Path:
    """Package a SpeechAura checkpoint + all referenced assets.

    Resolves the encoder checkpoint, vocab, and Aura base/tokenizer from the
    experiment YAML, copies them into `output_dir`, and rewrites the YAML so
    the exported directory is fully self-contained.
    """
    out = _prepare_output_dir(output_dir, overwrite)
    cfg = load_config(config_path)
    ckpt_dir = Path(checkpoint_dir)

    if not ckpt_dir.is_dir():
        raise ValueError(f"checkpoint must be a directory (got {checkpoint_dir})")

    # ---- 1. Copy projector + LoRA / full-LLM weights + training meta ----
    proj_src = ckpt_dir / "projector.pt"
    if not proj_src.exists():
        raise FileNotFoundError(f"Missing {proj_src} — expected a SpeechAura checkpoint directory")
    shutil.copy2(proj_src, out / "projector.pt")
    log.info(f"  + projector.pt           ({_size(proj_src)})")

    lora_src = ckpt_dir / "lora.pt"
    if lora_src.exists():
        shutil.copy2(lora_src, out / "lora.pt")
        log.info(f"  + lora.pt                ({_size(lora_src)})")

    llm_full_src = ckpt_dir / "llm_full.pt"
    if llm_full_src.exists():
        shutil.copy2(llm_full_src, out / "llm_full.pt")
        log.info(f"  + llm_full.pt            ({_size(llm_full_src)})")

    train_meta = {}
    train_meta_src = ckpt_dir / "meta.json"
    if train_meta_src.exists():
        with open(train_meta_src) as f:
            train_meta = json.load(f)
        shutil.copy2(train_meta_src, out / "training_meta.json")

    # ---- 2/3. Encoder weights + config, and the CTC vocab if it is needed ----
    needs_vocab = _export_encoder_and_vocab(cfg, out, config_path)

    # ---- 4. Aura base weights + tokenizer ----
    aura_dir_rel = "aura"
    aura_out = out / aura_dir_rel
    if not skip_aura_base:
        aura_out.mkdir(parents=True, exist_ok=True)

        aura_ckpt_src = Path(cfg["aura"]["checkpoint"])
        aura_tok_src  = Path(cfg["aura"]["tokenizer"])
        if not aura_ckpt_src.exists():
            raise FileNotFoundError(f"Aura checkpoint not found: {aura_ckpt_src}")
        if not aura_tok_src.exists():
            raise FileNotFoundError(f"Aura tokenizer not found: {aura_tok_src}")

        aura_ckpt_dst = aura_out / aura_ckpt_src.name
        aura_tok_dst  = aura_out / "tokenizer.json"

        log.info(f"Copying Aura base ← {aura_ckpt_src}")
        shutil.copy2(aura_ckpt_src, aura_ckpt_dst)
        log.info(f"  + aura/{aura_ckpt_src.name:<18}({_size(aura_ckpt_dst)})")
        shutil.copy2(aura_tok_src, aura_tok_dst)
        log.info(f"  + aura/tokenizer.json    ({_size(aura_tok_dst)})")

        new_aura_ckpt_ref = f"{aura_dir_rel}/{aura_ckpt_src.name}"
        new_aura_tok_ref  = f"{aura_dir_rel}/tokenizer.json"
    else:
        # Keep original absolute paths so a downstream user can wire them up
        new_aura_ckpt_ref = cfg["aura"]["checkpoint"]
        new_aura_tok_ref  = cfg["aura"]["tokenizer"]
        log.info("Skipping Aura base (--skip_aura_base)")

    # ---- 5. Rewritten config.yaml pointing at relative paths inside the export ----
    export_cfg = _rewrite_config_for_export(
        cfg,
        encoder_ref="encoder.pt",
        vocab_ref="vocab.json" if needs_vocab else None,
        aura_ckpt_ref=new_aura_ckpt_ref,
        aura_tok_ref=new_aura_tok_ref,
    )
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(export_cfg, f, sort_keys=False)

    # ---- 6. Export meta + README ----
    meta = {
        "kind":          "speech_aura",
        "task":          cfg["training"].get("task"),
        "stage":         _infer_stage(cfg["training"]),
        "step":          train_meta.get("step"),
        "epoch":         train_meta.get("epoch"),
        "has_lora":      (out / "lora.pt").exists(),
        "has_llm_full":  (out / "llm_full.pt").exists(),
        "has_vocab":     needs_vocab,
        "aura_size":     cfg["aura"].get("size", "1b"),
        "projector":     cfg.get("projector", {}).get("type", "mlp"),
        "ctc_compress":  (cfg.get("ctc_compress") or {}).get("enabled", False),
        "ctc_weight":    cfg["training"].get("ctc_weight", 0.0),
        "languages":     cfg["data"].get("languages") or cfg["data"].get("src_languages"),
        "skip_aura_base": skip_aura_base,
        "source_config": str(config_path),
        "source_checkpoint": str(checkpoint_dir),
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    _write_speech_aura_readme(out, meta, repo_url=repo_url)
    _log_export_summary(out)
    return out


# ============================================================================
# SpeechNLLB export
# ============================================================================

def export_speech_nllb(
    config_path: str,
    checkpoint_dir: str,
    output_dir: str,
    skip_nllb_base: bool = False,
    repo_url: str = DEFAULT_REPO_URL,
    overwrite: bool = False,
) -> Path:
    """Package a SpeechNLLB checkpoint + all referenced assets.

    Unlike the Aura export, the fine-tuned NLLB weights are *merged into* the base
    rather than shipped alongside it. A `trainable: all` checkpoint's
    nllb_trainable.pt is essentially the whole 600M model, so copying both would
    store every weight twice (~4.8G instead of ~2.3G). Merging also collapses the
    partial modes (cross_attn / encoder_cross_attn) to the same layout: whatever
    was not trained simply keeps its base value.

    The export therefore has no nllb_trainable.pt, which SpeechNLLB.load_checkpoint
    already tolerates — it loads the projector and skips the missing file.
    """
    out = _prepare_output_dir(output_dir, overwrite)
    cfg = load_config(config_path)
    ckpt_dir = Path(checkpoint_dir)

    if "nllb" not in cfg:
        raise ValueError(
            f"{config_path} has no `nllb:` block — use the speech_aura mode instead."
        )
    if not ckpt_dir.is_dir():
        raise ValueError(f"checkpoint must be a directory (got {checkpoint_dir})")

    # ---- 1. Projector + training meta ----
    proj_src = ckpt_dir / "projector.pt"
    if not proj_src.exists():
        raise FileNotFoundError(
            f"Missing {proj_src} — expected a SpeechNLLB checkpoint directory"
        )
    shutil.copy2(proj_src, out / "projector.pt")
    log.info(f"  + projector.pt           ({_size(proj_src)})")

    train_meta = {}
    train_meta_src = ckpt_dir / "meta.json"
    if train_meta_src.exists():
        with open(train_meta_src) as f:
            train_meta = json.load(f)
        shutil.copy2(train_meta_src, out / "training_meta.json")

    # ---- 2/3. Encoder weights + config, and the CTC vocab if it is needed ----
    needs_vocab = _export_encoder_and_vocab(cfg, out, config_path)

    # ---- 4. NLLB base with the fine-tuned weights merged in ----
    nllb_dir_rel = "nllb"
    if not skip_nllb_base:
        _export_merged_nllb(cfg, ckpt_dir, out / nllb_dir_rel)
        new_nllb_ref = nllb_dir_rel
    else:
        # Keep the original path so a downstream user can wire it up themselves.
        # The trained weights then have to travel as nllb_trainable.pt.
        trainable_src = ckpt_dir / "nllb_trainable.pt"
        if trainable_src.exists():
            shutil.copy2(trainable_src, out / "nllb_trainable.pt")
            log.info(f"  + nllb_trainable.pt      ({_size(trainable_src)})")
        new_nllb_ref = cfg["nllb"]["model"]
        log.info("Skipping NLLB base (--skip_nllb_base)")

    # ---- 5. Rewritten config.yaml pointing at relative paths inside the export ----
    export_cfg = _rewrite_config_for_export(
        cfg,
        encoder_ref="encoder.pt",
        vocab_ref="vocab.json" if needs_vocab else None,
        nllb_ref=new_nllb_ref,
    )
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(export_cfg, f, sort_keys=False)

    # ---- 6. Export meta + README ----
    meta = {
        "kind":           "speech_nllb",
        "task":           cfg["training"].get("task"),
        "stage":          _infer_nllb_stage(cfg["nllb"]),
        "step":           train_meta.get("step"),
        "epoch":          train_meta.get("epoch"),
        "attach":         cfg["nllb"].get("attach", "encoder_input"),
        "nllb_trainable": cfg["nllb"].get("trainable", "none"),
        "nllb_base":      cfg["nllb"]["model"],
        "has_vocab":      needs_vocab,
        "projector":      cfg.get("projector", {}).get("type", "mlp"),
        "ctc_compress":   (cfg.get("ctc_compress") or {}).get("enabled", False),
        "ctc_weight":     cfg["training"].get("ctc_weight", 0.0),
        "languages":      cfg["data"].get("languages") or cfg["data"].get("src_languages"),
        "skip_nllb_base": skip_nllb_base,
        "source_config":  str(config_path),
        "source_checkpoint": str(checkpoint_dir),
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    _write_speech_nllb_readme(out, meta, repo_url=repo_url)
    _log_export_summary(out)
    return out


def _export_merged_nllb(cfg: dict, ckpt_dir: Path, nllb_out: Path) -> None:
    """Load the NLLB base, apply the checkpoint's trained weights, save the result.

    Loads on CPU: the base (~2.3G) plus the state dict (~2.3G) peaks around 5-6G.
    """
    from st.models.nllb import NLLBSeq2Seq

    base_path = cfg["nllb"]["model"]
    if not Path(base_path).exists():
        raise FileNotFoundError(
            f"NLLB base not found at {base_path!r} (from the config's nllb.model)"
        )

    log.info(f"Loading NLLB base ← {base_path}")
    nllb = NLLBSeq2Seq(model_path=base_path, trainable="none")

    trainable_src = ckpt_dir / "nllb_trainable.pt"
    if trainable_src.exists():
        log.info(f"Merging fine-tuned NLLB weights ← {trainable_src} "
                 f"({_size(trainable_src)})")
        state = torch.load(trainable_src, map_location="cpu", weights_only=True)
        missing, unexpected = nllb.model.load_state_dict(state, strict=False)
        # `missing` is expected for the partial trainable modes — those keys keep
        # their base values. `unexpected` is not: it means the state dict does not
        # belong to this base, and the merge would silently drop trained weights.
        log.info(f"  merged {len(state):,} tensors; "
                 f"{len(missing):,} kept from base, {len(unexpected):,} unexpected")
        if unexpected:
            raise ValueError(
                f"{trainable_src} has {len(unexpected)} keys that do not exist in "
                f"the NLLB base (first few: {unexpected[:5]}). Wrong base model?"
            )
    else:
        # Stage 2 (trainable: none) writes no nllb_trainable.pt — the base is the
        # model, and only the projector was trained.
        log.info(f"No nllb_trainable.pt in {ckpt_dir} — exporting the pristine base "
                 f"(expected for a projector-only checkpoint).")

    nllb_out.mkdir(parents=True, exist_ok=True)
    nllb.model.save_pretrained(nllb_out)
    nllb.tokenizer.save_pretrained(nllb_out)
    log.info(f"  + {str(nllb_out.name) + '/':<22} ({_dir_size(nllb_out)})")


# ============================================================================
# Helpers
# ============================================================================

def _export_encoder_and_vocab(cfg: dict, out: Path, config_path: str) -> bool:
    """Write encoder.pt + encoder_config.yaml, and vocab.json if CTC needs it.

    Shared by the speech_aura and speech_nllb modes — both front the same Stage 1
    encoder and the same CTC compressor. Returns whether a vocab was written, which
    the caller needs in order to rewrite `data.vocab_path`.
    """
    enc_ckpt_path = cfg["encoder"].get("checkpoint")
    if not enc_ckpt_path or not Path(enc_ckpt_path).exists():
        raise FileNotFoundError(
            f"Encoder checkpoint not found at {enc_ckpt_path!r} "
            f"(from {config_path}:encoder.checkpoint)"
        )
    log.info(f"Loading encoder ← {enc_ckpt_path}")
    enc_ckpt = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)

    # Trust the YAML's encoder block as the source of truth for architecture —
    # this is what training uses to instantiate the encoder. Fall back to the
    # checkpoint's bundled config only for fields the YAML omits.
    enc_cfg = {**enc_ckpt.get("encoder_config", {}), **cfg["encoder"]}
    enc_cfg.pop("checkpoint", None)  # path field, not an architecture field

    torch.save({"model_state_dict": enc_ckpt["model_state_dict"]}, out / "encoder.pt")
    log.info(f"  + encoder.pt             ({_size(out / 'encoder.pt')})")
    with open(out / "encoder_config.yaml", "w") as f:
        yaml.safe_dump(enc_cfg, f, sort_keys=False)

    needs_vocab = (
        cfg["training"].get("ctc_weight", 0.0) > 0
        or (cfg.get("ctc_compress") or {}).get("enabled", False)
    )
    if needs_vocab:
        vocab_path = cfg["data"].get("vocab_path")
        vocab: dict[str, int] | None = None
        if vocab_path and Path(vocab_path).exists():
            with open(vocab_path) as f:
                vocab = json.load(f)
        elif "vocab" in enc_ckpt:
            vocab = enc_ckpt["vocab"]
        if vocab is None:
            raise FileNotFoundError(
                f"CTC compressor / aux loss enabled but no vocab found "
                f"(checked data.vocab_path={vocab_path!r} and encoder ckpt)."
            )
        with open(out / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        log.info(f"  + vocab.json             ({len(vocab)} tokens)")

    return needs_vocab


def _prepare_output_dir(output_dir: str, overwrite: bool) -> Path:
    out = Path(output_dir)
    if out.exists():
        if not overwrite:
            raise FileExistsError(
                f"{out} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(out)
    out.mkdir(parents=True)
    log.info(f"Export directory: {out}")
    return out


def _rewrite_config_for_export(
    cfg: dict,
    encoder_ref: str,
    vocab_ref: str | None,
    aura_ckpt_ref: str | None = None,
    aura_tok_ref: str | None = None,
    nllb_ref: str | None = None,
) -> dict:
    """Return a copy of cfg with paths rewritten to point inside the export dir.

    Handles both backbones: a config has either an `aura:` block or an `nllb:`
    block (build_model enforces exactly one), so the refs for the other are None.

    Training-time-only fields (output_dir, init_from, wandb settings, train/val
    index paths) are dropped — the exported config is for inference, not for
    resuming training.
    """
    out_cfg = json.loads(json.dumps(cfg))  # deep copy via json round-trip

    out_cfg["encoder"]["checkpoint"] = encoder_ref

    if aura_ckpt_ref is not None:
        out_cfg["aura"]["checkpoint"] = aura_ckpt_ref
    if aura_tok_ref is not None:
        out_cfg["aura"]["tokenizer"] = aura_tok_ref
    if nllb_ref is not None:
        out_cfg["nllb"]["model"] = nllb_ref

    if "nllb" in out_cfg:
        # The exported NLLB weights are already fine-tuned, and inference needs no
        # gradients — so nothing should be marked trainable on load.
        out_cfg["nllb"]["trainable"] = "none"
        out_cfg.get("training", {}).pop("gradient_checkpointing", None)

    if vocab_ref is not None:
        out_cfg.setdefault("data", {})["vocab_path"] = vocab_ref
    else:
        out_cfg.get("data", {}).pop("vocab_path", None)

    # Strip training-time-only fields. The exported config is inference-only;
    # leaving these in would just create broken paths for downstream users.
    # `init_from` matters especially for the NLLB path — _build_speech_nllb loads
    # it eagerly, so leaving it in makes the export depend on a runs/ dir again.
    train = out_cfg.get("training", {})
    for k in ("output_dir", "projector_checkpoint", "init_from",
              "wandb_project", "wandb_run_name", "wandb_entity", "no_wandb"):
        train.pop(k, None)

    data = out_cfg.get("data", {})
    for k in ("train_index", "val_index", "train_split", "val_split"):
        data.pop(k, None)

    out_cfg.pop("wandb", None)
    return out_cfg


def _infer_stage(train_cfg: dict) -> str:
    """Best-effort label: stage2 / stage3 / stage4_full / stage4_llm."""
    unfreeze_enc = train_cfg.get("unfreeze_encoder", False)
    unfreeze_llm = train_cfg.get("unfreeze_llm", False)
    lora_rank    = train_cfg.get("lora_rank", 0)
    if not unfreeze_enc and not unfreeze_llm and lora_rank == 0:
        return "stage2"
    if not unfreeze_enc and not unfreeze_llm and lora_rank > 0:
        return "stage3"
    if unfreeze_enc and unfreeze_llm:
        return "stage4_full"
    if not unfreeze_enc and unfreeze_llm:
        return "stage4_llm"
    return "custom"


def _infer_nllb_stage(nllb_cfg: dict) -> str:
    """Best-effort label from the NLLB curriculum's trainable mode.

    _infer_stage reads lora_rank / unfreeze_llm, which the NLLB path never sets —
    here the stage is carried entirely by `nllb.trainable`.
    """
    return {
        "none":               "stage2",       # projector only
        "cross_attn":         "stage3",       # + decoder cross-attention
        "encoder_cross_attn": "stage3",       # + text encoder
        "decoder":            "stage4",       # + full decoder
        "all":                "stage4_full",  # whole NLLB
    }.get(nllb_cfg.get("trainable", "none"), "custom")


def _size(p: Path) -> str:
    n = p.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _log_export_summary(out: Path) -> None:
    log.info("")
    log.info(f"Export complete: {out}")
    total = 0
    for p in sorted(out.rglob("*")):
        if p.is_file():
            s = p.stat().st_size
            total += s
            log.info(f"  {p.relative_to(out)}  ({_size(p)})")
    log.info(f"  ─────")
    log.info(f"  total: {_size_int(total)}")


def _size_int(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _dir_size(p: Path) -> str:
    return _size_int(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()))


def _format_language_section(languages: list[str] | None) -> str:
    """Render a markdown language list, or a fallback pointer to vocab.json."""
    if not languages:
        return "(language list not recorded — pass --config on export to include it)"
    lines = [f"- {lang}" for lang in languages]
    return f"{len(languages)} languages:\n\n" + "\n".join(lines)


def _write_encoder_readme(out: Path, meta: dict, repo_url: str = DEFAULT_REPO_URL) -> None:
    lang_section = _format_language_section(meta.get("languages"))

    provenance_block = ""
    prov = meta.get("provenance")
    if prov:
        provenance_block = (
            "\n## Training provenance\n\n"
            f"- train index: `{prov.get('train_index')}`\n"
            f"- max duration: {prov.get('max_duration')}s\n"
            f"- lowercase: {prov.get('lowercase')}\n"
            f"- source config: `{prov.get('source_config')}`\n"
        )

    txt = f"""# CTC Encoder Checkpoint

Multilingual African speech CTC encoder (IWSLT 2026).

## Files

- `encoder.pt`            — model weights (state_dict)
- `encoder_config.yaml`   — architecture (input_dim, encoder_dim, num_layers, ...)
- `vocab.json`            — CTC character vocabulary (index 0 = blank)
- `meta.json`             — training step / language coverage

## Stats

- encoder_dim: {meta.get('encoder_dim')}
- num_layers:  {meta.get('num_layers')}
- vocab_size:  {meta.get('vocab_size')}
- step:        {meta.get('step')}

## Languages

{lang_section}
{provenance_block}
## Usage

Install the package, then run the bundled console command:

```bash
pip install git+{repo_url}

# Download this checkpoint dir (or clone the HF repo), then:
ctc-encoder --checkpoint encoder.pt --audio test.wav
```

Live from the microphone (needs the `mic` extra + system PortAudio):

```bash
pip install "speechaura[mic] @ git+{repo_url}"
# conda install -c conda-forge portaudio   # if PortAudio isn't present
speech-aura-mic encoder --checkpoint encoder.pt
```

Equivalent module form (no console script): `python -m st.inference.ctc_generate
--checkpoint encoder.pt --audio test.wav`.

Or rebuild the encoder in Python:

```python
import json, yaml, torch
from st.models.encoder import SpeechEncoder

with open("encoder_config.yaml") as f: cfg = yaml.safe_load(f)
with open("vocab.json")         as f: vocab = json.load(f)

encoder = SpeechEncoder(**{{k: v for k, v in cfg.items() if k != "checkpoint"}},
                        vocab_size=len(vocab))
encoder.load_state_dict(torch.load("encoder.pt", weights_only=True)["model_state_dict"])
encoder.eval()
```
"""
    (out / "README.md").write_text(txt)


def _write_speech_aura_readme(out: Path, meta: dict, repo_url: str = DEFAULT_REPO_URL) -> None:
    lang_section = _format_language_section(meta.get("languages"))

    txt = f"""# SpeechAura Checkpoint

End-to-end speech translation model (IWSLT 2026).

## Files

- `config.yaml`           — inference config (paths rewritten to this directory)
- `encoder.pt`            — Stage 1 CTC encoder weights
- `encoder_config.yaml`   — encoder architecture
- `projector.pt`          — projector weights
- `lora.pt`               — (if Stage 3) LoRA adapter weights
- `llm_full.pt`           — (if Stage 4 full FT) fine-tuned LLM weights
- `vocab.json`            — (if CTC compressor / aux CTC loss enabled)
- `aura/`                 — Aura base LLM (model file + tokenizer.json)
- `training_meta.json`    — original training position
- `meta.json`             — export manifest

## Stats

- stage:        {meta.get('stage')}
- task:         {meta.get('task')}
- step:         {meta.get('step')}
- aura_size:    {meta.get('aura_size')}
- projector:    {meta.get('projector')}
- ctc_compress: {meta.get('ctc_compress')}
- has_lora:     {meta.get('has_lora')}
- has_llm_full: {meta.get('has_llm_full')}

## Languages

{lang_section}

## Usage

Install the package first, then run from inside this directory:

```bash
pip install git+{repo_url}

speech-aura --config config.yaml --checkpoint . \\
    --audio test.wav --src_language igbo --task asr
```

Direct speech translation (Yoruba → English):

```bash
speech-aura --config config.yaml --checkpoint . \\
    --audio test.wav --src_language yoruba --tgt_language english --task st
```

Live from the microphone (needs the `mic` extra + system PortAudio):

```bash
pip install "speechaura[mic] @ git+{repo_url}"
# conda install -c conda-forge portaudio   # if PortAudio isn't present
speech-aura-mic speech_aura --config config.yaml --checkpoint . \\
    --src-lang yoruba --tgt-lang english --task st --loop
```

Equivalent module form: `python -m st.inference.generate --config config.yaml
--checkpoint . --audio test.wav --src_language igbo --task asr`.

The config's `encoder.checkpoint`, `aura.checkpoint`, `aura.tokenizer`, and
`data.vocab_path` are already rewritten to point at files in this directory.
Run the command from inside the extracted directory.
"""
    (out / "README.md").write_text(txt)


def _write_speech_nllb_readme(out: Path, meta: dict, repo_url: str = DEFAULT_REPO_URL) -> None:
    lang_section = _format_language_section(meta.get("languages"))

    txt = f"""# SpeechNLLB Checkpoint

End-to-end speech translation with an NLLB-200 decoder (IWSLT 2026).

    audio → Conformer encoder → CTC compressor → projector → NLLB-200 → translation

## Files

- `config.yaml`           — inference config (paths rewritten to this directory)
- `encoder.pt`            — Stage 1 CTC encoder weights
- `encoder_config.yaml`   — encoder architecture
- `projector.pt`          — projector weights
- `vocab.json`            — CTC character vocabulary (the compressor needs it)
- `nllb/`                 — NLLB-200 with the fine-tuned weights already merged in,
                            plus its tokenizer. There is no separate weight delta.
- `training_meta.json`    — original training position
- `meta.json`             — export manifest

## Stats

- stage:        {meta.get('stage')}
- task:         {meta.get('task')}
- step:         {meta.get('step')}
- attach:       {meta.get('attach')}
- trained:      {meta.get('nllb_trainable')} (NLLB params that received gradients)
- projector:    {meta.get('projector')}
- ctc_compress: {meta.get('ctc_compress')}
- base model:   `{meta.get('nllb_base')}`

## Languages

{lang_section}

## Usage

Install the package first, then run from inside this directory:

```bash
pip install git+{repo_url}

speech-aura --config config.yaml --checkpoint . \\
    --audio test.wav --src_language yoruba --tgt_language english --task st
```

The same console script drives both backbones — `build_model` dispatches on the
config's `nllb:` block. Note this model is speech *translation* only: NLLB emits the
translation with no transcript, so `--task asr` has nothing to return.

Equivalent module form: `python -m st.inference.generate --config config.yaml
--checkpoint . --audio test.wav --src_language yoruba --task st`.

The config's `encoder.checkpoint`, `nllb.model`, and `data.vocab_path` are rewritten
to point at files in this directory, and are anchored to `--checkpoint` at load time,
so the commands work from any working directory.
"""
    (out / "README.md").write_text(txt)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Export a checkpoint for HF upload")
    sub    = parser.add_subparsers(dest="kind", required=True)

    p_enc = sub.add_parser("encoder", help="Export a Stage 1 CTC encoder")
    p_enc.add_argument("--checkpoint", required=True,
                       help="Stage 1 encoder .pt (must contain encoder_config + vocab)")
    p_enc.add_argument("--output",     required=True, help="Output directory")
    p_enc.add_argument("--config",     default=None,
                       help="Optional experiment YAML — enriches the README with the "
                            "training language list + provenance. Architecture/vocab "
                            "always come from the checkpoint.")
    p_enc.add_argument("--repo-url",   default=DEFAULT_REPO_URL,
                       help="Git URL shown in the generated README's pip-install line.")
    p_enc.add_argument("--overwrite",  action="store_true")

    p_sa = sub.add_parser("speech_aura", help="Export a SpeechAura (stage 2/3/4) checkpoint")
    p_sa.add_argument("--config",         required=True, help="Experiment YAML used for training")
    p_sa.add_argument("--checkpoint",     required=True, help="Checkpoint directory (contains projector.pt)")
    p_sa.add_argument("--output",         required=True, help="Output directory")
    p_sa.add_argument("--skip_aura_base", action="store_true",
                      help="Don't copy Aura base weights/tokenizer (saves ~2GB)")
    p_sa.add_argument("--repo-url",       default=DEFAULT_REPO_URL,
                      help="Git URL shown in the generated README's pip-install line.")
    p_sa.add_argument("--overwrite",      action="store_true")

    p_sn = sub.add_parser("speech_nllb", help="Export a SpeechNLLB (stage 2/3/4) checkpoint")
    p_sn.add_argument("--config",         required=True, help="Experiment YAML used for training")
    p_sn.add_argument("--checkpoint",     required=True, help="Checkpoint directory (contains projector.pt)")
    p_sn.add_argument("--output",         required=True, help="Output directory")
    p_sn.add_argument("--skip_nllb_base", action="store_true",
                      help="Don't write the merged NLLB base (~2.3GB); keep the original "
                           "nllb.model path and ship nllb_trainable.pt alongside instead")
    p_sn.add_argument("--repo-url",       default=DEFAULT_REPO_URL,
                      help="Git URL shown in the generated README's pip-install line.")
    p_sn.add_argument("--overwrite",      action="store_true")

    args = parser.parse_args()

    if args.kind == "encoder":
        export_encoder(
            args.checkpoint, args.output,
            config_path=args.config,
            repo_url=args.repo_url,
            overwrite=args.overwrite,
        )
    elif args.kind == "speech_nllb":
        export_speech_nllb(
            args.config, args.checkpoint, args.output,
            skip_nllb_base=args.skip_nllb_base,
            repo_url=args.repo_url,
            overwrite=args.overwrite,
        )
    else:
        export_speech_aura(
            args.config, args.checkpoint, args.output,
            skip_aura_base=args.skip_aura_base,
            repo_url=args.repo_url,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()