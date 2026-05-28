"""
Export a trained checkpoint into a single self-contained directory for HF upload.

Two modes:

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

from st.utils.config import load_config

log = logging.getLogger(__name__)

# Default repo URL shown in generated READMEs. Override with --repo-url so the
# model card's pip-install line points at the right place.
DEFAULT_REPO_URL = "https://github.com/<you>/iwslt2026"


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

    # ---- 2. Copy encoder weights + repack as a clean inference checkpoint ----
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

    # ---- 3. Vocab (only needed if CTC loss or CTC compressor is active) ----
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
        aura_ckpt_ref=new_aura_ckpt_ref,
        aura_tok_ref=new_aura_tok_ref,
        vocab_ref="vocab.json" if needs_vocab else None,
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
# Helpers
# ============================================================================

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
    aura_ckpt_ref: str,
    aura_tok_ref: str,
    vocab_ref: str | None,
) -> dict:
    """Return a copy of cfg with paths rewritten to point inside the export dir.

    Training-time-only fields (output_dir, projector_checkpoint, wandb settings,
    train/val index paths) are dropped — the exported config is for inference,
    not for resuming training.
    """
    out_cfg = json.loads(json.dumps(cfg))  # deep copy via json round-trip

    out_cfg["encoder"]["checkpoint"] = encoder_ref
    out_cfg["aura"]["checkpoint"]    = aura_ckpt_ref
    out_cfg["aura"]["tokenizer"]     = aura_tok_ref

    if vocab_ref is not None:
        out_cfg.setdefault("data", {})["vocab_path"] = vocab_ref
    else:
        out_cfg.get("data", {}).pop("vocab_path", None)

    # Strip training-time-only fields. The exported config is inference-only;
    # leaving these in would just create broken paths for downstream users.
    train = out_cfg.get("training", {})
    for k in ("output_dir", "projector_checkpoint",
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
pip install "iwslt2026[mic] @ git+{repo_url}"
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
pip install "iwslt2026[mic] @ git+{repo_url}"
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

    args = parser.parse_args()

    if args.kind == "encoder":
        export_encoder(
            args.checkpoint, args.output,
            config_path=args.config,
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