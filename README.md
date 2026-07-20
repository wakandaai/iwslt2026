# Aura-ASR v1

End-to-end speech translation, targeting African languages (Bemba, Hausa,
Igbo, Yoruba → English), built on Meta's `omniASR_CTC_1B` encoder + Aura-1B
LLM. Standalone snapshot — deliberately self-contained: no shared `core/`
package, no TTS, no NLLB.

## Status

**Stage 1 is complete.** Single H100, resumed + `WeightedLanguageSampler` +
online W&B: finished all 20k steps with zero OOM
(`max_batch_duration=35s / max_batch_size=6 / grad_accum=4`, ~0.8s/step),
final `val/wer=0.3265` (English best at 0.062, Yoruba/Tigrinya hardest at
0.63/0.57). Final checkpoint:
`runs/stage1_omniasr_ctc_22lang_h100_weighted/encoder_step20000.pt` (in the
original `iwslt2026` checkout, not copied into this repo).

**Stages 2-4 are built but unvalidated — no run has happened yet.**
`configs/experiment/stage{2,3,4}/` each hold a v1-scoped (22-language)
config plus historical smoke-test configs. The chain is: Stage 2 aligns the
projector (encoder frozen, computed live — not cached, since caching all 22
languages would need ~4TB against a filesystem with only ~4.4TB free); Stage
3 adds LoRA on top (encoder still frozen); Stage 4 unfreezes the encoder too
(LoRA continues via `--resume_from`). Batch settings everywhere are
reasoned-not-validated guesses. See `docs/stage1_omniasr_ctc_guide.md` for
the full design rationale and update history per stage.

The Conformer-based training path (`pretrain_ctc.py`, `train_st.py`'s
default `encoder.type: conformer`) still exists in the code but has no
config shipped for it anymore — it was removed as no longer useful once the
omniASR path covered all 4 stages. `st/models/encoder.py` (SpeechEncoder)
and `st/inference/generate.py` (mel-only, not omniASR-compatible yet) are
still present but currently unexercised by any shipped config.

## Architecture

```
Raw 16kHz audio → omniASR_CTC_1B
                        │
              (CTCCompressor — optional)
                        │
                Projector (MLP | Transformer)
                        │
                    Aura-1B
                        │
              Transcript / Translation
```

**Training stages:**

| Stage | What trains | Config | Status |
|---|---|---|---|
| 1 — CTC Pretraining | omniASR_CTC_1B encoder | `configs/experiment/stage1/`, `stage1_omniasr_ctc_h100_weighted.yaml` | **done — val/wer=0.3265** |
| 2 — Projector Alignment | Projector only, frozen encoder (live) | `configs/experiment/stage2/` | built, **not yet validated** |
| 3 — LoRA Fine-tuning | Projector + LoRA, frozen encoder | `configs/experiment/stage3/` | built, **not yet validated** |
| 4 — Full Fine-tuning | Encoder unfrozen + Projector + LoRA | `configs/experiment/stage4/` | built, **not yet validated** |

**CTC Compressor (optional):**
Sits between encoder and projector. Uses the encoder's CTC predictions to
merge consecutive frames predicted as the same token, removing blank frames
and reducing sequence length before the LLM. Enabled with `ctc_compress.enabled: true`.

**Auxiliary CTC loss:**
Not usable on this path — omniASR's own CTC head outputs over its
9812-piece SentencePiece vocab, incompatible as an auxiliary loss target
for the LLM stages. `ctc_weight: 0.0` is required (enforced by
`build_model()`'s `omniasr_live` guard) for Stages 2-4.

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

### Stage 1: Fine-tune omniASR_CTC_1B

Fine-tunes `omniASR_CTC_1B` (975M params, fairseq2/Wav2Vec2-based) directly
against a standalone CTC loss on its own 9812-piece SentencePiece vocab —
everything else (projector, compressor, Aura-1B LLM) never loads in this
path. See `docs/stage1_omniasr_ctc_guide.md` for the full design rationale
and run history.

**Must run under an isolated `torch==2.8.0+cu128` env** (`fairseq2n` is
incompatible with this repo's main `torch==2.6.0+cu124` stack). A ready-made
one lives at `/ocean/projects/cis250145p/tanghang/iwslt2026/.envs/omniasr_extract`.

```bash
# Single GPU
<omniasr_extract>/bin/python -m st.training.pretrain_omniasr_ctc \
    --config configs/experiment/stage1/stage1_omniasr_ctc_22lang_v100.yaml

# Multi-GPU DDP
<omniasr_extract>/bin/torchrun --standalone --nproc_per_node=2 \
    -m st.training.pretrain_omniasr_ctc \
    --config configs/experiment/stage1/stage1_omniasr_ctc_22lang_ddp_weighted.yaml

# Resume
<omniasr_extract>/bin/python -m st.training.pretrain_omniasr_ctc \
    --config configs/experiment/stage1_omniasr_ctc_h100_weighted.yaml \
    --resume_from /path/to/encoder_step3500.pt
```

SLURM launchers live in `scripts/stage1_omniasr_ctc/` (production) and
`scripts/stage1_omniasr_ctc/smoke/` (validation smoke tests) — copy and
adjust the checkpoint/venv paths at the top of each if your checkpoints
live somewhere other than the shared path baked in.

Pretrained checkpoint + tokenizer (not committed — 3.7GB binary) are
expected at:
```
/ocean/projects/cis250145p/tanghang/iwslt2026/checkpoints/omniasr_ctc_1b/omniASR-CTC-1B.pt
/ocean/projects/cis250145p/tanghang/iwslt2026/checkpoints/omniasr_ctc_1b/omniASR_tokenizer.model
```

### Stage 2: Train projector
```bash
sbatch scripts/stage2_omniasr_live/train_stage2_omniasr_live_22lang.sbatch
```
`stage2/smoke/stage2_omniasr_cached_smoke.yaml` is the original 4-language
cached-features validation run this was based on (kept as historical
reference for the caching approach that was dropped — see Status above).

### Stage 3: Train projector + LoRA
```bash
sbatch scripts/stage3_omniasr_live/train_stage3_omniasr_live_22lang.sbatch
```
Chains off Stage 2's projector output via `training.projector_checkpoint`
in the config, which doesn't exist yet.

### Stage 4: Unfreeze encoder
```bash
sbatch scripts/stage4_omniasr_live/train_stage4_omniasr_live_22lang.sbatch
```
Set `RESUME_FROM` at the top of that script to Stage 3's real checkpoint
directory once it exists (chains the LoRA adapters forward via
`--resume_from`). `stage4/smoke/` holds the original 4-language
validation-scale configs that proved gradients actually reach the encoder.

### Tests
```bash
pytest tests/ -v
```

`test_forward.py` covers the (now configless) Conformer/projector/
compressor/collator/sampler stack (CPU, synthetic data, no real weights).
`test_omniasr_ctc.py` covers the Stage 1 omniASR path — `RawAudioDataset`,
`CTCRawAudioCollator`, the DDP-aware and weighted samplers,
`build_omniasr_encoder_from_config`'s wiring (mocked, no real checkpoint
needed), and a sanity check over every shipped Stage 1 config.

## Project Structure

```
aura-asr-v1/
├── configs/experiment/
│   ├── stage1/                       # omniASR_CTC_1B configs (production) — validated
│   │   └── smoke/
│   ├── stage2/                       # projector alignment, frozen encoder (live) — unvalidated
│   │   └── smoke/                    # historical 4-language cached-features run
│   ├── stage3/                       # + LoRA, encoder still frozen — unvalidated
│   ├── stage4/                       # encoder unfrozen, LoRA continues — unvalidated
│   │   └── smoke/                    # historical 4-language validation configs
│   └── stage1_omniasr_ctc_h100_weighted.yaml   # the validated production run
│
├── docs/
│   ├── stage1_omniasr_ctc_guide.md  # design doc, run history, per-stage update notes
│   └── compare_ctc_vs_pipeline.py   # one-off diagnostic tied to a specific historical run — kept for reference, not active tooling
│
├── src/st/
│   ├── models/
│   │   ├── encoder.py               # SpeechEncoder (Conformer) — no config exercises this anymore
│   │   ├── omniasr_encoder.py       # OmniASREncoder (omniASR_CTC_1B wrapper)
│   │   ├── projector.py             # MLPProjector, TransformerProjector
│   │   ├── ctc_compressor.py        # CTCCompressor (optional frame merging)
│   │   ├── aura.py                  # AuraLLM wrapper (load, freeze, LoRA)
│   │   ├── llama3.py                # LLaMA-style transformer (RoPE, GQA, SwiGLU, KV cache)
│   │   ├── kvcache.py               # KVcache used by llama3.py's generate()
│   │   ├── model_factory.py         # model_presets (llama-iwslt sizes) + checkpoint loader
│   │   └── speech_aura.py           # SpeechAura: full model + forward + generate
│   │
│   ├── data/
│   │   ├── dataset.py               # SpeechDataset, CachedFeatureDataset, RawAudioDataset
│   │   ├── collator.py              # AuraCollator, CachedFeatureCollator, RawAudioCollator, CTCRawAudioCollator
│   │   ├── sampler.py               # DurationBucketSampler, WeightedLanguageSampler, WeightedPartitionSampler
│   │   └── vocab.py                 # CTC vocab build/save/load
│   │
│   ├── training/
│   │   ├── pretrain_ctc.py          # Conformer Stage 1 loop — no config ships for this anymore
│   │   ├── pretrain_omniasr_ctc.py  # Stage 1 (omniASR_CTC_1B) training loop — validated
│   │   └── train_st.py              # Stage 2/3/4 training loop (drives every omniASR stage 2-4 config)
│   │
│   ├── inference/
│   │   └── generate.py              # CLI inference — mel-only, not omniASR-compatible yet
│   │
│   └── utils/
│       ├── audio.py                 # build_feature_extractor
│       ├── metrics.py               # WER, BLEU, chrF
│       ├── schedulers.py            # CosineAnnealingWarmupRestarts
│       ├── config.py                # load_config, merge_configs
│       └── ddp_utils.py             # DDP setup/teardown/reduce/barrier
│
├── scripts/
│   ├── stage1_omniasr_ctc/          # SLURM launchers (Stage 1) — validated
│   │   └── smoke/
│   ├── stage2_omniasr_live/         # SLURM launcher (Stage 2) — unvalidated
│   ├── stage3_omniasr_live/         # SLURM launcher (Stage 3) — unvalidated
│   ├── stage4_omniasr_live/         # SLURM launcher (Stage 4) — unvalidated
│   ├── smoke_test_cached.py         # Stage 2 cached-mode smoke test (historical)
│   ├── smoke_test_omniasr_live.py   # Stage 4 live-encoder smoke test
│   ├── extract_omniasr_features.py  # offline feature extraction (historical cached-mode tool)
│   ├── corpus.py                    # per-language corpus stats
│   └── infer.sh                     # inference launcher (Conformer-only, see generate.py note)
├── tests/                           # pytest smoke tests (test_forward.py, test_omniasr_ctc.py)
└── pyproject.toml
```

## Config System

All experiment configs are plain YAML under `configs/experiment/`. Swap
configs to change stage/settings. Key sections:

- **encoder**: `type: omniasr_live` + checkpoint path + dropout/freeze settings
- **aura**: Aura-1B checkpoint path + size
- **projector**: `type: mlp | transformer` + transformer hyperparams
- **ctc_compress**: `enabled`, `strategy`, `remove_blanks`
- **data**: Index CSV path + language filter (`null` = all 22) + duration limits
- **training**: LR, steps, freeze flags, LoRA rank, W&B

To disable CTC compression: set `ctc_compress: null` or `ctc_compress.enabled: false`.
