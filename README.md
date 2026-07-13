# SpeechAura — End-to-End Speech Translation & TTS (Aura-1B)

End-to-end speech translation and text-to-speech built on the Aura-1B backend,
targeting African languages (Bemba, Hausa, Igbo, Yoruba → English).

## Architecture

```
Audio → Log-Mel → ConvSubsampler(4×) → Conformer Encoder
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

| Stage | What trains | Config |
|---|---|---|
| 1 — CTC Pretraining | Conformer encoder | `configs/experiment/pretrain_ctc.yaml` |
| 2 — Projector Alignment | Projector only | `configs/experiment/stage2.yaml` |
| 3 — LoRA Fine-tuning | Projector + LoRA adapters | `configs/experiment/stage3.yaml` |
| 4 — Full Fine-tuning | Encoder + Projector + Full LLM | `configs/experiment/stage4_full.yaml` |

**CTC Compressor (optional):**
Sits between encoder and projector. Uses the encoder's CTC predictions to
merge consecutive frames predicted as the same token, removing blank frames
and reducing sequence length before the LLM. Enabled with `ctc_compress.enabled: true`.

**Auxiliary CTC loss:**
When `ctc_weight > 0`, a weighted CTC loss on the encoder output is added to
the CE loss. Keeps encoder representations phonetically grounded during ST training.

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

### Stage 1: Pretrain encoder
```bash
bash scripts/pretrain_encoder.sh configs/experiment/pretrain_ctc.yaml
```

### Stage 2: Train projector
```bash
bash scripts/train_stage2.sh configs/experiment/stage2.yaml
```

### Stage 3: Train projector + LoRA
```bash
bash scripts/train_stage3.sh configs/experiment/stage3.yaml
# Resume:
bash scripts/train_stage3.sh configs/experiment/stage3.yaml runs/stage3/checkpoint_step10000
```

### Stage 4: Full fine-tune
```bash
bash scripts/train_stage4.sh configs/experiment/stage4_full.yaml
```

### Inference
```bash
bash scripts/infer.sh \
    configs/experiment/stage3.yaml \
    runs/stage3/checkpoint_step50000 \
    audio.wav \
    igbo \
    transcribe
```

### Tests
```bash
pytest tests/ -v
```

## Project Structure

```
speechaura/
├── configs/
│   ├── encoder/              # Conformer architecture configs
│   └── experiment/           # Full experiment configs (one per stage)
│
├── src/st/
│   ├── models/
│   │   ├── encoder.py        # SpeechEncoder (Conformer + CTC head)
│   │   ├── projector.py      # MLPProjector, TransformerProjector
│   │   ├── ctc_compressor.py # CTCCompressor (optional frame merging)
│   │   ├── aura.py           # AuraLLM wrapper (load, freeze, LoRA)
│   │   └── speech_aura.py    # SpeechAura: full model + forward + generate
│   │
│   ├── data/
│   │   ├── dataset.py        # SpeechDataset (CSV index reader)
│   │   ├── collator.py       # AuraCollator (builds Aura-format batches)
│   │   ├── sampler.py        # DurationBucketSampler
│   │   └── vocab.py          # CTC vocab build/save/load
│   │
│   ├── training/
│   │   ├── pretrain_ctc.py   # Stage 1 training loop
│   │   └── train_st.py       # Stage 2/3/4 training loop
│   │
│   ├── inference/
│   │   └── generate.py       # CLI inference
│   │
│   └── utils/
│       ├── audio.py          # build_feature_extractor
│       ├── metrics.py        # WER, BLEU, chrF
│       ├── schedulers.py     # CosineAnnealingWarmupRestarts
│       └── config.py         # load_config, merge_configs
│
├── scripts/                  # Shell launch scripts
├── tests/                    # pytest smoke tests
└── pyproject.toml
```

## Config System

All experiment configs are plain YAML under `configs/experiment/`. Swap
stages by pointing to a different config. Key sections:

- **encoder**: Architecture + checkpoint path + vocab_size for CTC
- **aura**: Checkpoint path + size
- **projector**: `type: mlp | transformer` + transformer hyperparams
- **ctc_compress**: `enabled`, `strategy`, `remove_blanks`
- **data**: Index CSV paths + language filters + duration limits
- **training**: LR, steps, freeze flags, LoRA rank, ctc_weight, W&B

To disable CTC compression: set `ctc_compress: null` or `ctc_compress.enabled: false`.
To disable auxiliary CTC loss: set `ctc_weight: 0.0` (also remove `vocab_size` from encoder).
