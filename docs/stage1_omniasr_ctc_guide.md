# Aura-ASR v2: Training Strategy
## omniASR_CTC_1B → African Fine-tuned CTC Encoder → Aura-1B LLM

**Goal:** Replace the current Conformer-18 encoder with a fine-tuned `omniASR_CTC_1B` (975M params, Apache 2.0) to achieve SOTA WER on all 27 public African ASR benchmarks (WAXAL×6, FLEURS×15, NCHLT×4, BIG-C×1, BembaSpeech×1) across 23 languages.

---

## 1. Architecture Overview

### Current Aura-ASR Pipeline
```
Log-Mel (80-dim, 10ms hop)
  → Conv Subsampler (4×) → 25Hz
  → Conformer-18 [D=1024, 16h, FFN=4096, kernel=31]
  → CTC head (Linear 1024 → vocab)
  → CTC Compressor (run-length merge, avg, W^T H)
  → Projector (2-layer Transformer, 1024 → 1280)
  → Aura-1B (36L, D=1280, GQA 4KV, SwiGLU, RoPE θ=500k, 64k BPE)
```

### Aura-ASR v2 Pipeline
```
Raw 16kHz audio
  → omniASR_CTC_1B CNN Frontend (320× downsample) → 50Hz
  → omniASR_CTC_1B Transformer [48L, D=1280, 16h, FFN=5120]
  → CTC head (Linear 1280 → vocab_ctc)   ← already trained on 1239 langs, fine-tuned further
  → CTC Compressor (run-length merge, avg, W^T H) [unchanged algorithm]
  → Projector (2-layer Transformer, 1280 → 1280)   ← no dim change!
  → Aura-1B (36L, D=1280, GQA 4KV, SwiGLU, RoPE θ=500k, 64k BPE)
```

**Key dimension insight:** `omniASR_CTC_1B` outputs 1280-dim, which matches Aura-1B's model dim exactly. The projector becomes a 1280→1280 refinement module — no upscaling, cleaner gradient flow.

### Why CTC Compressor over Frame Stacking

OmniASR's LLM variant uses `encoder_stacking` (stack N frames → `dim×N` → linear project). We keep our CTC Compressor because:
- Variable-length compression adapts to speech rate and phone duration
- CTC auxiliary loss provides a training signal aligned with the target text
- Better for morphologically complex African languages (Nguni clicks, tonal Yoruba, agglutinative Swahili) where uniform stacking loses phonological boundaries
- Already implemented and validated in Aura-ASR v1

### Checkpoint Distinction: W2V vs CTC

| Checkpoint | Params | What it is | When to use |
|------------|--------|-----------|-------------|
| `omniASR_W2V_1B` | 965,514,752 | Pure SSL encoder — no task head, no CTC | Starting from scratch with a custom vocab |
| `omniASR_CTC_1B` | 975,065,300 | W2V_1B + CTC head, fine-tuned on 1239 langs | **Our starting point** — already knows ASR |

**Correction (verified 2026-07-10):** the "exactly the CTC head" framing doesn't hold up numerically. The
real CTC head (`final_proj`) is `Linear(1280, 9812)` = 9812×1280 + 9812 = **12,569,172 params** — larger
than the entire stated 9,550,548 param difference between W2V_1B and CTC_1B. That means the two
checkpoints aren't simply "W2V + head bolted on"; the CTC model likely also drops some W2V-only
components (e.g. SSL quantizer codebooks used for contrastive pretraining, not needed once CTC-tuned),
whose removed size partially offsets the added head. We only have the CTC checkpoint downloaded and
inspected, not the W2V one, so the exact accounting is unconfirmed — but "exactly the projection head"
is provably wrong given the real vocab size. This doesn't change the practical recommendation (start
from `omniASR_CTC_1B`, not the SSL checkpoint), just the explanation of the param delta.

Starting from `omniASR_CTC_1B` means the encoder has already been adapted from SSL representations to ASR-discriminative features, and the CTC head already covers our 22/23 African languages. This gives faster convergence (~2–3× fewer steps) vs. starting from the W2V SSL checkpoint.

### omniASR_CTC_1B Architecture Reference

**Verified 2026-07-10** by downloading `omniASR-CTC-1B.pt` (3.7GB, HF `facebook/omniASR-CTC-1B`) and
inspecting the raw state dict directly with `torch.load` — no fairseq2 install required for this check.
fairseq2/omnilingual-asr were briefly installed to attempt a framework-level load but were reverted:
`fairseq2n` hard-requires torch==2.8.0+cu128 and is incompatible with this repo's pinned
torch==2.6.0+cu124 (breaks `torchaudio.models.Conformer`). Any future fairseq2 usage must go in an
isolated venv, never into the shared training env.

| Component | Config | Status |
|-----------|--------|--------|
| Model dim | 1280 | ✅ confirmed (every attn/ffn/layernorm tensor) |
| Transformer layers | 48 | ✅ confirmed (`encoder.layers.0`…`47`) |
| Attention heads | 16 | ⚠️ not stored in the checkpoint — q/k/v/output_proj are fused `(1280,1280)` matrices. 1280/16=80 is a clean head_dim and consistent with fairseq2's wav2vec2 head-count conventions, but this is inferred, not read from weights |
| FFN dim | 5120 | ✅ confirmed (`ffn.inner_proj` = (5120, 1280)) |
| CNN downsampling | 320× (7 layers, strides 5,2,2,2,2,2,2) | ✅ confirmed (conv weight shapes: `(512,1,10)` then six `(512,512,3 or 2)`) |
| Output frame rate | 50Hz (20ms) | consistent with 320× downsample at 16kHz, not independently verified |
| Pre-training | 3.84M hours SSL → ASR fine-tune, 1239 languages | unverified — not present in checkpoint; from public model card / paper claims only |
| Params | 975,065,300 | ✅ confirmed exactly via `sum(p.numel() for p in state_dict.values())` |
| CTC vocab size | **9812** | ✅ confirmed via `final_proj: (9812, 1280)` and independently via the real SentencePiece tokenizer (`sp.get_piece_size() == 9812`) |
| License | Apache 2.0 | from HF model card |
| Framework | fairseq2 (stock `Wav2Vec2` classes — `omnilingual_asr`'s `wav2vec2_asr` module is only a config registration, no custom model code) | ✅ confirmed via repo source |

**Frontend components not previously documented** (found in the state dict, all before the 48 transformer layers):
- `encoder_frontend.feature_extractor` — the 7-layer conv stack, outputs **512-dim**, not 1280.
- `encoder_frontend.model_dim_proj` — `Linear(512 → 1280)`, up-projects conv output to transformer width. This is a real layer that must be accounted for in any adapter; it's easy to miss since the guide's original pipeline diagram jumped straight from the CNN to the 1280-dim transformer.
- `encoder_frontend.post_extract_layer_norm` — LayerNorm at 512-dim, applied right after the conv stack, before `model_dim_proj`.
- `encoder_frontend.pos_encoder.conv` — weight-normalized grouped conv positional encoding, kernel=128 (standard wav2vec2-large-style positional conv, not mentioned in the original guide draft).
- `encoder.layer_norm` — a final LayerNorm after all 48 transformer layers, before `final_proj`.
- `final_proj` — `Linear(1280 → 9812)`, the CTC head. Named `final_proj`, not `ctc_head`.

---

## 2. Encoder Switch: Conformer-18 → omniASR_CTC_1B

### 2.1 Audio Compression Rate Comparison

Both architectures receive raw 16kHz audio, but downsample it at different rates:

| Metric | Conformer-18 (Aura-ASR v1) | omniASR_CTC_1B (Aura-ASR v2) |
|--------|---------------------------|-------------------------------|
| Input | Raw 16kHz audio | Raw 16kHz audio |
| Frontend | Log-Mel (80-dim, 10ms hop) + Conv Subsampler (4×) | CNN Feature Extractor (7 conv layers) |
| Downsampling factor | 640× (from raw samples) | 320× (from raw samples) |
| Output frame rate | 25 Hz (one frame per 40ms) | 50 Hz (one frame per 20ms) |
| Sequence length (30s audio) | 750 frames | 1,500 frames |

The Conformer's 640× compression comes from two stages: the log-mel hop (10ms = 160 samples) + 4× Conv Subsampler = 640× total from raw audio. The W2V CNN compresses directly from raw samples at 320×.

**Implication for the CTC Compressor:** At 50Hz the compressor receives 2× more frames before merging. This is an advantage — finer temporal resolution means more precise phoneme boundary detection. The merged output length to Aura-1B is determined by speech content (CTC blank ratio), not by the input frame rate.

### 2.2 Cost of the Switch

| Component | Change | Cost |
|-----------|--------|------|
| Log-Mel frontend | Eliminated entirely | Zero — simplification, not a cost |
| Conv Subsampler (4×) | Eliminated | Zero — replaced by W2V CNN |
| Conformer-18 (D=1024) | → W2V Transformer-48 (D=1280) | New encoder weights (pre-trained, no training cost) |
| CTC head | `Linear(1024, vocab)` → `Linear(1280, vocab)` | Input dim update; weights pre-trained in `omniASR_CTC_1B` |
| CTC Compressor | Input dim 1024 → 1280 | Code change only, algorithm unchanged |
| Projector | `1024→1280` → `1280→1280` | **Must retrain from scratch** — old weights not transferable |
| Aura-1B | Unchanged | Zero |
| Inference latency | 48 layers vs 18, but local attention | Larger model, similar or faster per-token with optimized kernels |

**The only real cost is the projector.** It must be retrained because its input dimension changes from 1024 to 1280. All other components are either unchanged, simplified, or come pre-trained.

### 2.3 Tokenizer Architecture

Aura-ASR v2 uses two completely independent tokenizers that never interact:

```
Raw audio
  → omniASR_CTC_1B encoder
  → [omniASR_tokenizer_v1 ~2000 chars]  ← CTC loss during training
  →                                         frame merging at inference
  → CTC Compressor → 1280-dim vectors   ← NO TEXT HERE, continuous vectors
  → Projector
  → Aura-1B
  → [Aura-1B BPE 64k subwords]          ← transcript generation
  → "Ninakwenda shuleni kesho"
```

**`omniASR_tokenizer_v1` (encoder side) — CORRECTED 2026-07-10**

The original draft badly understated this. Downloaded the real tokenizer
(`omniASR_tokenizer.model` from the HF repo) and loaded it with `sentencepiece` directly:

```python
>>> sp.get_piece_size()
9812
```

- **Vocab size is 9812, not ~2000.** This matches `final_proj`'s output dim in the checkpoint exactly, so it's confirmed from two independent sources.
- **It is a SentencePiece model, not a clean character-level vocab.** Sampling arbitrary piece IDs turned up CJK characters (e.g. `耷`, `氰`, `賬`) and Cyrillic-like glyphs (`љ`) — this is a broad multilingual vocab spanning many scripts used across OmniASR's full 1239-language training set, not something scoped to "Latin + Ethiopic."
- **No `{iso639-3}_{Script}`-style language ID tokens found** among any of the 9812 pieces. Either language conditioning happens through a different mechanism elsewhere in the OmniASR pipeline (e.g. a separate embedding table keyed by language code, outside the SentencePiece vocab) or this specific claim is simply wrong. Not resolved — we only inspected the tokenizer's piece list, not the full `omnilingual_asr` inference pipeline code, so don't assume either way without checking `omnilingual_asr/models/inference/pipeline.py` before relying on a language-conditioning mechanism.
- Used to compute CTC loss during Stage 1 training.
- Used at inference to determine which frames are "blank" vs "speech" for the CTC Compressor.
- After the Compressor runs, its job is done — output is continuous 1280-dim vectors, not text.
- **The "no vocabulary redesign needed, ~150-300 relevant chars converge" argument no longer holds as stated** — it was built on the false premise of a ~2000-entry char vocab where irrelevant *characters* zero out. With a 9812-entry SentencePiece vocab covering many scripts, the same intuition (fine-tuning concentrates gradient on the subset of *pieces* that actually appear in African-language training data) is still directionally reasonable, but "150-300" was a made-up number tied to the wrong vocab size and shouldn't be treated as a real estimate. If this matters for planning, count actual piece coverage empirically: tokenize the real African-language training corpus and see how many of the 9812 pieces ever get used.

**`Aura-1B BPE` (64k subwords — LLM output side)**
- Subword tokenizer used by Aura-1B to generate the final transcript
- Completely separate from the CTC vocabulary — no compatibility constraint
- Quality check: tokenize representative sentences from each language and verify tokens-per-word ratio. Above 3–4× for any language signals over-segmentation that will increase LLM WER

```python
# Quick tokenizer sanity check before Stage 3
test = {
    "swh": "Ninakwenda shuleni kesho asubuhi",
    "amh": "እኔ ወደ ትምህርት ቤት እሄዳለሁ",
    "yor": "Mo ń lọ sí ilé ìwé lọ́la",
    "zul": "Ngiyahamba ngiya esikoleni kusasa",
    "ful": "Mi yiɗaa jaŋde e janngo subaka",
}
for lang, text in test.items():
    tokens = tokenizer.tokenize(text)
    ratio = len(tokens) / len(text.split())
    print(f"{lang}: {ratio:.1f} tokens/word {'⚠️' if ratio > 3.5 else '✓'}")
```

**Why the two tokenizers are decoupled:** The CTC Compressor is the bridge. It consumes CTC character posteriors (omniASR_tokenizer_v1) to decide which frames to merge, then outputs averaged hidden state vectors. By the time Aura-1B sees anything, there are no tokens — only 1280-dim continuous acoustic representations. The LLM then generates subword tokens independently.

---

## 3. Training Dataset Statistics (Actual, On-Disk)

**Verified 2026-07-15**, originally from `/ocean/projects/cis250145p/shared/ASR_INDEX_V3_16k.csv`
(21 languages), **updated 2026-07-16** to include Afrikaans — now tracked in
`/ocean/projects/cis250145p/shared/ASR_INDEX_V4_16k.csv` (16kHz audio under
`/ocean/projects/cis250145p/shared/datasets_16k/` for the original 21 languages, and under
`/ocean/projects/cis250145p/tanghang/datasets_16k_afr/` for Afrikaans — see infra note below §3.1).
This is the dataset we are actually training/dev'ing against — **22 languages**, clean
`train`/`dev`/`test` split labels (no stray split values found).

| Language | Utterances | Hours | % Hours | Avg Dur (s) | Train (rows / hrs) | Dev (rows / hrs) | Test (rows / hrs) |
|---|---:|---:|---:|---:|---:|---:|---:|
| English | 2,143,326 | 5,575.0 | 27.2 | 9.4 | 2,059,358 / 5,364.4h | 42,007 / 105.2h | 41,961 / 105.4h |
| Swahili | 705,728 | 3,148.0 | 15.3 | 16.1 | 681,226 / 3,109.8h | 12,253 / 19.1h | 12,249 / 19.1h |
| Kinyarwanda | 1,035,076 | 1,467.4 | 7.1 | 5.1 | 1,002,881 / 1,418.5h | 15,984 / 24.5h | 16,211 / 24.5h |
| Arabic | 385,827 | 1,145.8 | 5.6 | 10.7 | 375,460 / 1,127.8h | 5,002 / 8.5h | 5,365 / 9.6h |
| French | 263,055 | 1,096.7 | 5.3 | 15.0 | 258,213 / 1,076.6h | 2,416 / 10.1h | 2,426 / 10.1h |
| Igbo | 594,704 | 672.9 | 3.3 | 4.1 | 592,648 / 664.3h | 998 / 3.7h | 1,058 / 4.9h |
| Hausa | 718,341 | 656.9 | 3.2 | 3.3 | 715,543 / 649.1h | 1,394 / 3.4h | 1,404 / 4.4h |
| Yoruba | 621,576 | 653.8 | 3.2 | 3.8 | 617,832 / 643.3h | 1,791 / 4.8h | 1,953 / 5.7h |
| Luganda | 175,055 | 623.8 | 3.0 | 12.8 | 147,011 / 572.6h | 14,018 / 25.1h | 14,026 / 26.1h |
| Amharic | 125,718 | 621.0 | 3.0 | 17.8 | 118,875 / 589.9h | 3,160 / 14.4h | 3,683 / 16.7h |
| Tigrinya | 126,397 | 602.1 | 2.9 | 17.1 | 116,938 / 565.0h | 4,425 / 16.8h | 5,034 / 20.3h |
| Shona | 101,120 | 574.3 | 2.8 | 20.4 | 97,726 / 554.7h | 1,683 / 9.7h | 1,711 / 9.9h |
| Xhosa | 70,189 | 478.0 | 2.3 | 24.5 | 62,783 / 427.3h | 3,768 / 25.2h | 3,638 / 25.6h |
| Sesotho | 73,961 | 477.6 | 2.3 | 23.2 | 66,321 / 427.5h | 3,922 / 25.0h | 3,718 / 25.1h |
| Zulu | 56,465 | 476.9 | 2.3 | 30.4 | 50,373 / 426.4h | 3,067 / 24.8h | 3,025 / 25.7h |
| Setswana | 94,528 | 476.4 | 2.3 | 18.1 | 84,319 / 425.6h | 4,945 / 24.9h | 5,264 / 25.8h |
| Tsonga | 74,764 | 474.3 | 2.3 | 22.8 | 67,118 / 428.0h | 3,252 / 20.6h | 4,394 / 25.7h |
| Lingala | 87,369 | 458.4 | 2.2 | 18.9 | 83,881 / 440.0h | 1,742 / 9.2h | 1,746 / 9.2h |
| Malagasy | 64,476 | 334.5 | 1.6 | 18.7 | 60,071 / 311.2h | 2,191 / 11.6h | 2,214 / 11.7h |
| Bemba | 103,400 | 205.5 | 1.0 | 7.2 | 94,773 / 188.1h | 4,480 / 8.6h | 4,147 / 8.8h |
| Portuguese | 39,230 | 168.3 | 0.8 | 15.4 | 37,533 / 161.0h | 826 / 3.6h | 871 / 3.7h |
| Afrikaans | 162,159 | 136.8 | 0.7 | 3.0 | 161,912 / 136.4h | 118 / 0.2h | 129 / 0.2h |
| **Total (22 langs)** | **7,822,464** | **20,524.5** | **100.0** | **9.4** | 7,552,795 / 19,707.7h | 133,442 / 398.9h | 136,227 / 417.9h |

**Afrikaans dev/test gap closed (2026-07-16).** Added `CommonVoice/cv-corpus-24.0-2025-12-05/af/`
(train+dev+test, 428 rows total: 181 train + 118 dev + 129 test) on top of the NCHLT/NCHLT_AUX rows,
via `build_asr_index.py --source commonvoice --language afrikaans --split {train,dev,test}`, then
resampled to 16kHz into the same `tanghang/datasets_16k_afr/` tree. Afrikaans now has a real (if very
thin) held-out signal instead of 0/0 — 118 dev / 129 test utterances is barely above a token validation
set (0.2h each), so per-language WER/BLEU on Afrikaans will be noisy; treat it as a smoke-test signal,
not a reliable eval, until more dev/test data is sourced.

**Found + fixed 2 more probing gaps during this addition:** 2 of the new CommonVoice rows
(`common_voice_af_39597042` train, `common_voice_af_40126763` dev) came back with blank
sample_rate/duration from the same kind of transient ffprobe hiccup as before — confirmed valid via
direct re-probe (32000 Hz / 7.596s and 32000 Hz / 5.004s respectively) and filled in, not removed. Also
caught and fixed a **carry-over gap in `ASR_INDEX_V4.csv` specifically** (not `_16k.csv`): the 65
nchlt_aux rows repaired in the previous pass had their `duration` fixed there already, but
`ASR_INDEX_V4.csv`'s `sample_rate` for those same 65 rows was still blank (only `_16k.csv`'s blanket
sample_rate=16000 rewrite had covered them) — fixed to `16000`, the correct native rate confirmed via
direct ffprobe on the original source files. Reverified after all fixes: 0 rows with blank/bad
sample_rate or duration across all 7,822,464 rows, 0 duplicate `audio_id`s or `path`s.

**Decision (2026-07-15): train/dev with these 21 languages as-is**, not the full 22/23 claimed by
`omniASR_tokenizer_v1` (§4). Cross-check against that table:
- **Afrikaans (`afr_Latn`)** — audio exists on disk (`NCHLT/nchlt_afr/`, `NCHLT_AUX/afr-aux{1,2}/afr/`,
  `CommonVoice/cv-corpus-24.0-2025-12-05/af/`) but is **not yet rolled into `ASR_INDEX_V3_16k.csv`** —
  indexing-only gap, not a data-availability gap.
  Portuguese and Tsonga are additionally present in the index but are not part of the tokenizer's
  claimed 22/23 language list.
- **Somali (`som_Latn`) and Wolof (`wol_Latn`)** — the tokenizer/checkpoint table in §4 lists both as
  supported, but **zero audio data exists anywhere under `/ocean/projects/cis250145p/shared/datasets_16k/`**
  for either language (confirmed via exhaustive directory search — no `somali`/`wolof`/`so`/`wo` dirs).
  Would need to be sourced from scratch (e.g. upstream FLEURS/CommonVoice do have `so`/`wo` splits that
  simply weren't downloaded here) if we ever want to cover them.

**Worth revisiting before finalizing dev/test protocol:** several low-resource languages have thin dev
sets relative to train — Igbo/Hausa/Yoruba dev sets are <2k utterances / <5h each, and Tsonga's test
split (4,394 rows) is oddly larger than its dev split (3,252 rows). Per-language WER/BLEU on these will
be noisy; consider re-splitting before locking the eval protocol.

### 3.1 Afrikaans added (2026-07-16) — now 22 languages, `ASR_INDEX_V4(.csv/_16k.csv)`

Closed the indexing-only gap noted above. Added via `build_asr_index.py --source nchlt --language
afrikaans` and `--source nchlt_aux --language afrikaans` (63,131 + 98,600 = 161,731 rows), then
resampled to 16kHz mono with `resample.py`. New files, **`ASR_INDEX_V3(.csv/_16k.csv)` left untouched**:
- `/ocean/projects/cis250145p/shared/datasets/ASR_INDEX_V4.csv` — source-path index (7,822,036 rows)
- `/ocean/projects/cis250145p/shared/ASR_INDEX_V4_16k.csv` — 16kHz index (7,822,036 rows, verified
  identical row count and Afrikaans row count to the source-path version)

**Infra note:** `/ocean/projects/cis250145p/shared/datasets_16k/` is owned by `gichamba` with mode `755`
— group members (including `tanghang`) can read but **not write** into it. The resampled Afrikaans
`.wav` files therefore live in a separate tree this user owns:
`/ocean/projects/cis250145p/tanghang/datasets_16k_afr/` (mirrors the same relative subpaths as
`NCHLT/`/`NCHLT_AUX/` under `datasets_16k`). `ASR_INDEX_V4_16k.csv`'s Afrikaans rows point there;
every other language's rows still point into the shared `datasets_16k` tree. Verified: all 161,731
resampled files exist on disk, ffprobe-spot-checked 15/15 at 16kHz mono, `sample_rate` column reads
16000 for all Afrikaans rows, 0 failed/missing during resampling. Afrikaans row is now merged directly
into the §3 table above (22 languages).

**Verified clean (2026-07-16):** 0 duplicate `audio_id`s, 0 duplicate `path`s (disk-based `sort |
uniq -d` over all 7,822,036 rows — an in-memory Python set OOM-killed on the login node, so this ran
via external sort instead), 0 invalid split labels, 0 empty transcripts/paths.

**65 zero/invalid-duration rows found, investigated, repaired (not removed).** Correction to an earlier
note in this file: these are **not** pre-existing rows from the original 21-language pool — all 65 are
newly-added Afrikaans rows (`source=nchlt_aux`, `NCHLT_AUX/afr-aux2`). Checked the actual audio directly
with `ffprobe` before deciding what to do: the files are valid (real durations 2.4s–3.5s, playable, not
corrupt) — the blank `duration` in the CSV was a transient probing gap from the original 98,600-file
`nchlt_aux` ffprobe batch (65 files didn't get probed that pass, for reasons unrelated to the audio
itself — one reproduced a transient sandbox/container-mount error on retry that resolved on the 2nd/3rd
attempt). Rather than discard 65 valid low-resource-language utterances, re-ran `ffprobe` on just these
65 files and filled in the real duration in both `ASR_INDEX_V4.csv` and `ASR_INDEX_V4_16k.csv`.
Reverified afterward: 0 rows with bad duration remain, 0 rows with bad sample_rate remain, row count
unchanged at 7,822,036.

**Found + fixed a real bug inherited from `ASR_INDEX_V3_16k.csv`:** 4,143,793 of 7,822,036 rows (all in
the pre-existing 21-language portion) had a **stale `sample_rate` column** — e.g. `48000`/`32000`/`44100`
left over from the pre-resample source metadata, never updated after the audio was actually resampled to
16kHz. Confirmed via `ffprobe` spot-checks (12/12 files genuinely 16kHz despite the CSV claiming
otherwise) and via reading `st/data/dataset.py:196-202`, which **trusts the CSV's `sample_rate` to skip
re-reading the real file header** — meaning if this index were ever wired into a training config as-is,
`AF.resample(waveform, orig_freq=<stale value>, new_freq=16000)` would run on audio that's already at
16000Hz, corrupting the majority of the dataset at load time. No training config currently points at
this file (all 8 configs use a separate `ASR_INDEX.csv`, 4 languages only — out of scope here per
instruction, to be addressed separately), so nothing was actually corrupted. Fixed by rewriting
`sample_rate` to `16000` for all 4,143,793 affected rows; reverified afterward — 0 rows now disagree.

**⚠️ Blocker before Afrikaans can be used for training-with-eval: zero dev/test data.** `nchlt` and
`nchlt_aux` are XML-manifest sources that hardcode `split="train"` for every row
(`_read_nchlt_xml()` in `build_asr_index.py` — the `--split` CLI flag has no effect on these two
sources; `dataset_files()` doesn't even branch on it for them). All 161,731 Afrikaans rows are `train`.
Before training Afrikaans with a real validation signal, do one of:
1. Carve a held-out dev/test split from the existing NCHLT/NCHLT_AUX pool (e.g. hold out speakers by
   `speaker_id`, matching how other languages get speaker-disjoint splits), or
2. Add `CommonVoice/cv-corpus-24.0-2025-12-05/af/` as a second Afrikaans source — CommonVoice already
   has native train/dev/test TSVs and `build_asr_index.py`'s `commonvoice` source already maps
   `"afrikaans": "af"` — this only requires an `--update-index ... --source commonvoice --language
   afrikaans` pass per split, no new code.
Until one of these is done, Afrikaans can be included in training but has no way to measure its own
WER/BLEU during validation.

---

## 4. Language Coverage

**Caveat (2026-07-10):** the table below frames coverage as "confirmed... `omniASR_tokenizer_v1`" but
we verified the tokenizer itself contains no per-language ID tokens (see §2.3) — so language coverage
here is inherited from OmniASR's published training-data claims, not something we independently
confirmed against the tokenizer or checkpoint. Treat the ISO/script list below as "OmniASR claims to
support these," not "we verified support for these," until the actual inference pipeline is run.

### Confirmed (22/23 languages, `omniASR_tokenizer_v1`)
| # | ISO | Language | Script |
|---|-----|----------|--------|
| 1 | afr | Afrikaans | afr_Latn |
| 2 | amh | Amharic | amh_Ethi |
| 3 | bem | Bemba | bem_Latn |
| 4 | ful | Fulah | ful_Latn |
| 5 | hau | Hausa | hau_Latn |
| 6 | ibo | Igbo | ibo_Latn |
| 7 | lin | Lingala | lin_Latn |
| 8 | lug | Luganda | lug_Latn |
| 9 | luo | Luo | luo_Latn |
| 10 | nso | Sepedi | nso_Latn |
| 11 | nya | Chichewa | nya_Latn |
| 12 | orm | Oromo | orm_Latn |
| 13 | sna | Shona | sna_Latn |
| 14 | som | Somali | som_Latn |
| 15 | swh | Swahili | swh_Latn |
| 16 | tir | Tigrinya | tir_Ethi |
| 17 | tsn | Setswana | tsn_Latn |
| 18 | umb | Umbundu | umb_Latn |
| 19 | wol | Wolof | wol_Latn |
| 20 | xho | Xhosa | xho_Latn |
| 21 | yor | Yoruba | yor_Latn |
| 22 | zul | Zulu | zul_Latn |

### Gap: sot_Latn (Sesotho / Southern Sotho)
`sot_Latn` is **absent** from `omniASR_tokenizer_v1`. Handling strategy:
1. Add `sot_Latn` as a new language ID token in the tokenizer
2. Initialize its language embedding via weighted interpolation of Bantu S-group neighbors:
   ```python
   emb["sot_Latn"] = 0.5 * emb["nso_Latn"] + 0.3 * emb["tsn_Latn"] + 0.2 * emb["zul_Latn"]
   ```
   (Sepedi is the closest relative; Setswana and Zulu share Bantu S-group phonology)
3. Fine-tune specifically on NCHLT Sesotho data during Stage 1 (it has supervised pairs)
4. The CTC character vocabulary already covers Latin script → no tokenizer surgery needed for output

---

## 5. Training Stages

### Stage 0: Setup & Architecture Prep

**Tasks:**
- Clone `omnilingual-asr` and install `fairseq2`
- Prepare dataset in `MIXTURE_PARQUET` or `MANIFEST` format with `lang` column using `{iso639-3}_{Script}` IDs
- Build `language_distribution.tsv` for the mixture sampling

**Dataset manifest columns required:**
```
audio_path | duration_ms | transcription | lang
```

**CTC vocabulary:** Use `omniASR_tokenizer_v1` character-level vocab (≈2000 tokens). For Sesotho: all characters are already covered by Latin script entries. Only the language ID token needs adding.

**Mixture sampling:** Use `beta_corpus=0.5, beta_language=0.5` (OmniASR default) for balanced temperature-based sampling across language sizes. Tune `beta_language` upward (→0.7) if low-resource languages underfit.

---

### Stage 1: African CTC Fine-tuning of the CTC Encoder

Starting from `omniASR_CTC_1B` (already ASR-fine-tuned on 1239 languages) using the `ctc-finetune` recipe. This continues training the existing CTC head rather than initializing one from scratch.

**Config (based on `ctc-finetune.yaml` + recommendation settings):**
```yaml
model:
  name: "omniASR_CTC_1B"            # ← start from CTC checkpoint, not W2V SSL

dataset:
  storage_mode: "MIXTURE_PARQUET"
  task_mode: "ASR"
  mixture_parquet_storage_config:
    dataset_summary_path: "/path/to/african_asr/language_distribution.tsv"
    beta_corpus: 0.5
    beta_language: 0.5
  asr_task_config:
    max_audio_len: 960_000           # 60s at 16kHz
    max_num_elements: 7_680_000      # up to 8 × 60s per batch

tokenizer:
  name: "omniASR_tokenizer_v1"

optimizer:
  config:
    lr: 1e-05                        # lower LR since we're fine-tuning, not training from scratch

trainer:
  freeze_encoder_for_n_steps: 0     # full fine-tune from step 0
  mixed_precision:
    dtype: "torch.bfloat16"
  grad_accumulation:
    num_batches: 8                   # adjust for GPU count (64 GPUs → num_batches: 1)

regime:
  num_steps: 20_000                  # fewer steps needed vs. from-scratch (CTC head already warm)
  validate_every_n_steps: 500
  validate_after_n_steps: 500
  checkpoint_every_n_steps: 500
  checkpoint_after_n_steps: 500
```

**Run command:**
```bash
cd omnilingual_asr
export OUTPUT_DIR="/path/to/checkpoints/aura_v2_ctc_stage1"
python -m workflows.recipes.wav2vec2.asr $OUTPUT_DIR \
  --config-file workflows/recipes/wav2vec2/asr/configs/ctc-finetune.yaml \
  --config model.name=omniASR_CTC_1B
```

**Why fewer steps (20k vs 30k):** The encoder is already ASR-discriminative and the CTC head already knows our 22 African language phoneme distributions. We're specializing, not learning ASR from scratch.

**Training data (all available):**
| Dataset | Languages | Notes |
|---------|-----------|-------|
| FLEURS (train+dev) | 15 of our 23 | Use both splits for training |
| WAXAL | 6 | Primary low-resource signal |
| NCHLT | 4 (afr, nso, tsn, sot) | Sesotho coverage here |
| BIG-C | 1 (yor) | Conversational Yoruba |
| BembaSpeech | 1 (bem) | |
| + any external unlabeled → pseudo-label | all 23 | Optional: use omniASR_CTC_1B itself before fine-tune |

**LAIL Integration (optional during Stage 1):**

Language-Aware Intermediate Loss aligns encoder intermediate states with the target LLM. Add at Transformer layer 24 (midpoint of 48):

```python
# During Stage 1 CTC training, after layer 24:
# 1. Extract h_24 ∈ R^{B×T×1280}
# 2. Run h_24 through a lightweight alignment MLP: align_proj(h_24) → 1280-dim
# 3. Freeze Aura-1B encoder states as targets
# 4. LAIL loss = MSE(align_proj(h_24), aura1b_encoder_states)
# 5. λ annealed: 0.0 → 0.3 linearly over steps 0-10k, held at 0.3

total_loss = L_ctc_final + 0.3 * L_ctc_intermediate_layer24 + λ(step) * L_lail
```

**Expected outcome:** A standalone `omniASR_CTC_1B` specialized on 23 African languages that beats or matches Whisper-large-v3 on all FLEURS/NCHLT benchmarks as a CTC-only system. This is the "excellent encoder out of the box" target. Because the CTC head is warm-started, convergence is faster and the model should reach low WER well before 20k steps — checkpoint the model every 500 steps and select by validation WER.

**Update (2026-07-20) — v1 run complete, real problems found, v2 launched:**

v1 (`stage1_omniasr_ctc_h100_weighted.yaml`, 20k steps, single H100) completed with
`val/wer=0.3265`, but two real issues surfaced on review, both fixed for v2:

- **Punctuation corruption.** The omniASR CTC vocab has zero tokens for `, . ; : ! ?`
  — encoding any of them maps to `<unk>` (confirmed against the tokenizer directly:
  `sp.encode(',')` → id 3 → decodes back as literal `" ⁇ "`). FLEURS/Waxal references
  are 60-99% punctuated, so this was corrupting both the CTC training targets and the
  WER references for ~18/22 languages. Retroactively stripping `⁇` from the existing
  val_preds closed part of the gap (overall WER 32.65%→29.98% at step 20000) but is
  much bigger for punctuation-heavy Bantu-orthography languages (zulu 20.7%→14.4%,
  afrikaans 32.4%→24.3%) than for languages like Yoruba/Igbo/Bemba, which stayed high
  — those are genuinely hard, not an artifact. Fix: `_strip_ctc_unsupported_punct()`
  added to `collator.py`'s `CTCRawAudioCollator.__call__`, applied before
  `sp_tokenizer.encode()` for both train and dev (must be symmetric — stripping only
  one side just relocates the corruption). Scoped to the CTC path specifically, not
  `dataset.py` — Stage 2-4's `AuraCollator`/`RawAudioCollator` tokenize the same raw
  `text` with Aura-1B's own tokenizer, which has full punctuation support, so stripping
  it at the dataset level would have needlessly denied Stages 2-4 real punctuation.
- **Batch settings were far too conservative.** v1's header called them a "reasoned,
  unvalidated estimate" for the first-ever H100 test. Real W&B system-metrics logs
  (`gpu.0.memoryAllocated`) show only ~30GB/80GB used, and GPU compute utilization
  averaged 69.6% (dipping to 0% repeatedly) — almost certainly from `eval_every: 250`
  triggering 80 synchronous CPU-bound validation+checkpoint stalls over the run.

**v2** (`stage1_v2.yaml`): fresh run, no `--resume_from`, 2x H100-80 DDP (`torchrun
--standalone --nproc_per_node=2`), `max_batch_duration=65/max_batch_size=10/grad_accum=1`
(real headroom from the v1 profiling data), `eval_every=1000/save_every=2000` (was
250/250), `max_steps=50000` with a 2-cycle `cosine_warmup_restarts` schedule
(`first_cycle_steps=25000, gamma=1.0` — anneals `1e-5`→`1e-7` then back to full `1e-5`
once more at step 25000, not one long single-cycle decay like v1's `first_cycle_steps=20000`).
Note: `cycle_mult` was silently dropped by `pretrain_omniasr_ctc.py`'s scheduler call
before this — added so the config key is actually live.

---

### Stage 2: CTC Compressor + Projector Training

**Encoder:** Frozen (best Stage 1 checkpoint by validation WER)

**Trained modules:**
- CTC Compressor: W^T H matmul (already differentiable, no change needed)
- Projector: 2-layer Transformer at 1280-dim (reinitialize from scratch or keep v1 weights with linear 1024→1280 adapter discarded)

**Loss:** Cross-entropy from Aura-1B (frozen) with teacher-forcing on ground-truth transcripts

```python
# Forward pass:
# audio → frozen_encoder → 1280-dim @ 50Hz
# → ctc_head → CTC logits (auxiliary CTC loss, weight 0.1)
# → ctc_compressor(encoder_out, ctc_logits) → compressed_seq [variable length]
# → projector(compressed_seq) → 1280-dim projected
# → frozen_aura1b(projected + text_prompt) → CE loss

loss = 0.9 * ce_loss + 0.1 * ctc_loss
```

**Config:**
- lr: 5e-4 (projector learning from scratch)
- Steps: 5,000
- Batch: max 32 samples
- No weight decay on projector LayerNorm

**Expected outcome:** Projector learns to map CTC-compressed encoder representations into Aura-1B's embedding space. Validation: attach frozen encoder + projector to Aura-1B and run greedy decode on FLEURS dev.

**Update (2026-07-20):** Built as `configs/experiment/stage2/stage2_v1.yaml`,
pointed at the real Stage 1 checkpoint (`encoder_step20000.pt`, val/wer=0.3265). Two deviations
from the plan above, both deliberate:
- **Live, not cached.** The original plan implied precomputing encoder features; caching the full
  22-language index would need ~4TB (extrapolated from the old 4-language cache's real
  ~55.5KB/sec-of-audio rate) against only ~4.4TB free on the shared filesystem project-wide. The
  environment reason for caching (fairseq2 needing an isolated torch==2.8.0+cu128 env, incompatible
  with the main torch==2.6.0+cu124 stack) no longer holds either — `omniasr_extract` already has
  everything `train_st.py` needs and has been confirmed to import/run every training entrypoint
  cleanly. So the encoder runs live instead (frozen, no optimizer state, no retained backward graph).
- **lr: 5e-5, not 5e-4.** The 5e-4 above was never validated — no Stage 2 run has ever actually used
  it. `stage2.yaml` (the working Conformer version) and this project's own historical cached-features
  validation run (`configs/experiment/stage2/smoke/stage2_omniasr_cached_smoke.yaml` — 4 languages,
  igbo/yoruba/hausa/bemba, 500 steps, features precomputed offline via
  `scripts/extract_omniasr_features.py` into `cache_dir` and loaded via
  `CachedFeatureDataset`/`CachedFeatureCollator`; superseded by the live config above) both use 5e-5;
  went with that real precedent instead.
- **ctc_weight: 0.0, not the 0.1 shown in the loss formula above.** `build_model()`'s `omniasr_live`
  path requires this — omniASR's own CTC head outputs over its 9812-piece SentencePiece vocab, not
  compatible as an auxiliary loss target the way the Conformer path's char-level CTC vocab is.

Getting the checkpoint to load at all required a real fix to `OmniASREncoder._build_and_load()`,
which only recognized the base pretrained checkpoint's `"model"` key — our own `save_checkpoint()`
format uses `"model_state_dict"`, and its keys carry a `_model.` prefix plus two exact-duplicate
`ctc_head.*` keys (`ctc_head` is just `self._model.final_proj` re-registered under a second
attribute name) that needed stripping/dropping before the raw fairseq2 model's `strict=True` load
would accept it. Batch settings in the new config are a reasoned-not-validated guess — no smoke test
has been run yet.

---

### Stage 3: Aura-1B LoRA Fine-tuning (Encoder Frozen)

**Frozen:** Entire encoder (Stage 1 checkpoint) + CTC Compressor

**Trainable:** Projector (full) + Aura-1B via LoRA

**LoRA config:**
```python
# Target: attention projections + FFN gate projections
lora_config = LoRAConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none"
)
# ~42M trainable params on top of frozen 1013M Aura-1B
```

**Prompt format** (matching Aura-ASR v1):
```
<lang_tag> transcribe speech: [CTC-compressed encoder tokens] → [transcription] <eos>
```

**Training:**
- lr: 1e-4 (LoRA), 5e-5 (projector)
- Steps: 10,000
- Warmup: 500 steps
- LR schedule: cosine decay
- Data: same African ASR datasets as Stage 1

**Expected outcome:** Full Aura-ASR v2 system. The LLM drives WER well below the CTC standalone, leveraging African text priors in Aura-1B's 64k BPE vocabulary.

**Update (2026-07-20):** Built as `configs/experiment/stage3/stage3_v1.yaml` —
`r=32/alpha=64` targeting `q_proj/v_proj` only (not the roadmap's full attention+FFN target list;
matches the real, working Conformer `stage3.yaml` precedent, not an untested plan), single
`lr: 5e-5` for both projector and LoRA (`train_st.py` has no per-param-group LR mechanism), and
`ctc_weight: 0.0` (required by `build_model()`'s `omniasr_live` guard — no auxiliary CTC loss is
possible against omniASR's SentencePiece vocab). Encoder and batch settings otherwise match
`stage2_v1.yaml`, since the memory profile is the same (encoder frozen either way;
LoRA adapters are negligible extra trainable params next to a 2-layer projector). Not runnable yet:
`projector_checkpoint` needs Stage 2's real output, which doesn't exist until that config actually
runs. Note: the existing Conformer `stage3.yaml` has a real bug — its header says "Freeze encoder"
but sets `unfreeze_encoder: true` — left as-is (pre-existing legacy code, not touched this pass).

---

### Stage 4 (Optional): End-to-End Polish

Unfreeze encoder with a very low LR for final WER squeeze:

| Module | LR |
|--------|----|
| Encoder (W2V_1B) | 5e-7 |
| CTC head | 1e-6 |
| CTC Compressor | 1e-5 |
| Projector | 1e-5 |
| Aura-1B LoRA | 5e-5 |

- Steps: 3,000–5,000 (monitor val WER closely, stop early)
- Use gradient clipping: max_norm=1.0
- Risk: encoder can drift from SSL representations; monitor CTC standalone WER as sanity check

**Update (2026-07-20):** Built as `configs/experiment/stage4_omniasr_live.yaml` /
`stage4_omniasr_live_h100.yaml` — encoder fully unfrozen (not the tiny per-module LRs above; single
`lr: 1.0e-5` for the whole encoder, much lower than a from-scratch projector LR since a 975M-param
pretrained encoder is easy to destroy at higher LR), LLM stays frozen (LoRA/Stage 3 not layered in
yet), 4-language scope (not v1's 22), unvalidated. Real OOM history on V100 (job 42197931): the
naive `max_batch_duration=40.0/grad_accum=16` OOM'd almost every batch past step 3 — a fully
unfrozen 975M-param encoder pays ~8GB of AdamW optimizer state that a frozen encoder (Stage 2) never
does — cut to `10.0/grad_accum=64` to compensate. `freeze_ctc_head: true` despite the encoder being
otherwise unfrozen: it gets no gradient anyway (`ctc_weight=0.0` + `ctc_compress.strategy=avg`), so
freezing it just avoids AdamW weight-decaying it for no benefit. Run
`scripts/smoke_test_omniasr_live.py` before scaling either config up.

---

## 6. Reference Baselines (Pre-Fine-Tuning)

These are the numbers for `omniASR_CTC_1B` **as-is**, before any African fine-tuning. They define the starting point for Aura-ASR v2 Stage 1.

### 6.1 WAXAL — Ethiopian Subset (Ethio-ASR paper, arXiv:2603.23654)

WER (%) on WAXAL test split, Ethiopian languages only. Note: OmniASR models were trained on a pre-released WAXAL subset (July 2025) — possible train/test overlap.

| Model | Amh | Orm | Tir | Sid | Wol | **Avg** |
|-------|-----|-----|-----|-----|-----|---------|
| omniASR-CTC-300M | 49.15 | 58.11 | 40.77 | 52.90 | 41.08 | 48.40 |
| **omniASR-CTC-1B** | **37.44** | **50.15** | **31.34** | **46.35** | **37.26** | **40.51** |
| omniASR-CTC-3B | 32.41 | 45.91 | 27.91 | 43.44 | 35.38 | 37.01 |
| omniASR-CTC-7B | 32.48 | 46.21 | 27.79 | 44.58 | 35.21 | 37.26 |
| omniASR-LLM-300M | 30.95 | 46.10 | 27.33 | 41.43 | 34.10 | 35.98 |
| omniASR-LLM-1B | 27.65 | 42.87 | 25.28 | 40.37 | 33.21 | 33.88 |
| omniASR-LLM-3B | 26.83 | 42.32 | 24.80 | 40.36 | 32.91 | 33.48 |
| omniASR-LLM-7B | 25.12 | 40.69 | 23.59 | 39.22 | 32.46 | **32.21** |
| Ethio-ASR (w2v-bert-2.0, 600M) | — | — | — | — | — | **30.48** ✓ |

Ethio-ASR (600M CTC-only) beats all OmniASR variants including LLM-7B on this subset, confirming that targeted fine-tuning on in-domain data is more efficient than scale alone.

### 6.2 FLEURS — Amharic & Oromo (Ethio-ASR paper, zero-shot for Ethio-ASR, trained for OmniASR)

| Model | Amh | Orm |
|-------|-----|-----|
| omniASR-CTC-300M | 34.41 | 77.53 |
| **omniASR-CTC-1B** | **48.23** | **68.38** |
| omniASR-CTC-3B | 20.59 | 62.76 |
| omniASR-CTC-7B | 16.19 | 61.96 |
| omniASR-LLM-300M | 18.84 | 61.48 |
| omniASR-LLM-1B | 19.97 | 58.11 |
| omniASR-LLM-3B | 13.84 | 56.98 |
| omniASR-LLM-7B | **12.77** | **50.08** |
| Ethio-ASR (w2v-bert-2.0) | 19.17 | — |

⚠️ CTC-1B is anomalously worse than CTC-300M on Amharic FLEURS (48.23 vs 34.41) — likely a normalization artifact. The paper also documents serious FLEURS data quality issues for Ethiopian languages. Use as indicative, not conclusive.

### 6.3 AfriVox-v2 — Conversational In-the-Wild, 14 Languages (arXiv:2605.03590)

Average WER across 14 African languages (conversational / unscripted speech):

| Model | Avg WER |
|-------|---------|
| omniASR-CTC-300M | 39.20 |
| **omniASR-CTC-1B** | **33.91** |
| omniASR-CTC-7B | 32.20 |
| Gemini-3-Flash | 32.13 |
| Sahara-v2 (closed-source) | **23.78** |

Per-language breakdown for **omniASR-CTC-1B** on AfriVox-v2:

| swa | tsn | xho | zul | sna | hau | yor | amh | ibo | lug | sot | ful |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| 9.73 | 21.00 | 25.57 | 25.64 | 32.23 | 36.63 | 36.48 | 37.38 | 39.51 | 42.82 | 47.30 | 56.86 |

Hardest languages: Fulah (56.86%), Sesotho (47.30%), Luganda (42.82%). These are the primary targets for improvement from African fine-tuning and LLM integration.

### 6.4 Interpretation for Aura-ASR v2

| Benchmark | omniASR-CTC-1B baseline | Target after Stage 3 | Gap to close |
|-----------|------------------------|----------------------|--------------|
| WAXAL Ethiopian avg | 40.51% | <30% | ~10 pts |
| FLEURS Amh/Orm avg | ~58% | <20% | ~38 pts |
| AfriVox-v2 avg | 33.91% | <25% | ~9 pts |
| Sahara-v2 (AfriVox-v2) | 23.78% | match/beat | hard without conversational data |

The FLEURS gap is large because FLEURS data is in OmniASR's training mix — the zero-shot anomaly on CTC-1B suggests the base model needs significant specialization. African fine-tuning should dramatically close this.

---

## 7. Expected Benchmark Impact

| Model | WAXAL Eth. (avg) | FLEURS SSA (avg) | NCHLT (avg) | AfriVox-v2 |
|-------|-----------------|-------------------|-------------|------------|
| Aura-ASR v1 (current) | — | — | — | not evaluated |
| omniASR-CTC-1B (baseline) | 40.51% | ~58% (amh/orm) | — | 33.91% |
| omniASR-LLM-7B | 32.21% | ~31% | — | 32.20% |
| Sahara-v2 (closed) | — | — | — | 23.78% |
| **Aura-ASR v2 Stage 1 (CTC only)** | ~32–35% | ~20–30% | TBD | ~28–32% |
| **Aura-ASR v2 Stage 3 (+ Aura-1B)** | ~25–30% | ~15–22% | TBD | ~24–28% |

**Key advantage over OmniASR-LLM-7B:** Same encoder backbone, 7× fewer LLM parameters, but Aura-1B has stronger African text priors. Domain-specialized LLM expected to outperform generic LLaMA-7B on our specific language set.

**Key constraint:** Sahara-v2 on conversational speech (23.78%) requires matching their training data volume (~50k hrs conversational). With read-speech-heavy training data, our target on AfriVox-v2 is ~25–28% — competitive but likely not a full beat without additional conversational data.

---

## 8. Implementation Checklist

### Data
- [ ] Convert all ASR datasets to `MIXTURE_PARQUET` or manifest format
- [ ] Add `lang` column with `{iso639-3}_{Script}` IDs
- [ ] Build `language_distribution.tsv` for mixture sampling
- [ ] Add `sot_Latn` token to tokenizer vocabulary
- [ ] Initialize `sot_Latn` embedding from Bantu neighbors

### Stage 1
- [ ] Install fairseq2 + omnilingual-asr
- [ ] Verify `omniASR_CTC_1B` loads correctly (check param count ≈ 975M)
- [ ] Evaluate `omniASR_CTC_1B` zero-shot on all 27 benchmark rows (baseline before African fine-tune)
- [ ] Run 500-step smoke test (check CTC loss decreases from baseline)
- [ ] Train 20k steps (monitor val WER, may converge earlier)
- [ ] Evaluate African fine-tuned CTC WER on all 27 benchmark rows
- [ ] Select best checkpoint by avg validation WER

### Stage 2
- [ ] Adapt CTC Compressor input dim: 1024 → 1280
- [ ] Reinitialize projector: 2-layer Transformer at 1280→1280
- [ ] Verify Aura-1B loads and forward pass runs
- [ ] Train 5k steps
- [ ] Evaluate full pipeline (encoder+compressor+projector+frozen LLM)

### Stage 3
- [ ] Apply LoRA to Aura-1B
- [ ] Train 10k steps
- [ ] Evaluate all 27 benchmark rows
- [ ] Compare against v1 results

### Stage 4 (if needed)
- [ ] Unfreeze encoder with conservative LR
- [ ] Monitor both CTC WER and LLM WER
- [ ] Stop if CTC WER degrades >2% relative

---

## 9. Key Differences from OmniASR-LLM

For reference, the `wav2vec2_llama/model.py` architecture uses:

| Component | OmniASR-LLM | Aura-ASR v2 |
|-----------|-------------|-------------|
| Temporal compression | Frame stacking (`encoder_stacking=N`) | CTC Compressor (variable-length) |
| Projection | Linear: `(dim×N) → llama_dim` | 2-layer Transformer: `1280 → 1280` |
| LLM | LLaMA (general) | Aura-1B (African text fine-tuned) |
| Lang conditioning | `lang_embeddings` + `lid_marker` tokens | `<lang_tag>` prefix token |
| Zero-shot | Yes (context examples) | No (supervised fine-tune) |
| Streaming | Yes (segmented audio) | No |

The CTC Compressor + Transformer Projector approach is more compute-intensive at inference than simple stacking, but produces better representations for the LLM — especially for languages where phoneme boundaries don't align to uniform time windows.

---

## 10. Compute Requirements

| Stage | GPUs (A100 80GB) | Time (est.) |
|-------|------------------|-------------|
| Stage 1 (20k steps) | 8–64× | 16–48h |
| Stage 2 (5k steps) | 4–8× | 4–8h |
| Stage 3 (10k steps) | 4–8× | 8–16h |
| Stage 4 (3k steps) | 4–8× | 4–6h |

Minimum viable: 8× A100 with `grad_accumulation.num_batches: 8` for Stage 1.

For constrained compute: start with `omniASR_CTC_300M` to validate the full pipeline end-to-end, then swap in the 1B checkpoint.