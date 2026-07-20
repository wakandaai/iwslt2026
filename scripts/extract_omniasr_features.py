"""
Offline feature extraction for omniASR_CTC_1B.

Runs the REAL Meta model (via fairseq2/omnilingual-asr) to precompute encoder
hidden_states + argmax CTC predicted_ids for audio in an ASR/AST index CSV,
caching them to disk for frozen-encoder training (see CachedFeatureDataset /
CachedFeatureCollator / SpeechAura.forward_cached).

STANDALONE SCRIPT — run only with the isolated conda env's python
(/ocean/projects/cis250145p/tanghang/iwslt2026/.envs/omniasr_extract), never
imported by src/st/*. That env has torch==2.8.0+cu128, incompatible with the
main training env's pinned torch==2.6.0+cu124.

Usage (from the repo root):
    PYTHONPATH=src <omniasr_extract>/bin/python scripts/extract_omniasr_features.py \\
        --index /ocean/projects/cis250145p/shared/ASR_INDEX_V4_16k.csv \\
        --checkpoint /ocean/projects/cis250145p/tanghang/iwslt2026/checkpoints/omniasr_ctc_1b/omniASR-CTC-1B.pt \\
        --cache-dir cache/omniasr_ctc_1b \\
        --split train --languages bemba --max-hours 5

Why the manual 3-step forward instead of model(...): Wav2Vec2AsrModel.forward()
only returns (logits, layout) when called with targets=None — it does not
expose the pre-CTC-head hidden states the projector needs. So we call the
three public submodules (encoder_frontend, encoder, final_proj) directly.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import soundfile as sf
import torchaudio.functional as AF

log = logging.getLogger(__name__)

EXTRACTOR_VERSION = "omniasr-1b-v1"
TARGET_SAMPLE_RATE = 16000
CTC_VOCAB_SIZE = 9812


# ============================================================================
# Model construction — hardcoded "1b" config, no fairseq2 asset/hub download
# ============================================================================

def build_model(checkpoint_path: str, device: torch.device):
    from fairseq2.models.transformer import TransformerNormOrder
    from fairseq2.models.wav2vec2 import Wav2Vec2EncoderConfig
    from fairseq2.models.wav2vec2.asr.config import Wav2Vec2AsrConfig
    from fairseq2.models.wav2vec2.asr.factory import Wav2Vec2AsrFactory

    encoder_config = Wav2Vec2EncoderConfig(
        model_dim=1280,
        feature_dim=512,
        num_encoder_layers=48,
        num_encoder_attn_heads=16,
        ffn_inner_dim=5120,
        feature_extractor_layer_descs=[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2,
        feature_extractor_bias=True,
        feature_extractor_layer_norm_convs=True,
        layer_norm_features=False,
        pos_encoder_type="conv",
        pos_conv_kernel_size=128,
        num_pos_conv_groups=16,
        norm_order=TransformerNormOrder.PRE,
        dropout_p=0.0,
        attn_dropout_p=0.0,
        ffn_inner_dropout_p=0.0,
        layer_drop_p=0.0,
    )
    config = Wav2Vec2AsrConfig(
        encoder_config=encoder_config,
        target_vocab_size=CTC_VOCAB_SIZE,
        use_masking=False,
        final_dropout_p=0.0,
    )

    model = Wav2Vec2AsrFactory(config).create_model()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # strict=True on purpose: a load failure is the loudest, cheapest signal
    # that one of the hardcoded config fields above is wrong.
    model.load_state_dict(state, strict=True)

    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    log.info(f"Loaded omniASR_CTC_1B ← {checkpoint_path} (strict=True)")
    return model


# ============================================================================
# Per-utterance preprocessing + manual 3-step forward
# ============================================================================

def load_and_preprocess(audio_path: str, entry_sample_rate: str | None) -> torch.Tensor:
    """Load audio, average to mono, resample to 16kHz, per-utterance
    zero-mean/unit-variance normalize — mirrors omnilingual_asr's real
    inference preprocessing (average channels, not channel-0-only like
    SpeechDataset; apply_audio_normalization = layer_norm(waveform, shape))."""
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # average channels to mono (matches Meta's real preprocessing)
    waveform = torch.from_numpy(data)

    if entry_sample_rate:
        sr = int(float(entry_sample_rate))
    if sr != TARGET_SAMPLE_RATE:
        waveform = AF.resample(waveform, sr, TARGET_SAMPLE_RATE)

    waveform = F.layer_norm(waveform, waveform.shape)
    return waveform


@torch.inference_mode()
def extract_batch(model, waveforms: list[torch.Tensor], device: torch.device):
    """Manual 3-step forward for a batch of same-or-different-length waveforms.

    Returns:
        hidden_states: (B, T, 1280) bf16, zero-padded
        predicted_ids: (B, T) long, zero-padded
        lengths:       (B,) valid frame counts per utterance (post-frontend)
    """
    from fairseq2.nn import BatchLayout

    B = len(waveforms)
    max_len = max(w.size(0) for w in waveforms)
    seqs = torch.zeros(B, max_len, device=device, dtype=torch.bfloat16)
    for i, w in enumerate(waveforms):
        seqs[i, : w.size(0)] = w.to(device=device, dtype=torch.bfloat16)
    seq_lens = [w.size(0) for w in waveforms]

    layout = BatchLayout.of(seqs, seq_lens=seq_lens)

    seqs, layout, _ = model.encoder_frontend.extract_features(seqs, layout)
    seqs, _ = model.encoder_frontend.process_features(seqs, layout, masker=None)
    hidden_states = model.encoder(seqs, layout)          # (B, T, 1280)
    logits = model.final_proj(hidden_states)              # (B, T, 9812)
    predicted_ids = logits.argmax(dim=-1)                  # (B, T)

    lengths = torch.as_tensor(layout.seq_lens, dtype=torch.long) if layout.padded \
        else torch.full((B,), hidden_states.size(1), dtype=torch.long)

    return hidden_states, predicted_ids, lengths


# ============================================================================
# Cache I/O
# ============================================================================

def cache_path(cache_dir: Path, audio_id: str) -> Path:
    return cache_dir / audio_id[:2] / f"{audio_id}.pt"


def is_cached(cache_dir: Path, audio_id: str) -> bool:
    path = cache_path(cache_dir, audio_id)
    if not path.exists():
        return False
    try:
        meta = torch.load(path, map_location="cpu", weights_only=True)
        return meta.get("extractor_version") == EXTRACTOR_VERSION
    except Exception:
        return False


def save_cache(cache_dir: Path, audio_id: str, hidden_states: torch.Tensor,
               predicted_ids: torch.Tensor, length: int) -> None:
    path = cache_path(cache_dir, audio_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".pt.tmp")

    torch.save({
        "hidden_states": hidden_states[:length].to(torch.float16).cpu(),
        "predicted_ids": predicted_ids[:length].to(torch.int16).cpu(),
        "length": int(length),
        "extractor_version": EXTRACTOR_VERSION,
    }, tmp_path)
    os.replace(tmp_path, path)  # atomic — a killed job must not leave a corrupt cache file


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract omniASR_CTC_1B features")
    parser.add_argument("--index",      required=True, help="ASR/AST index CSV path")
    parser.add_argument("--checkpoint", required=True, help="Path to omniASR-CTC-1B.pt")
    parser.add_argument("--cache-dir",  required=True, help="Output cache directory")
    parser.add_argument("--split",      default="train")
    parser.add_argument("--languages",  nargs="*", default=None)
    parser.add_argument("--sources",    nargs="*", default=None)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--min-duration", type=float, default=0.1)
    parser.add_argument("--max-hours",  type=float, default=None,
                         help="Cap cumulative audio duration extracted this run (validation-subset runs)")
    parser.add_argument("--limit",      type=int, default=None, help="Cap number of utterances (smoke tests)")
    parser.add_argument("--shard-id",   type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    from st.data.dataset import load_index_csv

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)

    model = build_model(args.checkpoint, device)

    entries = load_index_csv(
        args.index, args.split, args.languages, args.sources,
        args.min_duration, args.max_duration,
    )
    entries = entries[args.shard_id :: args.num_shards]
    log.info(f"{len(entries)} entries after shard {args.shard_id}/{args.num_shards} filtering")

    n_done, n_skipped, n_failed = 0, 0, 0
    cumulative_hours = 0.0
    t_start = time.time()

    for entry in entries:
        if args.limit is not None and n_done >= args.limit:
            break
        if args.max_hours is not None and cumulative_hours >= args.max_hours:
            log.info(f"Reached --max-hours={args.max_hours} cap, stopping.")
            break

        audio_id = entry.get("audio_id", "")
        audio_path = entry.get("path") or entry.get("audio_path") or ""
        dur = float(entry.get("duration", 0.0) or 0.0)

        if is_cached(cache_dir, audio_id):
            n_skipped += 1
            cumulative_hours += dur / 3600
            continue

        try:
            waveform = load_and_preprocess(audio_path, entry.get("sample_rate"))
            hidden_states, predicted_ids, lengths = extract_batch(model, [waveform], device)
            save_cache(cache_dir, audio_id, hidden_states[0], predicted_ids[0], int(lengths[0].item()))
            n_done += 1
            cumulative_hours += dur / 3600
        except Exception as exc:
            n_failed += 1
            log.warning(f"Failed to extract {audio_id} ({audio_path}): {exc}")

        if (n_done + n_skipped) % 50 == 0:
            elapsed = time.time() - t_start
            log.info(
                f"done={n_done} skipped={n_skipped} failed={n_failed} "
                f"cum_hours={cumulative_hours:.2f} elapsed={elapsed:.0f}s"
            )

    log.info(
        f"Finished: done={n_done} skipped={n_skipped} failed={n_failed} "
        f"cum_hours={cumulative_hours:.2f} → {cache_dir}"
    )


if __name__ == "__main__":
    main()
