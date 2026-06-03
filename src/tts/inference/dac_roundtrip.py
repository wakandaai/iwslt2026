"""
DAC round-trip diagnostic.

Decodes *cached* DAC codes (the exact training targets) straight back to audio
through the same decode path generation uses (tts_generate.decode_codes). If
these wavs sound like real speech, the cache + DAC decode + codebook ordering
are all correct, and any silence at generation time is the model's fault — not
the decode path. If these are also silence, the bug is upstream (bad cache or a
codebook-layout mismatch with DAC).

Also dumps per-codebook value spread so you can compare a real example's code
distribution against a generated one (constant codes across T == collapse).

Usage:
    python -m tts.inference.dac_roundtrip \
        --config configs/experiment/tts_stage1.yaml \
        --num 3 --out_dir roundtrip_out

    # or specific utterances:
    python -m tts.inference.dac_roundtrip --config ... --audio_id abc123 def456
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from core.utils.config import load_config
from tts.data.code_store import CodeStore
from tts.inference.tts_generate import decode_codes, load_dac

log = logging.getLogger(__name__)


def _code_stats(codes_TK: torch.Tensor) -> str:
    """One-line spread summary: distinct values + top value share per codebook."""
    T, K = codes_TK.shape
    parts = []
    for k in range(K):
        col = codes_TK[:, k]
        n_uniq = int(col.unique().numel())
        _, counts = col.unique(return_counts=True)
        top_share = float(counts.max().item()) / T
        parts.append(f"k{k}:{n_uniq}u/{top_share:.0%}")
    return " ".join(parts)


def run(args) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    store = CodeStore(cfg["data"]["dac_cache"])
    log.info(f"CodeStore: {len(store)} utts, K={store.K}, "
             f"card={store.cardinality}, frame_rate={store.frame_rate}")

    if args.audio_id:
        ids = args.audio_id
    else:
        ids = store.audio_ids()[: args.num]
    missing = [a for a in ids if a not in store]
    if missing:
        raise KeyError(f"audio_id(s) not in cache: {missing}")

    dac_model = load_dac(args.dac_model_type, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    for aid in ids:
        codes = store[aid]                              # (T, K) long
        dur = codes.size(0) / store.frame_rate
        log.info(f"{aid}: {codes.size(0)} frames ({dur:.2f}s) | {_code_stats(codes)}")
        wav = decode_codes(dac_model, codes, device)
        path = out_dir / f"roundtrip_{aid}.wav"
        sf.write(str(path), wav.numpy(), dac_model.sample_rate)
        peak = float(wav.abs().max().item())
        rms = float(wav.pow(2).mean().sqrt().item())
        log.info(f"  → {path}  ({wav.numel() / dac_model.sample_rate:.2f}s, "
                 f"peak={peak:.3f}, rms={rms:.4f})")
    log.info("Done. Listen to the wavs: real speech ⇒ decode path is fine.")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="DAC cache round-trip diagnostic")
    p.add_argument("--config", required=True, help="Experiment YAML (data.dac_cache)")
    p.add_argument("--audio_id", nargs="*", default=None,
                   help="Specific audio_id(s) to decode. Default: first --num in cache.")
    p.add_argument("--num", type=int, default=3, help="How many utts if no --audio_id")
    p.add_argument("--out_dir", default="roundtrip_out")
    p.add_argument("--dac_model_type", default="16khz")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
