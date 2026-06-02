#!/usr/bin/env python3
"""
verify_tts_caches.py — cross-check the DAC code cache and the speaker-embedding
cache before TTS training, and emit a single joined manifest of trainable ids.

Both precompute passes filter to rows WITH speaker_id, so in principle they
cover the same audio_id set. In practice they diverge: an utterance can encode
fine in DAC but fail RawNet3 (too short, NaN), or vice versa, and the two
passes may have been run to different completion across shards. Training joins
the two caches by audio_id, so the trainable set is their INTERSECTION — this
tool computes it, reports the gaps, and writes a manifest the dataset reads so
it never discovers a missing key mid-epoch.

Outputs (in --out, default the dac cache dir's parent):
    tts_trainable.json   {
        "audio_ids": [...],            # sorted intersection (dac ∩ spk)
        "n_trainable": int,
        "dac_only": int, "spk_only": int,   # coverage gap sizes
        "dac_cache": "...", "spk_cache": "...",
        "n_codebooks": int, "frame_rate": float, "spk_dim": int,
    }
    tts_coverage_report.txt   human-readable gap breakdown (sample missing ids)

Usage:
    python scripts/verify_tts_caches.py \
        --dac-cache ./tts_cache/dac \
        --spk-cache ./tts_cache/spk \
        --out       ./tts_cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_dac_ids(dac_dir: Path) -> tuple[set[str], dict]:
    man_path = dac_dir / "manifest.json"
    if not man_path.is_file():
        sys.exit(f"[error] no manifest.json in {dac_dir} "
                 f"(run precompute_dac_codes.py --merge first)")
    man = json.loads(man_path.read_text())
    return set(man["index"].keys()), man


def _load_spk_ids(spk_dir: Path) -> tuple[set[str], dict]:
    idx_path = spk_dir / "spk_index.json"
    if not idx_path.is_file():
        sys.exit(f"[error] no spk_index.json in {spk_dir} "
                 f"(run precompute_spk_embeddings.py --merge first)")
    meta = json.loads(idx_path.read_text())
    return set(meta["index"].keys()), meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dac-cache", required=True, help="dir with DAC manifest.json")
    ap.add_argument("--spk-cache", required=True, help="dir with spk_index.json")
    ap.add_argument("--out", default=None,
                    help="output dir for tts_trainable.json (default: dac-cache parent)")
    ap.add_argument("--sample", type=int, default=10,
                    help="how many missing ids to list per side in the report")
    args = ap.parse_args()

    dac_dir = Path(args.dac_cache)
    spk_dir = Path(args.spk_cache)
    out_dir = Path(args.out) if args.out else dac_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    dac_ids, dac_man = _load_dac_ids(dac_dir)
    spk_ids, spk_meta = _load_spk_ids(spk_dir)

    both     = dac_ids & spk_ids
    dac_only = dac_ids - spk_ids
    spk_only = spk_ids - dac_ids

    trainable = sorted(both)

    # ---- console + report ----
    def pct(n: int, d: int) -> str:
        return f"{(100*n/d):.1f}%" if d else "—"

    lines = [
        "TTS cache coverage verification",
        "=" * 60,
        f"DAC cache:  {dac_dir}   ({len(dac_ids):,} utts, K={dac_man['n_codebooks']}, "
        f"{dac_man['frame_rate']}Hz)",
        f"SPK cache:  {spk_dir}   ({len(spk_ids):,} utts, dim={spk_meta['dim']})",
        "-" * 60,
        f"trainable (dac ∩ spk): {len(both):,}",
        f"dac only  (no spk vec): {len(dac_only):,}  ({pct(len(dac_only), len(dac_ids))} of dac)",
        f"spk only  (no dac codes): {len(spk_only):,}  ({pct(len(spk_only), len(spk_ids))} of spk)",
    ]
    if dac_only:
        lines.append(f"  e.g. dac-only ids: {sorted(dac_only)[:args.sample]}")
    if spk_only:
        lines.append(f"  e.g. spk-only ids: {sorted(spk_only)[:args.sample]}")
    if not both:
        lines.append("!! WARNING: empty intersection — caches share NO audio_ids. "
                      "Were they built from the same index / same id scheme?")
    elif len(dac_only) + len(spk_only) > 0.2 * max(len(dac_ids), len(spk_ids)):
        lines.append("!! WARNING: >20% coverage gap — likely an incomplete pass. "
                     "Resume the lagging side before training.")
    report = "\n".join(lines)
    print(report, file=sys.stderr)

    (out_dir / "tts_coverage_report.txt").write_text(report + "\n")

    trainable_manifest = {
        "audio_ids": trainable,
        "n_trainable": len(trainable),
        "dac_only": len(dac_only),
        "spk_only": len(spk_only),
        "dac_cache": str(dac_dir),
        "spk_cache": str(spk_dir),
        "n_codebooks": dac_man["n_codebooks"],
        "cardinality": dac_man["cardinality"],
        "frame_rate": dac_man["frame_rate"],
        "sample_rate": dac_man["sample_rate"],
        "spk_dim": spk_meta["dim"],
    }
    (out_dir / "tts_trainable.json").write_text(json.dumps(trainable_manifest))
    print(f"\n[done] {len(trainable):,} trainable utts -> "
          f"{out_dir/'tts_trainable.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()