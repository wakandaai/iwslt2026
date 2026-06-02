#!/usr/bin/env python3
"""
precompute_dac_codes.py — cache frozen DAC-16kHz RVQ codes for every
speaker-labelled utterance in the ASR/AST index.

Pipeline position:
    ASR_INDEX_V3.csv ──▶ precompute_dac_codes.py ──▶ dac/  (uint16 RVQ codes)
                                                     │
                          TTSDataset reads dac/ + spk/ at train time, joining
                          on audio_id. DAC codes are the *targets* the Aura-TTS
                          temporal+depth transformers learn to emit.

Codec: Descript Audio Codec, 16 kHz variant (dac.utils.download("16khz")).
  - frame rate ~50 Hz (hop 320 @ 16 kHz)
  - K = 12 RVQ codebooks, cardinality 1024 (10-bit) each
  - codes shape from model.encode: (B, K, T_frames); we transpose to (T, K)

Why 16 kHz: it's the project-wide sample rate (encoder, mel front-end, RawNet3
all 16 kHz), so the codec consumes the exact waveforms everything else does —
no resample fork. 16 kHz also gives the lowest frame rate of the three DAC
variants (50 Hz vs 75/86), maximising temporal context budget for Aura's
1024-position window. The 12-codebook depth axis is absorbed by the depth
transformer, so it costs nothing on the temporal side.

Storage (per shard — variable T per utterance, so concatenated stream + index,
exactly like tokenize_corpus.py):
    dac_{shard:05d}.bin    uint16 LE, all utterances' (T_frames, K) grids,
                           time-major (frame t's K codes contiguous)
    dac_{shard:05d}.idx    uint64 LE frame offsets, length n_utts + 1
    dac_{shard:05d}.json   per-utterance records: audio_id -> (row, n_frames, ...)
A merge step (--merge) writes manifest.json tying shards together for the loader.

Reading one utterance at train time:
    off = np.fromfile("dac_00000.idx", dtype=np.uint64)
    buf = np.memmap("dac_00000.bin", dtype=np.uint16, mode="r")
    # utterance i (row in shard manifest), K=12:
    codes_TK = buf[K*off[i] : K*off[i+1]].reshape(-1, K)   # (T_i, 12)

Sharding / resume: identical contract to precompute_spk_embeddings.py.
    python precompute_dac_codes.py --index ... --output-dir ... \
        --shard-id $SLURM_ARRAY_TASK_ID --num-shards $SLURM_ARRAY_TASK_COUNT
Re-running skips audio_ids already in this shard's manifest; --overwrite rebuilds.

Usage:
    python scripts/precompute_dac_codes.py \
        --index /ocean/projects/cis250145p/shared/ASR_INDEX_V3_16k.csv \
        --output-dir ./tts_cache/dac \
        --model-type 16khz --n-codebooks 12 \
        --shard-id 0 --num-shards 8

    python scripts/precompute_dac_codes.py \
        --output-dir ./tts_cache/dac --merge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from precompute_common import (
    TARGET_SR, read_index, shard_rows, dedup_keep_first,
    CodeShardWriter, CODE_DTYPE_STR, OFFSET_DTYPE_STR,
    load_done_ids_codes, Heartbeat, fmt_count,
)

# 16 kHz DAC frame rate (hop 320). Used only for the duration->frames ETA hint
# and the manifest; the authoritative T comes from the actual encode.
DAC_16K_FRAME_RATE = 50.0


# ============================================================================
#  Model
# ============================================================================

def load_dac(model_type: str, device: str):
    """Download (cached) + load a frozen DAC model in eval mode."""
    import dac
    import torch

    model_path = dac.utils.download(model_type=model_type)
    model = dac.DAC.load(model_path)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[init] DAC-{model_type} loaded on {device} "
          f"(sample_rate={model.sample_rate})", file=sys.stderr)
    if model.sample_rate != TARGET_SR:
        print(f"[warn] DAC sample_rate {model.sample_rate} != project "
              f"{TARGET_SR}; waveforms will be resampled to {model.sample_rate} "
              f"by preprocess", file=sys.stderr)
    return model


def _codes_to_TK(codes, n_codebooks: int, tag: str):
    """Shared post-processing: (1,K_full,T)|(K_full,T) tensor -> (T, K) np, or None."""
    codes = codes.detach().to("cpu")
    if codes.dim() == 3:
        codes = codes[0]
    elif codes.dim() != 2:
        print(f"[warn] unexpected codes shape {tuple(codes.shape)} for {tag}",
              file=sys.stderr)
        return None
    K_full = codes.shape[0]
    k = min(n_codebooks, K_full)
    if n_codebooks > K_full:
        print(f"[warn] requested {n_codebooks} codebooks but model has {K_full}; "
              f"using {K_full}", file=sys.stderr)
    codes = codes[:k]                                          # (K, T) coarse->fine
    codes_TK = codes.transpose(0, 1).contiguous().numpy()      # (T, K) time-major
    return codes_TK if codes_TK.size else None


def encode_one(model, path: str, n_codebooks: int, chunk_long: bool):
    """Encode one file → (T_frames, K) int array, or None on failure.

    Uses model.compress() for long-file constant-memory chunked encoding when
    chunk_long=True (recommended for utterances over ~20 s); otherwise the
    direct preprocess+encode path (faster for typical short utterances).
    """
    import torch
    from audiotools import AudioSignal

    try:
        signal = AudioSignal(path)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] AudioSignal load failed {path}: {e}", file=sys.stderr)
        return None

    try:
        with torch.no_grad():
            if chunk_long:
                # compress() handles resample-to-model-SR, mono, dB norm, and
                # constant-memory chunking; returns a DACFile with full codes.
                signal = signal.to(model.device)
                dac_file = model.compress(signal)
                codes = dac_file.codes            # (1, K_full, T) on CPU
            else:
                signal = signal.to(model.device)
                x = model.preprocess(signal.audio_data, signal.sample_rate)
                _, codes, _, _, _ = model.encode(x)   # (1, K_full, T)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] encode failed {path}: {e}", file=sys.stderr)
        return None

    return _codes_to_TK(codes, n_codebooks, path)


# ============================================================================
#  Encode pass (one shard)
# ============================================================================

def run_shard(args) -> None:
    import torch

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "dac"

    rows = read_index(
        args.index, require_speaker=True,
        languages=[l.strip() for l in args.langs.split(",")] if args.langs else None,
        sources=[s.strip() for s in args.sources.split(",")] if args.sources else None,
        max_duration=args.max_duration, min_duration=args.min_duration,
    )
    rows = dedup_keep_first(rows)
    rows = shard_rows(rows, args.shard_id, args.num_shards)
    print(f"[shard {args.shard_id}/{args.num_shards}] {len(rows):,} utterances",
          file=sys.stderr)

    bin_path = out_dir / f"{prefix}_{args.shard_id:05d}.bin"
    idx_path = out_dir / f"{prefix}_{args.shard_id:05d}.idx"
    json_path = out_dir / f"{prefix}_{args.shard_id:05d}.json"

    done: set[str] = set()
    if args.overwrite:
        for p in (bin_path, idx_path, json_path):
            if p.exists():
                p.unlink()
    else:
        done = load_done_ids_codes(out_dir, prefix, args.shard_id)
        if done:
            print(f"[resume] {len(done):,} already encoded in this shard; "
                  f"appending the rest", file=sys.stderr)

    device = ("cuda" if (args.device == "cuda" and torch.cuda.is_available())
              else "cpu" if args.device == "cuda" else args.device)
    model = load_dac(args.model_type, device)

    # resume=True opens the existing bin in append mode and restores offsets +
    # records (never truncates prior bytes). On a fresh shard it starts clean.
    writer = CodeShardWriter(
        bin_path, idx_path, json_path,
        n_codebooks=args.n_codebooks, resume=bool(done),
    )

    hb = Heartbeat(total=len(rows), label=f"dac s{args.shard_id}")
    n_done = n_skip = n_fail = 0
    frames_total = 0
    for i, r in enumerate(rows):
        if r.audio_id in done:
            n_skip += 1
            hb.tick(i + 1, extra=f"ok={n_done} skip={n_skip} fail={n_fail} "
                                 f"{fmt_count(frames_total)}fr")
            continue
        codes_TK = encode_one(model, r.path, args.n_codebooks,
                              chunk_long=(r.duration == 0.0 or r.duration > args.chunk_threshold))
        if codes_TK is None:
            n_fail += 1
            hb.tick(i + 1, extra=f"ok={n_done} skip={n_skip} fail={n_fail} "
                                 f"{fmt_count(frames_total)}fr")
            continue
        writer.append(r.audio_id, codes_TK,
                      meta={"language": r.language, "speaker_id": r.speaker_id})
        done.add(r.audio_id)
        n_done += 1
        frames_total += codes_TK.shape[0]

        if args.flush_every and n_done % args.flush_every == 0:
            writer.flush()   # idx + json reflect current bin; crash-safe checkpoint
        hb.tick(i + 1, extra=f"ok={n_done} skip={n_skip} fail={n_fail} "
                             f"{fmt_count(frames_total)}fr")

    man = writer.close()
    hb.tick(len(rows), force=True,
            extra=f"ok={n_done} skip={n_skip} fail={n_fail} {fmt_count(frames_total)}fr")
    print(f"[done] shard {args.shard_id}: {man['n_utts']:,} utts / "
          f"{fmt_count(man['n_frames_total'])} frames ({n_fail:,} failed) "
          f"-> {bin_path.name}", file=sys.stderr)


# ============================================================================
#  Merge step (build a single manifest tying shards together)
# ============================================================================

def run_merge(args) -> None:
    out_dir = Path(args.output_dir)
    prefix = "dac"
    shard_jsons = sorted(out_dir.glob(f"{prefix}_*.json"))
    # exclude an existing top-level manifest.json if present
    shard_jsons = [p for p in shard_jsons if p.name != "manifest.json"]
    if not shard_jsons:
        print(f"[error] no shards under {out_dir}", file=sys.stderr)
        sys.exit(1)

    shards = []
    index: dict[str, list[int]] = {}   # audio_id -> [shard_idx, row]
    total_utts = total_frames = 0
    K = None
    dup = 0
    for sj in shard_jsons:
        man = json.loads(sj.read_text())
        shard_idx = int(sj.stem.split("_")[-1])
        K = man["n_codebooks"] if K is None else K
        if man["n_codebooks"] != K:
            print(f"[error] codebook mismatch in {sj.name}: "
                  f"{man['n_codebooks']} != {K}", file=sys.stderr)
            sys.exit(1)
        shards.append({
            "shard_idx": shard_idx,
            "bin_path": man["bin_path"], "idx_path": man["idx_path"],
            "n_utts": man["n_utts"], "n_frames_total": man["n_frames_total"],
        })
        for rec in man["records"]:
            aid = rec["audio_id"]
            if aid in index:
                dup += 1
                continue
            index[aid] = [shard_idx, rec["row"]]
        total_utts += man["n_utts"]
        total_frames += man["n_frames_total"]
        print(f"[merge] {sj.name}: {man['n_utts']:,} utts / "
              f"{fmt_count(man['n_frames_total'])} frames", file=sys.stderr)

    manifest = {
        "codec": f"dac-{args.model_type}",
        "sample_rate": TARGET_SR,
        "frame_rate": DAC_16K_FRAME_RATE,
        "n_codebooks": K,
        "cardinality": 1024,
        "code_dtype": CODE_DTYPE_STR,
        "offset_dtype": OFFSET_DTYPE_STR,
        "layout": "time_major_(T,K)",
        "n_utts": total_utts - dup,
        "n_frames_total": total_frames,
        "shards": shards,
        "index": index,   # audio_id -> [shard_idx, row]
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    if dup:
        print(f"[merge] dropped {dup:,} cross-shard duplicate ids", file=sys.stderr)
    print(f"[merge] -> manifest.json ({fmt_count(manifest['n_utts'])} utts, "
          f"{fmt_count(total_frames)} frames, K={K})", file=sys.stderr)
    _write_readme(out_dir, manifest)


def _write_readme(out_dir: Path, manifest: dict) -> None:
    K = manifest["n_codebooks"]
    (out_dir / "README.md").write_text(f"""# DAC code cache ({manifest['codec']})

Frozen Descript Audio Codec RVQ codes, keyed by `audio_id`. Targets for
Aura-TTS (temporal transformer predicts codebook 0 per frame; depth transformer
predicts codebooks 1..{K-1}).

## Config
- codec: {manifest['codec']}  |  sample_rate: {manifest['sample_rate']} Hz
- frame_rate: ~{manifest['frame_rate']} Hz  |  K = {K} codebooks, cardinality {manifest['cardinality']}
- code dtype: {manifest['code_dtype']}  |  layout: {manifest['layout']}
- {fmt_count(manifest['n_utts'])} utterances, {fmt_count(manifest['n_frames_total'])} frames

## Files
- `manifest.json` — {{audio_id -> [shard_idx, row]}} + shard list (loader entry point)
- `dac_*.bin` — uint16 code stream, time-major (T, {K}) per utterance concatenated
- `dac_*.idx` — uint64 frame offsets (length n_utts+1 per shard)
- `dac_*.json` — per-shard records (audio_id, row, n_frames, language, speaker_id)

## Reading one utterance
```python
import json, numpy as np
man = json.load(open("manifest.json"))
K = man["n_codebooks"]
shard_idx, row = man["index"]["some_audio_id"]
shard = next(s for s in man["shards"] if s["shard_idx"] == shard_idx)
off = np.fromfile(f"dac_{{shard_idx:05d}}.idx", dtype=np.uint64)
buf = np.memmap(f"dac_{{shard_idx:05d}}.bin", dtype=np.uint16, mode="r")
codes_TK = buf[K*off[row] : K*off[row+1]].reshape(-1, K)   # (T, {K})
```

## Notes
- Only speaker-labelled rows are encoded (matches the spk/ cache). Backfill
  speaker_id and re-run (resume skips done audio_ids) to extend coverage.
- 16 kHz keeps the whole project on one sample rate and minimises frame count.
- To decode codes back to audio: DAC `model.decode` after `quantizer.from_codes`.
""")


# ============================================================================
#  CLI
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", help="ASR/AST index CSV (required unless --merge)")
    ap.add_argument("--output-dir", required=True, help="Where to write dac/ cache")
    ap.add_argument("--model-type", default="16khz", choices=["16khz", "24khz", "44khz"])
    ap.add_argument("--n-codebooks", type=int, default=12,
                    help="Number of coarse-to-fine RVQ codebooks to keep "
                         "(16khz model has 12)")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--langs", default=None, help="Comma-separated language allow-list")
    ap.add_argument("--sources", default=None,
                    help="Comma-separated source allow-list (matches the `source` "
                         "column, e.g. commonvoice,fleurs,bigc). AND-combined with --langs.")
    ap.add_argument("--max-duration", type=float, default=None,
                    help="Skip utterances longer than this (s). For TTS training "
                         "a cap ~15s keeps frame counts inside Aura's context.")
    ap.add_argument("--min-duration", type=float, default=0.1)
    ap.add_argument("--chunk-threshold", type=float, default=20.0,
                    help="Use DAC compress() chunked encode above this duration (s)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--flush-every", type=int, default=2000,
                    help="Persist idx+json every N new utterances (crash safety)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="Build manifest.json tying all shards together")
    args = ap.parse_args()

    if args.merge:
        run_merge(args)
        return
    if not args.index:
        ap.error("--index is required unless --merge is given")
    run_shard(args)


if __name__ == "__main__":
    main()