#!/usr/bin/env python3
"""
precompute_spk_embeddings.py — cache frozen RawNet3 speaker embeddings for
every speaker-labelled utterance in the ASR/AST index.

Pipeline position:
    ASR_INDEX_V3.csv ──▶ precompute_spk_embeddings.py ──▶ spk/  (192-d vectors)
                                                          │
                            TTSDataset reads spk/ + dac/ at train time, joining
                            on audio_id, and conditions SpeechAura-TTS on a
                            *sibling*-utterance embedding (same speaker_id,
                            different audio_id) to avoid same-utterance leakage.

Model: espnet/voxcelebs12_rawnet3 via espnet2 Speech2Embedding (frozen, eval).
Input: 16 kHz mono waveform (same rate as the rest of the project). Output:
one (192,) vector per audio_id.

Why a precompute pass (not in the dataloader):
  - RawNet3 lives behind ESPnet's Speech2Embedding wrapper (own frontend,
    own config); running it per-batch inside DDP is slow and awkward to place
    on-device cleanly. Conditioning is frozen, so compute once and cache.
  - Sibling-speaker sampling at train time only needs the cached vectors.

Storage (per shard, flat — vectors are fixed 192-d so no offset index):
    spk_{shard:05d}.npy        float16 (n_utts, 192)
    spk_{shard:05d}.ids.json   list[str] audio_id in row order
A separate merge step (--merge) stitches all shards into:
    spk_embeddings.npy         float16 (N, 192)
    spk_index.json             {audio_id -> row, dim, n_utts, ...}

Sharding / resume (matches tokenize_corpus.py ethos):
    Run one process per shard across Bridges2 array tasks:
        python precompute_spk_embeddings.py --index ... --output-dir ... \
            --shard-id $SLURM_ARRAY_TASK_ID --num-shards $SLURM_ARRAY_TASK_COUNT
    Re-running a shard skips audio_ids already present in that shard's sidecar.
    --overwrite rebuilds the shard from scratch.

Usage:
    # single shard (or whole corpus with --num-shards 1)
    python scripts/precompute_spk_embeddings.py \
        --index /ocean/projects/cis250145p/shared/ASR_INDEX_V3_16k.csv \
        --spk-model-dir ./models/voxcelebs12_rawnet3 \
        --output-dir ./tts_cache/spk  \
        --shard-id 0 --num-shards 8

    # after all shards finish, merge:
    python scripts/precompute_spk_embeddings.py \
        --output-dir ./tts_cache/spk  \
        --merge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from precompute_common import (
    TARGET_SR, read_index, shard_rows, dedup_keep_first,
    VectorShardWriter, VEC_DTYPE_STR, load_done_ids_vectors, Heartbeat, fmt_count,
)

SPK_DIM = 192   # RawNet3 (voxcelebs12_rawnet3) embedding dimension


# ============================================================================
#  Audio loading (matches st.inference.generate.load_audio: 16 kHz mono float32)
# ============================================================================

def load_waveform_16k_mono(path: str) -> "np.ndarray | None":
    """Return a 1-D float32 numpy waveform at 16 kHz, or None on failure."""
    import soundfile as sf
    import torchaudio.functional as AF
    import torch

    try:
        data, sr = sf.read(path, dtype="float32")
    except Exception as e:  # noqa: BLE001 — keep the pass alive on bad files
        print(f"[warn] read failed {path}: {e}", file=sys.stderr)
        return None
    if data.ndim > 1:
        data = data[:, 0]
    wav = torch.from_numpy(np.ascontiguousarray(data))
    if sr != TARGET_SR:
        wav = AF.resample(wav, sr, TARGET_SR)
    return wav.numpy().astype(np.float32, copy=False)


# ============================================================================
#  Model
# ============================================================================

def load_spk_model(model_dir: str, device: str):
    """Load the frozen RawNet3 Speech2Embedding from a snapshot dir.

    Mirrors the user's known-good loader: locate config.yaml + *.pth by glob
    rather than hardcoding the exp subdir name.
    """
    from espnet2.bin.spk_inference import Speech2Embedding

    model_dir = Path(model_dir)
    try:
        config = next(model_dir.rglob("config.yaml"))
        model_file = next(model_dir.rglob("*.pth"))
    except StopIteration:
        raise FileNotFoundError(
            f"Could not find config.yaml / *.pth under {model_dir}. "
            f"Download first, e.g. huggingface_hub.snapshot_download("
            f"'espnet/voxcelebs12_rawnet3', allow_patterns=['*.yaml','*.pth'], "
            f"local_dir='{model_dir}')."
        )
    print(f"[init] RawNet3 config={config.name} weights={model_file.name} "
          f"device={device}", file=sys.stderr)
    return Speech2Embedding(
        train_config=str(config), model_file=str(model_file), device=device,
    )


def embed_one(spk_model, wav: np.ndarray) -> "np.ndarray | None":
    """Run RawNet3 on a 1-D 16 kHz waveform → (192,) float32, or None."""
    import torch
    if wav.size < TARGET_SR // 10:   # < 100 ms is too short to be meaningful
        return None
    try:
        with torch.no_grad():
            emb = spk_model(wav)            # (1, 192) tensor
        emb = emb.squeeze(0).detach().to("cpu").numpy().astype(np.float32)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] embed failed: {e}", file=sys.stderr)
        return None
    if emb.shape[-1] != SPK_DIM:
        print(f"[warn] unexpected embedding dim {emb.shape}; skipping",
              file=sys.stderr)
        return None
    return emb


# ============================================================================
#  Encode pass (one shard)
# ============================================================================

def run_shard(args) -> None:
    import torch

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "spk"

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

    npy_path = out_dir / f"{prefix}_{args.shard_id:05d}.npy"
    ids_path = out_dir / f"{prefix}_{args.shard_id:05d}.ids.json"

    done: set[str] = set()
    if args.overwrite:
        for p in (npy_path, ids_path):
            if p.exists():
                p.unlink()
    else:
        done = load_done_ids_vectors(out_dir, prefix, args.shard_id)
        if done:
            print(f"[resume] {len(done):,} already embedded in this shard; "
                  f"appending the rest", file=sys.stderr)

    device = ("cuda" if (args.device == "cuda" and torch.cuda.is_available())
              else args.device if args.device != "cuda" else "cpu")
    spk_model = load_spk_model(args.spk_model_dir, device)

    writer = VectorShardWriter(npy_path, ids_path, dim=SPK_DIM)
    # On resume we must re-load prior vectors so close() rewrites the full shard.
    if done:
        prev = np.load(npy_path)
        prev_ids = json.loads(ids_path.read_text())
        for aid, vec in zip(prev_ids, prev):
            writer.append(aid, vec)

    hb = Heartbeat(total=len(rows), label=f"spk s{args.shard_id}")
    n_done = n_skip = n_fail = 0
    for i, r in enumerate(rows):
        if r.audio_id in done:
            n_skip += 1
            hb.tick(i + 1, extra=f"ok={n_done} skip={n_skip} fail={n_fail}")
            continue
        wav = load_waveform_16k_mono(r.path)
        if wav is None:
            n_fail += 1
            hb.tick(i + 1, extra=f"ok={n_done} skip={n_skip} fail={n_fail}")
            continue
        emb = embed_one(spk_model, wav)
        if emb is None:
            n_fail += 1
            hb.tick(i + 1, extra=f"ok={n_done} skip={n_skip} fail={n_fail}")
            continue
        writer.append(r.audio_id, emb)
        done.add(r.audio_id)
        n_done += 1

        # Periodic flush so a kill late in a long shard doesn't lose everything.
        if args.flush_every and n_done % args.flush_every == 0:
            writer.close()  # rewrites full shard; safe to call repeatedly
        hb.tick(i + 1, extra=f"ok={n_done} skip={n_skip} fail={n_fail}")

    man = writer.close()
    hb.tick(len(rows), force=True, extra=f"ok={n_done} skip={n_skip} fail={n_fail}")
    print(f"[done] shard {args.shard_id}: wrote {man['n_utts']:,} vectors "
          f"({n_fail:,} failed) -> {npy_path.name}", file=sys.stderr)


# ============================================================================
#  Merge step (all shards -> single matrix + index)
# ============================================================================

def run_merge(args) -> None:
    out_dir = Path(args.output_dir)
    prefix = "spk"

    shard_npys = sorted(out_dir.glob(f"{prefix}_*.npy"))
    if not shard_npys:
        print(f"[error] no shards found under {out_dir}", file=sys.stderr)
        sys.exit(1)

    all_ids: list[str] = []
    mats: list[np.ndarray] = []
    seen: set[str] = set()
    dup = 0
    for npy in shard_npys:
        ids_path = npy.with_suffix("").with_suffix(".ids.json")
        ids = json.loads(ids_path.read_text())
        mat = np.load(npy)
        assert mat.shape[0] == len(ids), f"{npy} length mismatch"
        # Dedup across shards (shouldn't happen with strided sharding, but be safe)
        keep_rows = []
        for j, aid in enumerate(ids):
            if aid in seen:
                dup += 1
                continue
            seen.add(aid)
            all_ids.append(aid)
            keep_rows.append(j)
        if keep_rows:
            mats.append(mat[keep_rows])
        print(f"[merge] {npy.name}: {len(ids):,} rows", file=sys.stderr)

    full = np.concatenate(mats, axis=0) if mats else np.zeros((0, SPK_DIM), np.float16)
    index = {aid: row for row, aid in enumerate(all_ids)}

    out_npy = out_dir / "spk_embeddings.npy"
    out_idx = out_dir / "spk_index.json"
    np.save(out_npy, full)
    out_idx.write_text(json.dumps({
        "dim": SPK_DIM,
        "vec_dtype": VEC_DTYPE_STR,
        "n_utts": int(full.shape[0]),
        "model": "espnet/voxcelebs12_rawnet3",
        "index": index,
    }))
    if dup:
        print(f"[merge] dropped {dup:,} cross-shard duplicate ids", file=sys.stderr)
    print(f"[merge] -> {out_npy} ({fmt_count(full.shape[0])} vectors, "
          f"{full.nbytes/1e9:.2f} GB) + {out_idx.name}", file=sys.stderr)
    _write_readme(out_dir, n=int(full.shape[0]))


def _write_readme(out_dir: Path, n: int) -> None:
    (out_dir / "README.md").write_text(f"""# Speaker embedding cache (RawNet3)

Frozen `espnet/voxcelebs12_rawnet3` speaker embeddings, one {SPK_DIM}-d vector
per speaker-labelled utterance, keyed by `audio_id`.

## Files
- `spk_embeddings.npy` — float16 (N, {SPK_DIM}), N={n:,}
- `spk_index.json`     — {{audio_id -> row}} into the matrix
- `spk_*.npy` / `spk_*.ids.json` — per-shard intermediates (safe to delete after merge)

## Reading
```python
import json, numpy as np
mat = np.load("spk_embeddings.npy", mmap_mode="r")   # (N, {SPK_DIM}) fp16
idx = json.load(open("spk_index.json"))["index"]
vec = mat[idx["some_audio_id"]]                       # (192,) fp16
```

## Notes
- Input is 16 kHz mono (project standard). RawNet3 is VoxCeleb-trained; identity
  vectors transfer across languages but the space is less discriminative for
  out-of-domain speakers — acceptable for v1 zero-shot TTS.
- Only rows with a populated `speaker_id` are embedded. Backfill speaker_id and
  re-run (resume skips already-done audio_ids) to extend coverage.
- Train-time conditioning samples a SIBLING utterance (same speaker_id, different
  audio_id) to prevent same-utterance content leakage.
""")


# ============================================================================
#  CLI
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", help="ASR/AST index CSV (required unless --merge)")
    ap.add_argument("--spk-model-dir", default="./models/voxcelebs12_rawnet3",
                    help="Dir containing RawNet3 config.yaml + *.pth")
    ap.add_argument("--output-dir", required=True, help="Where to write spk/ cache")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--langs", default=None, help="Comma-separated language allow-list")
    ap.add_argument("--sources", default=None,
                    help="Comma-separated source allow-list (matches the `source` "
                         "column, e.g. commonvoice,fleurs,bigc). AND-combined with --langs.")
    ap.add_argument("--max-duration", type=float, default=None,
                    help="Skip utterances longer than this (s)")
    ap.add_argument("--min-duration", type=float, default=0.1,
                    help="Skip utterances shorter than this (s)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--flush-every", type=int, default=5000,
                    help="Rewrite shard files every N new vectors (crash safety)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Rebuild this shard from scratch")
    ap.add_argument("--merge", action="store_true",
                    help="Merge all shards into spk_embeddings.npy + spk_index.json")
    args = ap.parse_args()

    if args.merge:
        run_merge(args)
        return
    if not args.index:
        ap.error("--index is required unless --merge is given")
    run_shard(args)


if __name__ == "__main__":
    main()