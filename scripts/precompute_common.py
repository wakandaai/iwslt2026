#!/usr/bin/env python3
"""
precompute_common.py — shared helpers for the TTS precompute passes.

Two precompute scripts consume this:
    precompute_spk_embeddings.py   ASR_INDEX_V3.csv -> spk/  (RawNet3 192-d vectors)
    precompute_dac_codes.py        ASR_INDEX_V3.csv -> dac/  (DAC-16kHz RVQ codes)

Both passes share:
  - the same row filter (must have a populated `speaker_id`),
  - the same `audio_id`-keyed identity (robust to later index reordering),
  - the same shard/resume contract (`--shard-id N --num-shards M`),
  - the same "concatenated binary stream + offset index + manifest" storage
    pattern used by tokenize_corpus.py, so the training-time loader is an
    mmap slice with flat RSS across DDP workers.

Nothing here imports torch / dac / espnet — it's pure stdlib + numpy so it can
be imported cheaply by either pass and by the (CPU-side) merge step.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# Audio loading is identical to the rest of the repo (st.inference.generate.load_audio):
# 16 kHz mono float32. Both DAC-16kHz and RawNet3 expect 16 kHz, so there is a
# single resample target and no per-pass fork.
TARGET_SR = 16000

# CSV is large (~7.9M rows). Bump the field-size limit so a pathological long
# transcript can't abort the reader mid-pass.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


# ============================================================================
#  Index row
# ============================================================================

@dataclass
class IndexRow:
    audio_id: str
    path: str
    speaker_id: str
    language: str
    source: str
    duration: float


def _get(row: dict, *names: str, default: str = "") -> str:
    """First non-empty value among `names` (handles ASR/AST column aliases)."""
    for n in names:
        v = row.get(n)
        if v is not None and str(v).strip():
            return str(v).strip()
    return default


def read_index(
    index_path: str | Path,
    require_speaker: bool = True,
    languages: list[str] | None = None,
    sources: list[str] | None = None,
    max_duration: float | None = None,
    min_duration: float | None = None,
) -> list[IndexRow]:
    """Stream the ASR/AST index CSV into a list of IndexRow.

    Filters (in order, matching the project's dataset.py semantics). Multiple
    filters are AND-combined — a row must pass every active filter to be kept:
      - drop rows with no `speaker_id` when require_speaker=True (per request:
        rows without a speaker label are excluded; speaker_id will be
        backfilled separately and the pass re-run for the new rows),
      - optional language allow-list (matches `language` or `src_language`),
      - optional source allow-list (matches the `source` column),
      - optional duration bounds.

    Rows are returned in CSV order; callers shard deterministically afterwards
    so two passes over the same CSV agree on which audio_id lands in which shard.
    """
    index_path = Path(index_path)
    lang_set = set(languages) if languages else None
    source_set = set(sources) if sources else None

    rows: list[IndexRow] = []
    n_total = 0
    n_no_spk = 0
    n_no_path = 0
    n_lang = 0
    n_source = 0
    n_dur = 0

    with open(index_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            n_total += 1

            spk = _get(r, "speaker_id", "speaker", "spk")
            if require_speaker and not spk:
                n_no_spk += 1
                continue

            path = _get(r, "path", "audio_path")
            if not path:
                n_no_path += 1
                continue

            lang = _get(r, "language", "src_language")
            if lang_set is not None and lang not in lang_set:
                n_lang += 1
                continue

            source = _get(r, "source")
            if source_set is not None and source not in source_set:
                n_source += 1
                continue

            dur_str = _get(r, "duration")
            dur = float(dur_str) if dur_str else 0.0
            if max_duration is not None and dur > max_duration:
                n_dur += 1
                continue
            if min_duration is not None and dur and dur < min_duration:
                n_dur += 1
                continue

            audio_id = _get(r, "audio_id")
            if not audio_id:
                # Fall back to path stem if audio_id is absent — still stable
                # across reruns as long as paths are stable.
                audio_id = Path(path).stem

            rows.append(IndexRow(
                audio_id=audio_id, path=path, speaker_id=spk,
                language=lang, source=source, duration=dur,
            ))

    print(
        f"[index] {index_path}: {n_total:,} rows -> kept {len(rows):,} "
        f"(dropped: no_speaker={n_no_spk:,}, no_path={n_no_path:,}, "
        f"lang_filtered={n_lang:,}, source_filtered={n_source:,}, "
        f"dur_filtered={n_dur:,})",
        file=sys.stderr,
    )
    return rows


def shard_rows(rows: list[IndexRow], shard_id: int, num_shards: int) -> list[IndexRow]:
    """Strided shard slice: shard s gets rows[s::num_shards].

    Strided (not contiguous) so each shard sees a roughly uniform mix of
    languages / durations — the same load-balancing logic as the audit pass.
    """
    if num_shards <= 1:
        return rows
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id {shard_id} out of range [0,{num_shards})")
    return rows[shard_id::num_shards]


def dedup_keep_first(rows: list[IndexRow]) -> list[IndexRow]:
    """Drop duplicate audio_ids (some merged indices repeat ids). Keep first."""
    seen: set[str] = set()
    out: list[IndexRow] = []
    for r in rows:
        if r.audio_id in seen:
            continue
        seen.add(r.audio_id)
        out.append(r)
    if len(out) != len(rows):
        print(f"[index] dropped {len(rows) - len(out):,} duplicate audio_ids",
              file=sys.stderr)
    return out


# ============================================================================
#  Variable-length code store  (DAC) — concatenated bin + offset idx + manifest
# ============================================================================
#
# Mirrors tokenize_corpus.py's shard format. One shard == one writer ==
# one (shard_id) of the fan-out. Layout per shard:
#
#   {prefix}_{shard:05d}.bin   uint16 LE, all utterances' (T_frames, K) grids
#                              concatenated row-major (time-major; see below)
#   {prefix}_{shard:05d}.idx   uint64 LE frame offsets, length n_utts + 1
#   {prefix}_{shard:05d}.json  per-utterance records: audio_id -> (row, T, ...)
#
# Time-major means each utterance is stored as a contiguous (T_frames, K)
# block, so frame t's K codebooks are adjacent in memory. The training loader
# slices utterance i as bin[K*off[i] : K*off[i+1]].reshape(T_i, K) and the
# depth transformer consumes each length-K row directly.

CODE_DTYPE = np.uint16        # cardinality 1024 (10-bit) fits; tight on disk
CODE_DTYPE_STR = "uint16"
OFFSET_DTYPE = np.uint64
OFFSET_DTYPE_STR = "uint64"


class CodeShardWriter:
    """Append (T_frames, K) uint16 code grids to one shard. O(n_utts) RAM.

    Two open modes:
      resume=False (default): truncate any existing bin and start fresh.
      resume=True: open the existing bin in APPEND mode and restore offsets +
                   records from the sidecar idx/json, so previously-encoded
                   utterances are preserved. The bin is NEVER truncated in this
                   mode — critical, since truncating then "reloading" would lose
                   all prior bytes (the prior offsets would point past EOF).
    """

    def __init__(self, bin_path: Path, idx_path: Path, json_path: Path,
                 n_codebooks: int, resume: bool = False):
        self.bin_path = bin_path
        self.idx_path = idx_path
        self.json_path = json_path
        self.K = n_codebooks

        if resume and bin_path.exists() and idx_path.exists() and json_path.exists():
            man = json.loads(json_path.read_text())
            if man.get("n_codebooks", n_codebooks) != n_codebooks:
                raise ValueError(
                    f"resume codebook mismatch: shard has {man.get('n_codebooks')}, "
                    f"requested {n_codebooks}")
            offsets = np.fromfile(idx_path, dtype=OFFSET_DTYPE).tolist()
            self.frame_offsets = offsets if offsets else [0]
            self.records = man.get("records", [])
            self.n_frames_total = man.get(
                "n_frames_total", self.frame_offsets[-1] if self.frame_offsets else 0)
            # Truncate the bin to exactly the bytes the offsets account for, in
            # case a hard kill left a partial trailing write past the last
            # persisted offset. Then append.
            expected_bytes = self.n_frames_total * self.K * np.dtype(CODE_DTYPE).itemsize
            with open(bin_path, "r+b") as f:
                f.truncate(expected_bytes)
            self.bin_f = open(bin_path, "ab")
        else:
            self.bin_f = open(bin_path, "wb")
            self.frame_offsets = [0]            # cumulative frame counts
            self.records = []
            self.n_frames_total = 0

    def append(self, audio_id: str, codes_TK: np.ndarray, meta: dict) -> None:
        """codes_TK: (T_frames, K) array, any int dtype; cast to uint16 here."""
        assert codes_TK.ndim == 2 and codes_TK.shape[1] == self.K, (
            f"expected (T,{self.K}) got {codes_TK.shape}")
        if codes_TK.max(initial=0) >= np.iinfo(CODE_DTYPE).max:
            raise ValueError(f"code value exceeds {CODE_DTYPE_STR} range")
        arr = np.ascontiguousarray(codes_TK, dtype=CODE_DTYPE)
        self.bin_f.write(arr.tobytes())
        T = arr.shape[0]
        self.n_frames_total += T
        self.frame_offsets.append(self.n_frames_total)
        rec = {"audio_id": audio_id, "row": len(self.records),
               "n_frames": int(T)}
        rec.update(meta)
        self.records.append(rec)

    def close(self) -> dict:
        self.bin_f.close()
        np.asarray(self.frame_offsets, dtype=OFFSET_DTYPE).tofile(self.idx_path)
        shard_manifest = {
            "bin_path": self.bin_path.name,
            "idx_path": self.idx_path.name,
            "n_codebooks": self.K,
            "code_dtype": CODE_DTYPE_STR,
            "offset_dtype": OFFSET_DTYPE_STR,
            "n_utts": len(self.records),
            "n_frames_total": self.n_frames_total,
            "records": self.records,
        }
        self.json_path.write_text(json.dumps(shard_manifest))
        return shard_manifest

    def flush(self) -> None:
        """Persist idx + json for crash safety without closing the bin handle.

        The bin stream is flushed to the OS; the offset array and manifest are
        rewritten so a kill mid-shard leaves a fully readable, consistent shard
        that a later resume=True open can continue from.
        """
        self.bin_f.flush()
        np.asarray(self.frame_offsets, dtype=OFFSET_DTYPE).tofile(self.idx_path)
        self.json_path.write_text(json.dumps({
            "bin_path": self.bin_path.name, "idx_path": self.idx_path.name,
            "n_codebooks": self.K, "code_dtype": CODE_DTYPE_STR,
            "offset_dtype": OFFSET_DTYPE_STR, "n_utts": len(self.records),
            "n_frames_total": self.n_frames_total, "records": self.records,
        }))

    def existing_audio_ids(self) -> set[str]:
        return {r["audio_id"] for r in self.records}


def load_done_ids_codes(out_dir: Path, prefix: str, shard_id: int) -> set[str]:
    """Resume helper: audio_ids already written for this (prefix, shard)."""
    json_path = out_dir / f"{prefix}_{shard_id:05d}.json"
    if not json_path.exists():
        return set()
    try:
        man = json.loads(json_path.read_text())
        return {r["audio_id"] for r in man.get("records", [])}
    except (json.JSONDecodeError, OSError):
        # Corrupt/partial manifest from a hard kill — treat as not-done so the
        # shard is rebuilt cleanly under --overwrite, or warn under resume.
        print(f"[warn] unreadable manifest {json_path}; shard treated as empty",
              file=sys.stderr)
        return set()


# ============================================================================
#  Fixed-length vector store  (speaker embeddings) — flat matrix + id sidecar
# ============================================================================
#
# Speaker vectors are fixed 192-d, so no offset index needed: a dense
# (n_utts, 192) matrix + a parallel audio_id list. Stored fp16 (3 GB for 7.9M;
# RawNet3 vectors are cosine-compared downstream, fp16 is ample).
#
#   {prefix}_{shard:05d}.npy        float16 (n_utts, dim)
#   {prefix}_{shard:05d}.ids.json   list[str] audio_id in row order

VEC_DTYPE = np.float16
VEC_DTYPE_STR = "float16"


class VectorShardWriter:
    """Accumulate fixed-dim vectors for one shard, flush on close.

    Held in RAM until close because a shard of 7.9M/M rows at 192-d fp16 is
    small (a full unsharded 7.9M x 192 fp16 is ~3 GB; per-shard far less).
    Append is O(1); we np.stack once at the end.
    """

    def __init__(self, npy_path: Path, ids_path: Path, dim: int):
        self.npy_path = npy_path
        self.ids_path = ids_path
        self.dim = dim
        self.vecs: list[np.ndarray] = []
        self.ids: list[str] = []

    def append(self, audio_id: str, vec: np.ndarray) -> None:
        v = np.asarray(vec, dtype=VEC_DTYPE).reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {v.shape[0]}")
        self.vecs.append(v)
        self.ids.append(audio_id)

    def close(self) -> dict:
        if self.vecs:
            mat = np.stack(self.vecs, axis=0).astype(VEC_DTYPE)
        else:
            mat = np.zeros((0, self.dim), dtype=VEC_DTYPE)
        np.save(self.npy_path, mat)
        self.ids_path.write_text(json.dumps(self.ids))
        return {
            "npy_path": self.npy_path.name,
            "ids_path": self.ids_path.name,
            "dim": self.dim,
            "vec_dtype": VEC_DTYPE_STR,
            "n_utts": len(self.ids),
        }


def load_done_ids_vectors(out_dir: Path, prefix: str, shard_id: int) -> set[str]:
    ids_path = out_dir / f"{prefix}_{shard_id:05d}.ids.json"
    if not ids_path.exists():
        return set()
    try:
        return set(json.loads(ids_path.read_text()))
    except (json.JSONDecodeError, OSError):
        print(f"[warn] unreadable ids sidecar {ids_path}; shard treated as empty",
              file=sys.stderr)
        return set()


# ============================================================================
#  Heartbeat
# ============================================================================

class Heartbeat:
    """Throttled progress line on stderr — every `every_s` or `every_n` items."""

    def __init__(self, total: int, label: str, every_s: float = 30.0, every_n: int = 2000):
        import time
        self._time = time
        self.total = total
        self.label = label
        self.every_s = every_s
        self.every_n = every_n
        self.t0 = time.time()
        self.t_last = self.t0
        self.n_last = 0

    def tick(self, n_done: int, extra: str = "", force: bool = False) -> None:
        now = self._time.time()
        if not force and (now - self.t_last < self.every_s
                          and n_done - self.n_last < self.every_n):
            return
        elapsed = now - self.t0
        rate = n_done / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - n_done) / rate if rate > 0 else 0.0
        eta = (f"{remaining/60:.1f}min" if remaining < 3600
               else f"{remaining/3600:.1f}h")
        pct = (n_done / self.total * 100) if self.total else 0.0
        print(f"  [{self.label}] {n_done:,}/{self.total:,} ({pct:.0f}%) | "
              f"{rate:.1f} it/s | eta {eta}{(' | ' + extra) if extra else ''}",
              file=sys.stderr, flush=True)
        self.t_last = now
        self.n_last = n_done


def fmt_count(n: int) -> str:
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.2f}K"
    return str(int(n))