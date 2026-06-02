"""
code_store.py — train-time reader for the DAC code cache produced by
scripts/precompute_dac_codes.py.

Mirrors the `_Shard` mmap pattern in the Aura text dataloader: lazily mmap each
shard's .bin, slice one utterance's (T, K) code grid on demand. mmap keeps RSS
flat across DataLoader workers / DDP ranks (pages are reclaimed by the OS; no
per-element Python refcounts), so 7.9M-utterance caches don't balloon memory.

Cache layout (see precompute_dac_codes.py):
    manifest.json   {audio_id -> [shard_idx, row]} + shards[] + K + dtype
    dac_{shard:05d}.bin   uint16, time-major (T, K) per utterance concatenated
    dac_{shard:05d}.idx   uint64 frame offsets, length n_utts + 1
Utterance at (shard, row) spans frames [off[row], off[row+1]); flat slice is
    bin[K*off[row] : K*off[row+1]].reshape(-1, K)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class _CodeShard:
    """Lazy mmap of one (bin, idx) pair for the DAC cache."""

    __slots__ = ("bin_path", "idx_path", "K", "_codes", "_offsets")

    def __init__(self, bin_path: Path, idx_path: Path, n_codebooks: int):
        self.bin_path = bin_path
        self.idx_path = idx_path
        self.K = n_codebooks
        self._codes: np.memmap | None = None
        self._offsets: np.ndarray | None = None

    def _ensure(self) -> None:
        if self._codes is None:
            self._codes = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
            self._offsets = np.fromfile(self.idx_path, dtype=np.uint64)

    def get(self, row: int) -> np.ndarray:
        """Return (T, K) uint16 view-copy for one utterance."""
        self._ensure()
        start = int(self._offsets[row])
        end = int(self._offsets[row + 1])
        flat = self._codes[self.K * start : self.K * end]
        # .copy() detaches from the mmap so torch.from_numpy owns writable memory
        return np.asarray(flat, dtype=np.uint16).reshape(end - start, self.K).copy()


class CodeStore:
    """Random-access reader over a sharded DAC code cache.

    Args:
        cache_dir:  directory containing manifest.json + dac_*.{bin,idx}

    Usage:
        store = CodeStore("/ocean/.../tts_cache/dac")
        codes = store["some_audio_id"]      # (T, K) long tensor
        store.has("some_audio_id")          # bool
        store.frame_count("some_audio_id")  # T without loading the codes
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"No manifest.json in {self.cache_dir}. Run precompute_dac_codes.py "
                f"with --merge after the shard passes complete.")
        man = json.loads(manifest_path.read_text())

        self.K: int = man["n_codebooks"]
        self.cardinality: int = man["cardinality"]
        self.frame_rate: float = man["frame_rate"]
        self.sample_rate: int = man["sample_rate"]
        self.codec: str = man["codec"]
        # audio_id -> [shard_idx, row]
        self._index: dict[str, list[int]] = man["index"]

        # Build shard lookup, and capture per-shard frame offsets lazily so we
        # can answer frame_count() without mmapping the bin.
        self._shards: dict[int, _CodeShard] = {}
        self._idx_paths: dict[int, Path] = {}
        for s in man["shards"]:
            si = s["shard_idx"]
            self._shards[si] = _CodeShard(
                self.cache_dir / s["bin_path"],
                self.cache_dir / s["idx_path"],
                self.K,
            )
            self._idx_paths[si] = self.cache_dir / s["idx_path"]
        self._offset_cache: dict[int, np.ndarray] = {}

    # ---- membership / metadata ----

    def __contains__(self, audio_id: str) -> bool:
        return audio_id in self._index

    def has(self, audio_id: str) -> bool:
        return audio_id in self._index

    def audio_ids(self) -> list[str]:
        return list(self._index.keys())

    def __len__(self) -> int:
        return len(self._index)

    def _offsets(self, shard_idx: int) -> np.ndarray:
        if shard_idx not in self._offset_cache:
            self._offset_cache[shard_idx] = np.fromfile(
                self._idx_paths[shard_idx], dtype=np.uint64)
        return self._offset_cache[shard_idx]

    def frame_count(self, audio_id: str) -> int:
        """T (number of DAC frames) for an utterance, without loading codes.

        Useful for the duration-bucket sampler: frame count is the true
        sequence-length cost on the temporal transformer, more accurate than
        the raw audio duration after the codec's fixed hop.
        """
        shard_idx, row = self._index[audio_id]
        off = self._offsets(shard_idx)
        return int(off[row + 1] - off[row])

    # ---- code access ----

    def get_np(self, audio_id: str) -> np.ndarray:
        """(T, K) uint16 numpy array for one utterance."""
        shard_idx, row = self._index[audio_id]
        return self._shards[shard_idx].get(row)

    def __getitem__(self, audio_id: str) -> torch.Tensor:
        """(T, K) long tensor for one utterance (ready for embedding lookup)."""
        arr = self.get_np(audio_id)
        # uint16 -> int64: codebook ids index nn.Embedding, which needs long.
        return torch.from_numpy(arr.astype(np.int64, copy=False))