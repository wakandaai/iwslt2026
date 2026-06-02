"""
speaker_store.py — train-time reader for the RawNet3 speaker-embedding cache
produced by scripts/precompute_spk_embeddings.py, plus sibling-speaker
sampling for zero-shot TTS conditioning.

Two responsibilities:
  1. O(1) lookup of a cached 192-d speaker vector by audio_id (mmap'd matrix).
  2. Pick a *conditioning* embedding for a target utterance that comes from a
     DIFFERENT utterance of the SAME speaker ("sibling"), so the model can't
     cheat by reading identity-correlated content from the target itself.
     Speakers with a single utterance fall back to self (weaker zero-shot
     generalization for those, unavoidable without more clips).

The speaker→utterances grouping is built from the (audio_id, speaker_id) pairs
the dataset already has; the store only needs the embedding matrix + index.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


class SpeakerStore:
    """Reader over the merged speaker-embedding cache.

    Args:
        cache_dir: directory containing spk_embeddings.npy + spk_index.json

    Usage:
        store = SpeakerStore("/ocean/.../tts_cache/spk")
        vec = store["some_audio_id"]                  # (192,) float tensor
        store.register_speakers(audio_ids, speaker_ids)   # once, from dataset
        cond_id = store.sibling("target_audio_id", rng)   # different utt, same spk
        cond = store[cond_id]
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        idx_path = self.cache_dir / "spk_index.json"
        npy_path = self.cache_dir / "spk_embeddings.npy"
        if not idx_path.is_file() or not npy_path.is_file():
            raise FileNotFoundError(
                f"Missing spk_embeddings.npy / spk_index.json in {self.cache_dir}. "
                f"Run precompute_spk_embeddings.py with --merge after the shards.")
        meta = json.loads(idx_path.read_text())
        self.dim: int = meta["dim"]
        self._row: dict[str, int] = meta["index"]          # audio_id -> row
        # mmap the matrix: flat RSS across workers, pages reclaimed by OS.
        self._mat = np.load(npy_path, mmap_mode="r")        # (N, dim) fp16
        assert self._mat.shape[1] == self.dim, "matrix/index dim mismatch"

        # speaker_id -> [audio_id, ...], filled by register_speakers().
        self._spk_to_ids: dict[str, list[str]] = {}
        # audio_id -> speaker_id, for sibling lookup.
        self._id_to_spk: dict[str, str] = {}

    # ---- membership ----

    def __contains__(self, audio_id: str) -> bool:
        return audio_id in self._row

    def has(self, audio_id: str) -> bool:
        return audio_id in self._row

    def __len__(self) -> int:
        return len(self._row)

    # ---- vector access ----

    def __getitem__(self, audio_id: str) -> torch.Tensor:
        """(dim,) float32 tensor. (Stored fp16; upcast for the projector.)"""
        row = self._row[audio_id]
        vec = np.asarray(self._mat[row], dtype=np.float32)   # copy out of mmap
        return torch.from_numpy(vec)

    # ---- speaker grouping (call once from the dataset) ----

    def register_speakers(self, audio_ids: list[str], speaker_ids: list[str]) -> None:
        """Build the speaker→utterance index used by sibling().

        Only audio_ids that are BOTH in this embedding cache are registered, so
        sibling() never returns an id without a cached vector. Call once after
        the dataset's row filtering so the grouping matches the trained set.
        """
        self._spk_to_ids = defaultdict(list)
        self._id_to_spk = {}
        n_skipped = 0
        for aid, spk in zip(audio_ids, speaker_ids):
            if aid not in self._row:
                n_skipped += 1
                continue
            self._spk_to_ids[spk].append(aid)
            self._id_to_spk[aid] = spk
        self._spk_to_ids = dict(self._spk_to_ids)
        if n_skipped:
            # Expected if the dataset retained rows whose embedding failed to
            # compute; those utterances simply have no sibling pool entry.
            import logging
            logging.getLogger(__name__).info(
                f"SpeakerStore: {n_skipped} dataset ids had no cached embedding")

    def sibling(self, audio_id: str, rng: random.Random) -> str:
        """Return a different utterance id from the same speaker, or `audio_id`
        itself when the speaker has only one (or grouping wasn't registered).

        Deterministic given `rng` — pass a per-epoch / per-worker seeded RNG so
        runs are reproducible. The choice is resampled each epoch by design:
        seeing varied references for the same target improves zero-shot
        robustness (the model learns speaker identity is a property carried by
        the reference vector, not by this specific clip).
        """
        spk = self._id_to_spk.get(audio_id)
        if spk is None:
            return audio_id                       # not registered → self
        pool = self._spk_to_ids.get(spk, ())
        if len(pool) <= 1:
            return audio_id                       # singleton speaker → self
        # Rejection sample to avoid returning the target itself.
        for _ in range(8):
            cand = pool[rng.randrange(len(pool))]
            if cand != audio_id:
                return cand
        # Pathological (all draws hit target); linear fallback.
        for cand in pool:
            if cand != audio_id:
                return cand
        return audio_id

    def conditioning_vector(self, audio_id: str, rng: random.Random) -> torch.Tensor:
        """Convenience: sibling() then lookup. (dim,) float32 tensor."""
        return self[self.sibling(audio_id, rng)]