"""
TTSDataset: joins the DAC code cache (CodeStore) and the speaker-embedding
cache (SpeakerStore) by audio_id, returning one (text, codes, speaker) example
per utterance for Aura-TTS training.

Index CSV columns (same index files the ASR/ST pipeline uses):
    audio_id, path, transcript, language, split, source, speaker_id,
    sample_rate, duration

What the dataset does NOT do: read audio. The acoustic targets are the
precomputed DAC codes (CodeStore) and the conditioning is a precomputed
speaker vector (SpeakerStore) — both produced by scripts/precompute_*.py. The
index is used only for text, language, speaker grouping, and row filtering.

Bucketing key
-------------
`self.durations` holds the DAC **frame count** per utterance, not seconds. Frame
count is the true sequence-length cost on the temporal transformer (audio
seconds × codec frame rate, after the fixed hop), so DurationBucketSampler —
which buckets on `.durations` and caps batches at `target_duration` — becomes a
frame-budget sampler with no changes. Read `target_duration` as "frames per
batch" in the TTS config.

Speaker conditioning
--------------------
The conditioning vector for a target utterance is drawn from a DIFFERENT
utterance of the SAME speaker (SpeakerStore.sibling), so the model can't read
identity off the target's own codes. The sibling is resampled each epoch (seed
combines a base seed, the epoch, and the row index) — varied references improve
zero-shot robustness. Call set_epoch() at the start of each epoch.

Memory model
------------
Mirrors SpeechDataset: index columns are stored as numpy object/primitive
arrays (not a list of dicts) so RSS stays flat across DataLoader workers / DDP
ranks. The mmap'd stores add no per-element refcounts either.
"""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from tts.data.code_store import CodeStore
from tts.data.speaker_store import SpeakerStore

log = logging.getLogger(__name__)


class TTSDataset(Dataset):
    """Text + speaker → DAC codes, backed by an index CSV and the two caches.

    Args:
        index_path:    Path to the ASR/ST index CSV.
        dac_dir:       Directory of the DAC code cache (CodeStore).
        spk_dir:       Directory of the speaker-embedding cache (SpeakerStore).
        split:         Keep rows with this `split` value. None = all.
        languages:     Keep only these language codes. None = all.
        sources:       Keep only these source names. None = all.
        min_frames:    Drop utterances with fewer DAC frames than this.
        max_frames:    Drop utterances with more DAC frames than this. None = no cap.
        lowercase:     Lowercase the text.
        seed:          Base seed for per-epoch sibling resampling.
    """

    def __init__(
        self,
        index_path: str | Path,
        dac_dir: str | Path,
        spk_dir: str | Path,
        split: str | None = "train",
        languages: list[str] | None = None,
        sources: list[str] | None = None,
        min_frames: int = 1,
        max_frames: int | None = None,
        lowercase: bool = False,
        seed: int = 0,
    ):
        self.code_store    = CodeStore(dac_dir)
        self.speaker_store = SpeakerStore(spk_dir)
        self.lowercase     = lowercase
        self.seed          = seed
        self._epoch        = 0

        # Codec metadata surfaced for the model / collator / config sanity checks.
        self.n_codebooks = self.code_store.K
        self.cardinality = self.code_store.cardinality
        self.frame_rate  = self.code_store.frame_rate
        self.speaker_dim = self.speaker_store.dim

        rows = self._load_index(
            index_path, split, languages, sources, min_frames, max_frames,
        )
        n = len(rows)
        if n == 0:
            raise RuntimeError(
                f"TTSDataset: no usable rows from {index_path} "
                f"(split={split}). Check the caches cover this index.")

        # Columnar storage (shared-memory-friendly across forks).
        self._audio_ids    = np.array([r["audio_id"]    for r in rows], dtype=object)
        self._transcripts  = np.array([r["transcript"]  for r in rows], dtype=object)
        self._languages    = np.array([r["language"]    for r in rows], dtype=object)
        self._speaker_ids  = np.array([r["speaker_id"]  for r in rows], dtype=object)
        self._sources      = np.array([r["source"]      for r in rows], dtype=object)
        # Frame counts double as the sampler bucket key (see module docstring).
        self.durations: np.ndarray = np.array(
            [r["frames"] for r in rows], dtype=np.float32)

        # Register the speaker→utterance grouping so sibling() only ever returns
        # ids that survived filtering AND have a cached vector.
        self.speaker_store.register_speakers(
            self._audio_ids.tolist(), self._speaker_ids.tolist())

        del rows

        total_frames = float(self.durations.sum())
        hours = total_frames / self.frame_rate / 3600
        unique_langs, counts = np.unique(self._languages, return_counts=True)
        log.info(
            f"TTSDataset: {n} examples from {index_path} "
            f"[split={split}, ~{hours:.1f}h, K={self.n_codebooks}]")
        log.info(f"  Languages: {dict(zip(unique_langs.tolist(), counts.tolist()))}")

    # ------------------------------------------------------------------

    def _load_index(
        self,
        path: str | Path,
        split: str | None,
        languages: list[str] | None,
        sources: list[str] | None,
        min_frames: int,
        max_frames: int | None,
    ) -> list[dict[str, Any]]:
        lang_set   = set(languages) if languages else None
        source_set = set(sources) if sources else None

        rows: list[dict[str, Any]] = []
        n_missing_cache = 0
        n_no_text = 0

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split is not None and row.get("split", "") != split:
                    continue

                aid = row.get("audio_id", "")
                # Both caches must cover the utterance.
                if aid not in self.code_store or aid not in self.speaker_store:
                    n_missing_cache += 1
                    continue

                lang = row.get("language") or row.get("src_language") or ""
                if lang_set is not None and lang not in lang_set:
                    continue
                if source_set is not None and row.get("source", "") not in source_set:
                    continue

                transcript = row.get("transcript", "").strip()
                if not transcript:
                    n_no_text += 1
                    continue

                frames = self.code_store.frame_count(aid)
                if frames < min_frames:
                    continue
                if max_frames is not None and frames > max_frames:
                    continue

                rows.append({
                    "audio_id":   aid,
                    "transcript": transcript,
                    "language":   lang,
                    "speaker_id": row.get("speaker_id", "") or aid,
                    "source":     row.get("source", ""),
                    "frames":     frames,
                })

        if n_missing_cache:
            log.warning(
                f"TTSDataset: {n_missing_cache} rows skipped — absent from the "
                f"DAC and/or speaker cache.")
        if n_no_text:
            log.warning(f"TTSDataset: {n_no_text} rows skipped — empty transcript.")
        return rows

    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Reseed sibling sampling so references vary per epoch (call each epoch)."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._audio_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        audio_id = self._audio_ids[idx]

        # Deterministic per (seed, epoch, row): reproducible yet epoch-varied.
        rng = random.Random((self.seed * 1_000_003 + self._epoch) * 1_000_003 + idx)
        cond_vec = self.speaker_store.conditioning_vector(audio_id, rng)  # (spk_dim,)

        codes = self.code_store[audio_id]            # (T, K) long
        text = self._transcripts[idx]
        if self.lowercase:
            text = text.lower()

        return {
            "audio_id":     audio_id,
            "codes":        codes,
            "frame_count":  codes.size(0),
            "text":         text,
            "language":     self._languages[idx],
            "speaker_vec":  cond_vec,
            "speaker_id":   self._speaker_ids[idx],
        }
