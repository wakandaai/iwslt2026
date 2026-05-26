"""
SpeechDataset: reads ASR/AST index CSVs and returns mel + text pairs.

Index CSV columns (ASR):
    audio_id, path, transcript, language, split, source, speaker_id,
    sample_rate, duration

Index CSV columns (AST, superset):
    ... + translation, src_language, tgt_language

Memory model
------------
With multi-worker DataLoader + DDP, naive `self.entries = [dict, dict, ...]`
balloons RSS: each worker process forks a copy, and CPython refcount writes
defeat copy-on-write. At 7.4M rows × 40 procs that's ~120GB just for index
state.

We store columns as numpy object arrays (str dtype=object) and primitive
numpy arrays (durations, sample_rates). Numpy buffers stay in shared memory
across forks because they don't carry per-element refcounts.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import soundfile as sf
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

# Default mel transform settings (25ms window, 10ms hop, 80 bins @ 16kHz)
_DEFAULT_SR      = 16000
_DEFAULT_N_MELS  = 80
_DEFAULT_N_FFT   = 400
_DEFAULT_HOP     = 160


class SpeechDataset(Dataset):
    """Speech dataset backed by an ASR/AST index CSV.

    Args:
        index_path:       Path to CSV index file.
        split:            Value of the 'split' column to keep (e.g. "train", "dev").
                          None = keep all rows.
        languages:        Keep only these language codes. None = keep all.
        sources:          Keep only these source names. None = keep all.
        max_duration:     Drop utterances longer than this (seconds).
        min_duration:     Drop utterances shorter than this (seconds).
        sample_rate:      Resample all audio to this rate.
        lowercase:        Lowercase all text targets.
    """

    def __init__(
        self,
        index_path: str | Path,
        split: str | None = "train",
        task: str = "asr",
        languages: list[str] | None = None,
        src_languages: list[str] | None = None,
        tgt_languages: list[str] | None = None,
        sources: list[str] | None = None,
        max_duration: float = 30.0,
        min_duration: float = 0.1,
        sample_rate: int = _DEFAULT_SR,
        lowercase: bool = False,
    ):
        if task not in ("asr", "cot", "st"):
            raise ValueError(f"task must be 'asr', 'cot', or 'st', got '{task}'")

        self.task = task
        self.sample_rate = sample_rate
        self.lowercase   = lowercase
        self.max_frames  = int(max_duration * sample_rate)

        effective_src = src_languages if src_languages is not None else languages

        # Load CSV → list of dicts (transient). Immediately convert to columnar
        # numpy arrays and drop the list of dicts before workers fork.
        entries = self._load_index(
            index_path, split, task, effective_src, tgt_languages,
            sources, min_duration, max_duration,
        )

        n = len(entries)
        # Columnar storage. Object arrays for strings (still Python str objects,
        # but stored contiguously in one numpy buffer rather than 7M dict slots).
        self._paths        = np.array(
            [e.get("path") or e.get("audio_path") or "" for e in entries],
            dtype=object,
        )
        self._audio_ids    = np.array(
            [e.get("audio_id", "") for e in entries],
            dtype=object,
        )
        self._transcripts  = np.array(
            [e.get("transcript", "") for e in entries],
            dtype=object,
        )
        self._sources      = np.array(
            [e.get("source", "") for e in entries],
            dtype=object,
        )
        self._src_languages = np.array(
            [e.get("language") or e.get("src_language") or "" for e in entries],
            dtype=object,
        )

        # Only needed in CoT mode — skip allocating for ASR-only runs.
        if task in ("cot", "st"):
            self._translations  = np.array(
                [e.get("translation", "") for e in entries],
                dtype=object,
            )
            self._tgt_languages = np.array(
                [e.get("tgt_language", "english") for e in entries],
                dtype=object,
            )
        else:
            self._translations  = None
            self._tgt_languages = None

        # Primitive numeric arrays — these are truly shared-memory across forks.
        self._sample_rates = np.array(
            [int(float(e["sample_rate"])) if e.get("sample_rate", "").strip() else 0
             for e in entries],
            dtype=np.int32,
        )
        self.durations: np.ndarray = np.array(
            [float(e["duration"]) if e.get("duration") else max_duration
             for e in entries],
            dtype=np.float32,
        )

        # Compatibility: code elsewhere (e.g. build_val_generate_indices) treats
        # entries as iterable to read `language`/`src_language` per row. Provide
        # a lazy view that returns a minimal dict per index instead of holding
        # the original list.
        self.entries = _EntriesView(self)

        # Drop the heavy list of dicts BEFORE any workers spawn.
        del entries

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=_DEFAULT_N_FFT,
            hop_length=_DEFAULT_HOP,
            n_mels=_DEFAULT_N_MELS,
        )

        # Log summary
        unique_langs, counts = np.unique(self._src_languages, return_counts=True)
        lang_counts = dict(zip(unique_langs.tolist(), counts.tolist()))

        total_hours = float(self.durations.sum()) / 3600
        log.info(
            f"SpeechDataset: {n} examples from {index_path} "
            f"[split={split}, {total_hours:.1f}h]"
        )
        log.info(f"  Languages: {lang_counts}")

    # ------------------------------------------------------------------

    @staticmethod
    def _load_index(
        path: str | Path,
        split: str | None,
        task: str,
        src_languages: list[str] | None,
        tgt_languages: list[str] | None,
        sources: list[str] | None,
        min_duration: float,
        max_duration: float,
    ) -> list[dict[str, str]]:
        src_set    = set(src_languages) if src_languages else None
        tgt_set    = set(tgt_languages) if tgt_languages else None
        source_set = set(sources) if sources else None

        entries = []
        skipped_no_dur = 0
        skipped_no_translation = 0

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Duration filter
                dur_str = row.get("duration", "").strip()
                if not dur_str:
                    skipped_no_dur += 1
                    continue
                dur = float(dur_str)
                if dur < min_duration or dur > max_duration:
                    continue

                # Split filter
                if split is not None and row.get("split", "") != split:
                    continue

                # Source-language filter
                if src_set is not None:
                    src_lang = row.get("language") or row.get("src_language") or ""
                    if src_lang not in src_set:
                        continue

                # Target-language filter (CoT only — ASR has no tgt_language column)
                if tgt_set is not None and task == "cot":
                    if row.get("tgt_language", "") not in tgt_set:
                        continue

                # Source filter
                if source_set is not None and row.get("source", "") not in source_set:
                    continue

                # CoT requires both transcript and translation populated
                # Also enforced for ST to validate encoder
                if task in ("cot", "st"):
                    transcript  = row.get("transcript",  "").strip()
                    translation = row.get("translation", "").strip()
                    if not transcript or not translation:
                        skipped_no_translation += 1
                        continue

                entries.append(row)

        if skipped_no_dur:
            log.warning(f"Skipped {skipped_no_dur} rows with missing duration in {path}")
        if skipped_no_translation:
            log.warning(
                f"Skipped {skipped_no_translation} rows missing transcript/translation "
                f"(required for task=cot)"
            )
        return entries

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Retry up to 10 neighbors on load failure (corrupt / missing audio)
        for offset in range(10):
            actual = (idx + offset) % len(self)
            try:
                return self._load_sample(actual)
            except Exception as exc:
                if offset == 0:
                    log.warning(
                        f"Failed to load {self._audio_ids[actual]} "
                        f"({self._paths[actual]}): {exc}"
                    )
        raise RuntimeError(f"Could not load any sample near index {idx}")

    def _load_sample(self, idx: int) -> dict[str, Any]:
        audio_path = self._paths[idx]

        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        waveform = torch.from_numpy(data)

        # Use index sample_rate when available (avoids soundfile header re-read)
        idx_sr = int(self._sample_rates[idx])
        if idx_sr > 0:
            sr = idx_sr

        if sr != self.sample_rate:
            waveform = AF.resample(waveform, sr, self.sample_rate)

        if waveform.size(0) > self.max_frames:
            waveform = waveform[: self.max_frames]

        # Log-mel spectrogram
        mel = self.mel_transform(waveform)               # (80, T)
        mel = torch.clamp(mel, min=1e-10).log10()
        mel = mel.T                                       # (T, 80)

        src_language = self._src_languages[idx]

        transcript = self._transcripts[idx]
        if self.lowercase:
            transcript = transcript.lower()

        if self.task == "asr":
            translation  = ""
            tgt_language = src_language
        else:  # cot or st
            translation = self._translations[idx]
            if self.lowercase:
                translation = translation.lower()
            tgt_language = self._tgt_languages[idx]

        return {
            "audio_id":     self._audio_ids[idx],
            "mel":          mel,
            "mel_len":      mel.size(0),
            "transcript":   transcript,
            "translation":  translation,
            "src_language": src_language,
            "tgt_language": tgt_language,
            "task":         self.task,
            "source":       self._sources[idx],
        }


class _EntriesView:
    """Lazy backward-compat view: dataset.entries[i] returns a minimal dict.

    Some helpers (build_val_generate_indices) iterate dataset.entries to bucket
    indices by language. We don't want to keep the original list of dicts
    around, so we synthesize per-row dicts on demand from the columnar arrays.
    """

    def __init__(self, ds: SpeechDataset):
        self._ds = ds

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict[str, str]:
        ds = self._ds
        d = {
            "audio_id":     ds._audio_ids[idx],
            "path":         ds._paths[idx],
            "transcript":   ds._transcripts[idx],
            "language":     ds._src_languages[idx],
            "src_language": ds._src_languages[idx],
            "source":       ds._sources[idx],
            "duration":     str(float(ds.durations[idx])),
            "sample_rate":  str(int(ds._sample_rates[idx])),
        }
        if ds._translations is not None:
            d["translation"]  = ds._translations[idx]
            d["tgt_language"] = ds._tgt_languages[idx]
        return d

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]