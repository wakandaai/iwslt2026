"""
SpeechDataset: reads ASR/AST index CSVs and returns mel + text pairs.

Index CSV columns (ASR):
    audio_id, path, transcript, language, split, source, speaker_id,
    sample_rate, duration

Index CSV columns (AST, superset):
    ... + translation, src_language, tgt_language

The dataset is intentionally dumb — it just reads audio and returns mel
features + text strings. All tokenization happens in the collator so the
dataset stays reusable across training stages.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

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


def load_index_csv(
    path: str | Path,
    split: str | None,
    languages: list[str] | None,
    sources: list[str] | None,
    min_duration: float,
    max_duration: float,
) -> list[dict[str, str]]:
    """Read an ASR/AST index CSV, filtered by split/language/source/duration.

    Shared between SpeechDataset (raw audio + mel) and CachedFeatureDataset
    (precomputed encoder features) — CSV filtering is identical for both.
    """
    lang_set   = set(languages) if languages else None
    source_set = set(sources) if sources else None
    entries    = []
    skipped    = 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Duration filter
            dur_str = row.get("duration", "").strip()
            if not dur_str:
                skipped += 1
                continue
            dur = float(dur_str)
            if dur < min_duration or dur > max_duration:
                continue

            # Split filter
            if split is not None and row.get("split", "") != split:
                continue

            # Language filter
            if lang_set is not None:
                lang = row.get("language") or row.get("src_language") or ""
                if lang not in lang_set:
                    continue

            # Source filter
            if source_set is not None and row.get("source", "") not in source_set:
                continue

            entries.append(row)

    if skipped:
        log.warning(f"Skipped {skipped} rows with missing duration in {path}")

    return entries


class SpeechDataset(Dataset):
    """Speech dataset backed by an ASR/AST index CSV.

    Args:
        index_path:       Path to CSV index file.
        split:            Value of the 'split' column to keep (e.g. "train", "dev").
                          None = keep all rows.
        languages:        Keep only these language codes. None = keep all.
        sources:          Keep only these source names. None = keep all.
        text_column:      Column to use as the primary text target.
                          "transcript" for ASR, "translation" for AST.
        max_duration:     Drop utterances longer than this (seconds).
        min_duration:     Drop utterances shorter than this (seconds).
        sample_rate:      Resample all audio to this rate.
        lowercase:        Lowercase all text targets.
    """

    def __init__(
        self,
        index_path: str | Path,
        split: str | None = "train",
        languages: list[str] | None = None,
        sources: list[str] | None = None,
        text_column: str = "transcript",
        max_duration: float = 30.0,
        min_duration: float = 0.1,
        sample_rate: int = _DEFAULT_SR,
        lowercase: bool = False,
    ):
        self.text_column = text_column
        self.sample_rate = sample_rate
        self.lowercase   = lowercase
        self.max_frames  = int(max_duration * sample_rate)

        self.entries: list[dict[str, str]] = self._load_index(
            index_path, split, languages, sources, min_duration, max_duration
        )

        # Pre-extract durations for DurationBucketSampler (avoids probing files at runtime)
        self.durations: list[float] = [
            float(e["duration"]) if e.get("duration") else max_duration
            for e in self.entries
        ]

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=_DEFAULT_N_FFT,
            hop_length=_DEFAULT_HOP,
            n_mels=_DEFAULT_N_MELS,
        )

        # Log summary
        lang_counts: dict[str, int] = {}
        for e in self.entries:
            lang = e.get("language") or e.get("src_language") or "?"
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        total_hours = sum(self.durations) / 3600
        log.info(
            f"SpeechDataset: {len(self.entries)} examples from {index_path} "
            f"[split={split}, {total_hours:.1f}h]"
        )
        log.info(f"  Languages: {lang_counts}")

    # ------------------------------------------------------------------

    @staticmethod
    def _load_index(
        path: str | Path,
        split: str | None,
        languages: list[str] | None,
        sources: list[str] | None,
        min_duration: float,
        max_duration: float,
    ) -> list[dict[str, str]]:
        return load_index_csv(path, split, languages, sources, min_duration, max_duration)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Retry up to 10 neighbors on load failure (corrupt / missing audio)
        for offset in range(10):
            actual = (idx + offset) % len(self.entries)
            try:
                return self._load_sample(actual)
            except Exception as exc:
                if offset == 0:
                    e = self.entries[actual]
                    log.warning(
                        f"Failed to load {e.get('audio_id', '?')} "
                        f"({e.get('path', '?')}): {exc}"
                    )
        raise RuntimeError(f"Could not load any sample near index {idx}")

    def _load_sample(self, idx: int) -> dict[str, Any]:
        entry      = self.entries[idx]
        audio_path = entry.get("path") or entry.get("audio_path") or ""

        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        waveform = torch.from_numpy(data)

        # Use index sample_rate when available (avoids soundfile header re-read)
        sr_str = entry.get("sample_rate", "").strip()
        if sr_str:
            sr = int(float(sr_str))

        if sr != self.sample_rate:
            waveform = AF.resample(waveform, sr, self.sample_rate)

        if waveform.size(0) > self.max_frames:
            waveform = waveform[: self.max_frames]

        # Log-mel spectrogram
        mel = self.mel_transform(waveform)               # (80, T)
        mel = torch.clamp(mel, min=1e-10).log10()
        mel = mel.T                                       # (T, 80)

        text = entry.get(self.text_column, "")
        if self.lowercase:
            text = text.lower()

        language = entry.get("language") or entry.get("src_language") or ""

        return {
            "audio_id":  entry.get("audio_id", ""),
            "mel":       mel,
            "mel_len":   mel.size(0),
            "text":      text,
            "language":  language,
            "source":    entry.get("source", ""),
        }


class CachedFeatureDataset(Dataset):
    """Dataset backed by precomputed omniASR_CTC_1B encoder features instead
    of raw audio + mel (see scripts/extract_omniasr_features.py).

    Reads the same ASR/AST index CSV as SpeechDataset, but each item loads a
    cached {hidden_states, predicted_ids, length} bundle from cache_dir
    instead of running audio + mel extraction. Only meant for frozen-encoder
    training stages (see SpeechAura.forward_cached / encode_audio_cached).

    Args:
        index_path: Path to CSV index file (same format as SpeechDataset).
        cache_dir:  Directory of cached feature files, sharded as
                    {cache_dir}/{audio_id[:2]}/{audio_id}.pt.
        split, languages, sources, text_column, max_duration, min_duration,
        lowercase: same semantics as SpeechDataset.
    """

    def __init__(
        self,
        index_path: str | Path,
        cache_dir: str | Path,
        split: str | None = "train",
        languages: list[str] | None = None,
        sources: list[str] | None = None,
        text_column: str = "transcript",
        max_duration: float = 30.0,
        min_duration: float = 0.1,
        lowercase: bool = False,
    ):
        self.cache_dir   = Path(cache_dir)
        self.text_column = text_column
        self.lowercase   = lowercase

        entries: list[dict[str, str]] = load_index_csv(
            index_path, split, languages, sources, min_duration, max_duration
        )

        # The cache_dir may only cover a subset of `entries` — e.g. a
        # duration-capped validation extraction (scripts/extract_omniasr_features.py
        # --max-hours) rather than the full language/duration-filtered index.
        # Filter down to entries that actually have a cached feature file, via a
        # directory listing (cheap: bounded by cache size, not index size) rather
        # than one Path.exists() stat per entry (would be 2M+ Lustre stats).
        cached_ids = {p.stem for p in self.cache_dir.glob("*/*.pt")}
        before = len(entries)
        self.entries: list[dict[str, str]] = [
            e for e in entries if e.get("audio_id", "") in cached_ids
        ]
        if len(self.entries) < before:
            log.info(
                f"CachedFeatureDataset: {len(self.entries)}/{before} entries have cached "
                f"features in {cache_dir} (cache covers a subset of the filtered index)"
            )

        # Pre-extract durations for DurationBucketSampler (same contract as SpeechDataset)
        self.durations: list[float] = [
            float(e["duration"]) if e.get("duration") else max_duration
            for e in self.entries
        ]

        lang_counts: dict[str, int] = {}
        for e in self.entries:
            lang = e.get("language") or e.get("src_language") or "?"
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        total_hours = sum(self.durations) / 3600
        log.info(
            f"CachedFeatureDataset: {len(self.entries)} examples from {index_path} "
            f"[split={split}, {total_hours:.1f}h, cache_dir={cache_dir}]"
        )
        log.info(f"  Languages: {lang_counts}")

    # ------------------------------------------------------------------

    def _cache_path(self, audio_id: str) -> Path:
        return self.cache_dir / audio_id[:2] / f"{audio_id}.pt"

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Retry up to 10 neighbors on load failure (missing / corrupt cache file)
        for offset in range(10):
            actual = (idx + offset) % len(self.entries)
            try:
                return self._load_sample(actual)
            except Exception as exc:
                if offset == 0:
                    e = self.entries[actual]
                    log.warning(
                        f"Failed to load cached features for {e.get('audio_id', '?')}: {exc}"
                    )
        raise RuntimeError(f"Could not load any cached sample near index {idx}")

    def _load_sample(self, idx: int) -> dict[str, Any]:
        entry    = self.entries[idx]
        audio_id = entry.get("audio_id", "")
        cached   = torch.load(self._cache_path(audio_id), map_location="cpu", weights_only=True)

        text = entry.get(self.text_column, "")
        if self.lowercase:
            text = text.lower()

        language = entry.get("language") or entry.get("src_language") or ""

        return {
            "audio_id":      audio_id,
            "hidden_states": cached["hidden_states"],   # (T, 1280) fp16
            "predicted_ids": cached["predicted_ids"],   # (T,) int16
            "feature_len":   int(cached["length"]),
            "text":          text,
            "language":      language,
            "source":        entry.get("source", ""),
        }


_LIVE_OMNIASR_SAMPLE_RATE = 16000


class RawAudioDataset(Dataset):
    """Dataset backed by raw waveforms, for the live/unfrozen omniASR_CTC_1B
    encoder (see src/st/models/omniasr_encoder.py). Unlike SpeechDataset (which
    computes mel spectrograms for the Conformer), this preprocesses audio the
    way Meta's real omniASR pipeline expects: mono-averaged (not channel-0-only),
    resampled to 16kHz, per-utterance layer_norm-normalized — same preprocessing
    as scripts/extract_omniasr_features.py::load_and_preprocess (kept as a
    separate, small duplicate here rather than importing that standalone script,
    which is documented as never imported by src/st/*).

    Args:
        index_path: Path to CSV index file (same format as SpeechDataset).
        split, languages, sources, text_column, max_duration, min_duration,
        lowercase: same semantics as SpeechDataset.
    """

    def __init__(
        self,
        index_path: str | Path,
        split: str | None = "train",
        languages: list[str] | None = None,
        sources: list[str] | None = None,
        text_column: str = "transcript",
        max_duration: float = 30.0,
        min_duration: float = 0.1,
        lowercase: bool = False,
    ):
        self.text_column = text_column
        self.lowercase   = lowercase
        self.max_samples = int(max_duration * _LIVE_OMNIASR_SAMPLE_RATE)

        self.entries: list[dict[str, str]] = load_index_csv(
            index_path, split, languages, sources, min_duration, max_duration
        )

        self.durations: list[float] = [
            float(e["duration"]) if e.get("duration") else max_duration
            for e in self.entries
        ]

        lang_counts: dict[str, int] = {}
        for e in self.entries:
            lang = e.get("language") or e.get("src_language") or "?"
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        total_hours = sum(self.durations) / 3600
        log.info(
            f"RawAudioDataset: {len(self.entries)} examples from {index_path} "
            f"[split={split}, {total_hours:.1f}h]"
        )
        log.info(f"  Languages: {lang_counts}")

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Retry up to 10 neighbors on load failure (corrupt / missing audio)
        for offset in range(10):
            actual = (idx + offset) % len(self.entries)
            try:
                return self._load_sample(actual)
            except Exception as exc:
                if offset == 0:
                    e = self.entries[actual]
                    log.warning(
                        f"Failed to load {e.get('audio_id', '?')} "
                        f"({e.get('path', '?')}): {exc}"
                    )
        raise RuntimeError(f"Could not load any sample near index {idx}")

    def _load_sample(self, idx: int) -> dict[str, Any]:
        entry      = self.entries[idx]
        audio_path = entry.get("path") or entry.get("audio_path") or ""

        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)  # average channels to mono (matches Meta's real preprocessing)
        waveform = torch.from_numpy(data)

        sr_str = entry.get("sample_rate", "").strip()
        if sr_str:
            sr = int(float(sr_str))
        if sr != _LIVE_OMNIASR_SAMPLE_RATE:
            waveform = AF.resample(waveform, sr, _LIVE_OMNIASR_SAMPLE_RATE)

        if waveform.size(0) > self.max_samples:
            waveform = waveform[: self.max_samples]

        waveform = torch.nn.functional.layer_norm(waveform, waveform.shape)

        text = entry.get(self.text_column, "")
        if self.lowercase:
            text = text.lower()

        language = entry.get("language") or entry.get("src_language") or ""

        return {
            "audio_id":     entry.get("audio_id", ""),
            "waveform":     waveform,
            "waveform_len": waveform.size(0),
            "text":         text,
            "language":     language,
            "source":       entry.get("source", ""),
        }
