"""
Smoke tests for the Stage 1 omniASR_CTC_1B training path (aura-asr-v1).

Mirrors test_forward.py's style (CPU-only, synthetic data, no real weights)
but covers the NEW v1-specific plumbing: RawAudioDataset, CTCRawAudioCollator,
the DDP-aware/weighted samplers, and the config->constructor wiring for
OmniASREncoder. The real OmniASREncoder itself needs the 3.7GB checkpoint +
fairseq2 (only importable under the isolated omniasr_extract env), so it is
never instantiated here — build_omniasr_encoder_from_config is tested by
monkeypatching OmniASREncoder.__init__ to record what it was called with.

Run with: pytest tests/ -v
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from st.data.dataset import RawAudioDataset, load_index_csv
from st.data.collator import CTCRawAudioCollator
from st.data.sampler import (
    DurationBucketSampler,
    WeightedLanguageSampler,
    WeightedPartitionSampler,
)


# ============================================================================
# Synthetic audio + CSV index fixture
# ============================================================================

def _write_wav(path: Path, duration_s: float, sample_rate: int = 16000) -> None:
    import soundfile as sf
    n = int(duration_s * sample_rate)
    data = (np.random.randn(n) * 0.01).astype("float32")
    sf.write(str(path), data, sample_rate)


@pytest.fixture
def synthetic_index(tmp_path):
    """3 tiny synthetic wav files + a matching ASR index CSV (2 train, 1 dev)."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    rows = [
        {"audio_id": "a1", "path": str(audio_dir / "a1.wav"), "transcript": "hello world",
         "language": "igbo", "split": "train", "source": "synth", "speaker_id": "s1",
         "sample_rate": "16000", "duration": "1.0"},
        {"audio_id": "a2", "path": str(audio_dir / "a2.wav"), "transcript": "bonjour",
         "language": "french", "split": "train", "source": "synth", "speaker_id": "s2",
         "sample_rate": "16000", "duration": "1.5"},
        {"audio_id": "a3", "path": str(audio_dir / "a3.wav"), "transcript": "sawubona",
         "language": "zulu", "split": "dev", "source": "synth", "speaker_id": "s3",
         "sample_rate": "16000", "duration": "0.8"},
    ]
    for r in rows:
        _write_wav(Path(r["path"]), float(r["duration"]))

    csv_path = tmp_path / "index.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


# ============================================================================
# RawAudioDataset
# ============================================================================

class TestRawAudioDataset:
    def test_load_index_csv_filters_by_split(self, synthetic_index):
        entries = load_index_csv(
            synthetic_index, split="train", languages=None,
            sources=None, min_duration=0.1, max_duration=10.0,
        )
        assert len(entries) == 2
        assert all(e["split"] == "train" for e in entries)

    def test_dataset_len_and_durations(self, synthetic_index):
        ds = RawAudioDataset(index_path=synthetic_index, split="train")
        assert len(ds) == 2
        assert ds.durations == pytest.approx([1.0, 1.5])

    def test_getitem_shapes(self, synthetic_index):
        ds = RawAudioDataset(index_path=synthetic_index, split="train")
        item = ds[0]
        assert item["waveform"].ndim == 1
        assert item["waveform_len"] == item["waveform"].size(0)
        assert item["text"] == "hello world"
        assert item["language"] == "igbo"

    def test_max_duration_truncates_mismatched_metadata(self, tmp_path):
        # max_duration is used both as a CSV pre-filter (on the *stated*
        # duration column) AND as a waveform safety cap at load time — so
        # to exercise the truncation path we need a row whose stated
        # duration passes the filter but whose real audio is longer (the
        # exact "stale duration metadata" mismatch found in the real index
        # earlier this project — see guide.md).
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        wav_path = audio_dir / "mismatched.wav"
        _write_wav(wav_path, duration_s=1.0)  # real audio: 1.0s

        row = {"audio_id": "m1", "path": str(wav_path), "transcript": "hi",
               "language": "igbo", "split": "train", "source": "synth",
               "speaker_id": "s1", "sample_rate": "16000", "duration": "0.3"}
        csv_path = tmp_path / "index.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

        ds = RawAudioDataset(index_path=csv_path, split="train", max_duration=0.5)
        assert len(ds) == 1  # passes the CSV filter (stated duration 0.3 < 0.5)
        item = ds[0]
        assert item["waveform_len"] <= int(0.5 * 16000)  # real 1.0s audio gets capped

    def test_language_filter(self, synthetic_index):
        ds = RawAudioDataset(index_path=synthetic_index, split="train", languages=["igbo"])
        assert len(ds) == 1
        assert ds[0]["language"] == "igbo"

    def test_dev_split(self, synthetic_index):
        ds = RawAudioDataset(index_path=synthetic_index, split="dev")
        assert len(ds) == 1
        assert ds[0]["language"] == "zulu"


# ============================================================================
# CTCRawAudioCollator
# ============================================================================

class _FakeSPTokenizer:
    """Stand-in for sentencepiece.SentencePieceProcessor — encodes each
    character as an int id, close enough to the real int-id contract for
    shape/padding/drop-too-long tests without needing the real model."""

    def encode(self, text, out_type=int):
        return [ord(c) % 100 + 1 for c in text]

    def decode(self, ids):
        return "".join(chr((i - 1) % 100 + 32) for i in ids)


class TestCTCRawAudioCollator:
    @pytest.fixture
    def collator(self):
        return CTCRawAudioCollator(sp_tokenizer=_FakeSPTokenizer(), max_target_tokens=50)

    @pytest.fixture
    def batch(self):
        return [
            {"waveform": torch.randn(1600), "waveform_len": 1600,
             "text": "hello", "language": "igbo"},
            {"waveform": torch.randn(2400), "waveform_len": 2400,
             "text": "hi", "language": "hausa"},
        ]

    def test_output_keys(self, collator, batch):
        out = collator(batch)
        for key in ("audio_features", "audio_lengths", "ctc_labels",
                    "ctc_label_lengths", "language"):
            assert key in out

    def test_padding_shapes(self, collator, batch):
        out = collator(batch)
        assert out["audio_features"].shape == (2, 2400)
        assert out["audio_lengths"].tolist() == [1600, 2400]
        assert out["ctc_labels"].size(0) == 2
        assert out["ctc_label_lengths"].tolist() == [len("hello"), len("hi")]

    def test_drops_too_long(self):
        collator = CTCRawAudioCollator(sp_tokenizer=_FakeSPTokenizer(), max_target_tokens=1)
        batch = [{"waveform": torch.randn(1600), "waveform_len": 1600,
                  "text": "this is way too long", "language": "igbo"}]
        assert collator(batch) is None

    def test_all_dropped_returns_none(self, collator):
        batch = [{"waveform": torch.randn(10), "waveform_len": 10,
                  "text": "", "language": "igbo"}]
        assert collator(batch) is None


# ============================================================================
# DurationBucketSampler — DDP-specific behavior (rank slicing, resume)
# ============================================================================

class _FakeDataset:
    def __init__(self, n=40):
        self.durations = [1.0] * n


class TestDurationBucketSamplerDDP:
    def test_rank_slices_are_disjoint_and_equal_length(self):
        ds = _FakeDataset()
        world_size = 2
        s0 = DurationBucketSampler(ds, target_duration=5.0, max_batch_size=4,
                                    rank=0, world_size=world_size, seed=7)
        s1 = DurationBucketSampler(ds, target_duration=5.0, max_batch_size=4,
                                    rank=1, world_size=world_size, seed=7)
        idx0 = {i for batch in s0 for i in batch}
        idx1 = {i for batch in s1 for i in batch}
        assert idx0.isdisjoint(idx1)
        assert len(s0) == len(s1)

    def test_set_epoch_is_deterministic_and_varies_by_epoch(self):
        ds = _FakeDataset()
        s = DurationBucketSampler(ds, target_duration=5.0, max_batch_size=4, seed=1)
        s.set_epoch(0)
        first = list(s)
        s.set_epoch(0)
        assert list(s) == first  # idempotent for the same epoch

        s.set_epoch(1)
        assert list(s) != first  # different epoch => different shuffle

    def test_skip_resumes_mid_epoch(self):
        ds = _FakeDataset()
        s = DurationBucketSampler(ds, target_duration=5.0, max_batch_size=4, seed=3)
        s.set_epoch(0)
        full = list(s)
        s.set_epoch(0)
        s.skip(2)
        assert list(s) == full[2:]


# ============================================================================
# Weighted samplers (language-temperature reweighting)
# ============================================================================

class _FakeWeightedDataset:
    """3 languages with very different amounts of data — enough to make
    temperature reweighting visibly change draw frequency."""

    def __init__(self):
        self.entries = (
            [{"language": "english", "source": "s"} for _ in range(100)]
            + [{"language": "igbo", "source": "s"} for _ in range(10)]
            + [{"language": "yoruba", "source": "s"} for _ in range(2)]
        )
        self.durations = [2.0] * len(self.entries)


class TestWeightedLanguageSampler:
    def test_weights_sum_to_one(self):
        ds = _FakeWeightedDataset()
        sampler = WeightedLanguageSampler(
            ds, beta_language=0.7, target_duration=10.0, max_batch_size=4, num_batches=500,
        )
        assert set(sampler.partition_weight.keys()) == {"english", "igbo", "yoruba"}
        assert sum(sampler.partition_weight.values()) == pytest.approx(1.0, abs=1e-6)

    def test_beta_zero_is_uniform(self):
        ds = _FakeWeightedDataset()
        sampler = WeightedLanguageSampler(ds, beta_language=0.0, num_batches=100)
        weights = list(sampler.partition_weight.values())
        assert weights == pytest.approx([weights[0]] * len(weights), abs=1e-6)

    def test_beta_one_is_proportional_to_data_size(self):
        ds = _FakeWeightedDataset()
        sampler = WeightedLanguageSampler(ds, beta_language=1.0, num_batches=100)
        assert sampler.partition_weight["english"] > sampler.partition_weight["igbo"]
        assert sampler.partition_weight["igbo"] > sampler.partition_weight["yoruba"]

    def test_lower_beta_boosts_minority_language(self):
        ds = _FakeWeightedDataset()
        proportional = WeightedLanguageSampler(ds, beta_language=1.0, num_batches=100)
        flattened = WeightedLanguageSampler(ds, beta_language=0.3, num_batches=100)
        prop_ratio = proportional.partition_weight["english"] / proportional.partition_weight["yoruba"]
        flat_ratio = flattened.partition_weight["english"] / flattened.partition_weight["yoruba"]
        assert flat_ratio < prop_ratio

    def test_ddp_rank_slicing_equal_lengths(self):
        ds = _FakeWeightedDataset()
        world_size = 2
        samplers = [
            WeightedLanguageSampler(
                ds, beta_language=0.7, num_batches=20, rank=r, world_size=world_size, seed=42,
            )
            for r in range(world_size)
        ]
        assert len(samplers[0]) == len(samplers[1]) == 20 // world_size

    def test_same_seed_same_rank_reproducible(self):
        ds = _FakeWeightedDataset()
        s1 = WeightedLanguageSampler(ds, beta_language=0.7, num_batches=20, seed=99)
        s2 = WeightedLanguageSampler(ds, beta_language=0.7, num_batches=20, seed=99)
        assert list(s1) == list(s2)


class TestWeightedPartitionSampler:
    def test_two_level_weights_sum_to_one(self):
        ds = _FakeWeightedDataset()
        sampler = WeightedPartitionSampler(
            ds, beta_corpus=0.5, beta_language=0.5, num_batches=100,
        )
        assert sum(sampler.partition_weight.values()) == pytest.approx(1.0, abs=1e-6)


# ============================================================================
# build_omniasr_encoder_from_config — wiring only, no real checkpoint needed
# ============================================================================

class TestBuildOmniASREncoderFromConfig:
    def test_passes_through_config_keys(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "st.models.omniasr_encoder.OmniASREncoder.__init__",
            lambda self, **kw: captured.update(kw),
        )
        from st.models.omniasr_encoder import build_omniasr_encoder_from_config

        build_omniasr_encoder_from_config({
            "checkpoint": "/fake/path.pt",
            "dropout_p": 0.2,
            "freeze_ctc_head": False,
            "gradient_checkpointing_every_n": 4,
        })

        assert captured["checkpoint_path"] == "/fake/path.pt"
        assert captured["dropout_p"] == 0.2
        assert captured["freeze_ctc_head"] is False
        assert captured["gradient_checkpointing_every_n"] == 4

    def test_defaults_applied(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "st.models.omniasr_encoder.OmniASREncoder.__init__",
            lambda self, **kw: captured.update(kw),
        )
        from st.models.omniasr_encoder import build_omniasr_encoder_from_config

        build_omniasr_encoder_from_config({"checkpoint": "/fake/path.pt"})

        assert captured["dropout_p"] == 0.1
        assert captured["freeze_ctc_head"] is True
        assert captured["gradient_checkpointing_every_n"] == 0


# ============================================================================
# Shipped Stage 1 omniASR configs — catches accidental YAML/path breakage
# ============================================================================

EXPERIMENT_DIR = Path(__file__).parent.parent / "configs" / "experiment"


def _stage1_omniasr_configs() -> list[Path]:
    paths = set(EXPERIMENT_DIR.glob("stage1_omniasr_ctc*.yaml"))
    paths |= set((EXPERIMENT_DIR / "stage1").glob("*.yaml"))
    paths |= set((EXPERIMENT_DIR / "stage1" / "smoke").glob("*.yaml"))
    return sorted(paths)


class TestStage1OmniASRConfigs:
    @pytest.mark.parametrize("config_path", _stage1_omniasr_configs(), ids=lambda p: p.name)
    def test_config_has_required_sections(self, config_path):
        import yaml
        cfg = yaml.safe_load(config_path.read_text())
        assert {"encoder", "data", "training"} <= cfg.keys()
        assert "checkpoint" in cfg["encoder"]
        assert {"train_index", "sp_tokenizer_path"} <= cfg["data"].keys()
        assert {"output_dir", "max_steps", "lr"} <= cfg["training"].keys()

    @pytest.mark.parametrize("config_path", _stage1_omniasr_configs(), ids=lambda p: p.name)
    def test_referenced_paths_exist_if_machine_has_them(self, config_path):
        import yaml
        cfg = yaml.safe_load(config_path.read_text())
        for path in (cfg["encoder"]["checkpoint"], cfg["data"]["sp_tokenizer_path"]):
            if not os.path.isabs(path):
                pytest.skip(f"{config_path.name}: relative path {path!r} — nothing to check here")
            if not os.path.exists(path):
                pytest.skip(f"{path} not available on this machine (no shared filesystem mount)")
            assert os.path.getsize(path) > 0
