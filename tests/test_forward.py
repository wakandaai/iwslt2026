"""
Smoke tests for the SpeechAura model stack.

Tests run on CPU with a tiny synthetic model — no Aura weights needed.
Run with: pytest tests/ -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from st.models.encoder import SpeechEncoder, ConvSubsampler
from st.models.projector import MLPProjector, TransformerProjector, build_projector
from st.models.ctc_compressor import CTCCompressor, build_ctc_compressor
from st.data.collator import AuraCollator
from st.data.sampler import DurationBucketSampler


# ============================================================================
# Fixtures
# ============================================================================

ENCODER_DIM = 64
LLM_HIDDEN  = 128
VOCAB_SIZE  = 50
B, T, F     = 2, 40, 80   # batch, mel frames, mel bins


@pytest.fixture
def encoder():
    return SpeechEncoder(
        input_dim=F,
        encoder_dim=ENCODER_DIM,
        num_heads=4,
        ffn_dim=128,
        num_layers=2,
        depthwise_conv_kernel_size=7,
        dropout=0.0,
        vocab_size=VOCAB_SIZE,
    )


@pytest.fixture
def features():
    return torch.randn(B, T, F)


@pytest.fixture
def lengths():
    return torch.tensor([T, T - 4])


# ============================================================================
# Encoder tests
# ============================================================================

class TestSpeechEncoder:
    def test_forward_shape(self, encoder, features, lengths):
        out = encoder(features, lengths)
        assert "hidden_states" in out
        assert "lengths" in out
        assert "ctc_logits" in out
        T_out = out["hidden_states"].size(1)
        assert out["ctc_logits"].shape == (B, T_out, VOCAB_SIZE)

    def test_subsampled_lengths(self, encoder, features, lengths):
        out = encoder(features, lengths)
        # 4x subsampling: each length should be ~ original // 4
        for i in range(B):
            expected = ((lengths[i] - 1) // 2 + 1)
            expected = ((expected - 1) // 2 + 1)
            assert out["lengths"][i] == expected

    def test_freeze_unfreeze(self, encoder):
        encoder.freeze()
        assert all(not p.requires_grad for p in encoder.parameters())
        encoder.unfreeze()
        assert all(p.requires_grad for p in encoder.parameters())

    def test_no_ctc_head(self, features, lengths):
        enc = SpeechEncoder(
            input_dim=F, encoder_dim=ENCODER_DIM,
            num_heads=4, ffn_dim=128, num_layers=2,
            depthwise_conv_kernel_size=7, vocab_size=None,
        )
        out = enc(features, lengths)
        assert "ctc_logits" not in out


# ============================================================================
# Projector tests
# ============================================================================

class TestProjectors:
    def test_mlp_projector(self):
        proj = MLPProjector(ENCODER_DIM, LLM_HIDDEN)
        x    = torch.randn(B, 10, ENCODER_DIM)
        out  = proj(x)
        assert out.shape == (B, 10, LLM_HIDDEN)

    def test_transformer_projector(self):
        proj = TransformerProjector(ENCODER_DIM, LLM_HIDDEN, num_layers=1, num_heads=4)
        x    = torch.randn(B, 10, ENCODER_DIM)
        lens = torch.tensor([10, 8])
        out  = proj(x, lengths=lens)
        assert out.shape == (B, 10, LLM_HIDDEN)

    def test_build_projector_mlp(self):
        proj = build_projector({"type": "mlp"}, ENCODER_DIM, LLM_HIDDEN)
        assert isinstance(proj, MLPProjector)

    def test_build_projector_transformer(self):
        proj = build_projector(
            {"type": "transformer", "num_layers": 1, "num_heads": 4},
            ENCODER_DIM, LLM_HIDDEN,
        )
        assert isinstance(proj, TransformerProjector)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            build_projector({"type": "qformer"}, ENCODER_DIM, LLM_HIDDEN)


# ============================================================================
# CTC compressor tests
# ============================================================================

class TestCTCCompressor:
    @pytest.fixture
    def compressor(self):
        return CTCCompressor(strategy="avg", blank_id=0, remove_blanks=True)

    def test_output_shape(self, compressor, encoder, features, lengths):
        out        = encoder(features, lengths)
        hidden     = out["hidden_states"]
        ctc_logits = out["ctc_logits"]
        enc_lengths = out["lengths"]

        compressed, new_lengths = compressor(hidden, ctc_logits, enc_lengths)
        assert compressed.size(0) == B
        assert compressed.size(2) == ENCODER_DIM
        assert new_lengths.shape == (B,)

    def test_compressed_shorter(self, compressor, encoder, features, lengths):
        out     = encoder(features, lengths)
        _, new_lengths = compressor(
            out["hidden_states"], out["ctc_logits"], out["lengths"]
        )
        for i in range(B):
            assert new_lengths[i] <= out["lengths"][i]

    def test_weighted_strategy(self, encoder, features, lengths):
        comp = CTCCompressor(strategy="weighted", remove_blanks=False)
        out  = encoder(features, lengths)
        compressed, new_lengths = comp(
            out["hidden_states"], out["ctc_logits"], out["lengths"]
        )
        assert compressed.shape[0] == B

    def test_build_from_config(self):
        cfg  = {"enabled": True, "strategy": "softmax", "remove_blanks": True}
        comp = build_ctc_compressor(cfg)
        assert isinstance(comp, CTCCompressor)
        assert comp.strategy == "softmax"

    def test_disabled_returns_none(self):
        assert build_ctc_compressor(None) is None
        assert build_ctc_compressor({"enabled": False}) is None


# ============================================================================
# Collator tests (no Aura weights needed)
# ============================================================================

class TestAuraCollator:
    @pytest.fixture
    def mock_tokenizer(self):
        class _Tok:
            def encode(self, text, add_special_tokens=False):
                return [ord(c) % 100 + 10 for c in text[:10]]
        return _Tok()

    @pytest.fixture
    def collator(self, mock_tokenizer):
        return AuraCollator(
            tokenizer=mock_tokenizer,
            max_target_tokens=512,
        )

    @pytest.fixture
    def batch(self):
        return [
            {"mel": torch.randn(40, 80), "mel_len": 40,
             "text": "hello", "language": "igbo", "audio_id": "a1", "source": "x"},
            {"mel": torch.randn(30, 80), "mel_len": 30,
             "text": "world", "language": "hausa", "audio_id": "a2", "source": "x"},
        ]

    def test_output_keys(self, collator, batch):
        out = collator(batch)
        assert out is not None
        for key in ("audio_features", "audio_lengths", "target_ids",
                    "target_lengths", "language"):
            assert key in out

    def test_batch_size(self, collator, batch):
        out = collator(batch)
        assert out["audio_features"].size(0) == 2
        assert len(out["language"]) == 2

    def test_labels_masked(self, collator, batch):
        # collator no longer produces labels — that's done in model forward
        out = collator(batch)
        assert "labels" not in out
        assert out["target_ids"].shape[0] == 2

    def test_all_too_long_returns_none(self, mock_tokenizer):
        collator = AuraCollator(
            tokenizer=mock_tokenizer,
            max_target_tokens=1,   # impossibly short
        )
        batch = [
            {"mel": torch.randn(40, 80), "mel_len": 40,
             "text": "hi there", "language": "igbo", "audio_id": "a1", "source": "x"},
        ]
        assert collator(batch) is None


# ============================================================================
# Sampler tests
# ============================================================================

class TestDurationBucketSampler:
    @pytest.fixture
    def fake_dataset(self):
        class _DS:
            durations = [1.0, 2.0, 1.5, 3.0, 2.5, 1.2, 4.0, 0.8]
        return _DS()

    def test_length(self, fake_dataset):
        sampler = DurationBucketSampler(
            fake_dataset, target_duration=5.0, max_batch_size=10
        )
        batches = list(sampler)
        assert len(batches) == len(sampler)

    def test_all_indices_covered(self, fake_dataset):
        sampler  = DurationBucketSampler(
            fake_dataset, target_duration=100.0, max_batch_size=100
        )
        all_idx  = sorted(idx for batch in sampler for idx in batch)
        assert all_idx == list(range(len(fake_dataset.durations)))

    def test_max_batch_duration(self, fake_dataset):
        target   = 3.0
        sampler  = DurationBucketSampler(
            fake_dataset, target_duration=target, max_batch_size=100
        )
        for batch in sampler:
            total = sum(fake_dataset.durations[i] for i in batch)
            # Allow one sample to push slightly over (first sample in empty batch)
            assert total <= target + max(fake_dataset.durations)