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
                return [ord(c) % 100 + 20 for c in text[:10]]
        return _Tok()

    @pytest.fixture
    def collator(self, mock_tokenizer):
        return AuraCollator(tokenizer=mock_tokenizer, max_target_tokens=512)

    def test_asr_batch(self, collator):
        batch = [
            {"mel": torch.randn(40, 80), "mel_len": 40,
             "transcript": "hello", "translation": "",
             "src_language": "igbo", "tgt_language": "igbo",
             "task": "asr", "audio_id": "a1", "source": "x"},
            {"mel": torch.randn(30, 80), "mel_len": 30,
             "transcript": "world", "translation": "",
             "src_language": "hausa", "tgt_language": "hausa",
             "task": "asr", "audio_id": "a2", "source": "x"},
        ]
        out = collator(batch)
        assert out is not None
        assert out["audio_features"].size(0) == 2
        assert out["transcript_lengths"].tolist() == [5, 5]
        assert out["translation_lengths"].tolist() == [0, 0]
        assert out["task"] == ["asr", "asr"]

    def test_cot_batch(self, collator):
        batch = [
            {"mel": torch.randn(40, 80), "mel_len": 40,
             "transcript": "hello", "translation": "world peace",
             "src_language": "igbo", "tgt_language": "english",
             "task": "cot", "audio_id": "a1", "source": "x"},
        ]
        out = collator(batch)
        assert out is not None
        assert out["transcript_lengths"].tolist() == [5]
        assert out["translation_lengths"][0].item() > 0
        assert out["task"] == ["cot"]


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

class _MockAura:
    def __init__(self, hidden_size=128):
        import torch.nn as nn
        self.hidden_size = hidden_size
        self.bos_id = 0
        self.eos_id = 1
        self.transcript_start_id = 12
        self.translate_start_id = 15
        self.task_asr_id = 13
        self.task_cot_id = 14
        self._embed = nn.Embedding(300, hidden_size)
        self._lora_layers = None

    def get_embed_layer(self):
        return self._embed


class _MockSpeechAura:
    def __init__(self):
        from st.models.speech_aura import SpeechAura
        self._build_inputs = SpeechAura._build_inputs.__get__(self, _MockSpeechAura)
        self.aura = _MockAura()


class TestSequenceAssembly:
    """Hard assertions on label positions — catches teacher-forcing bugs as regressions."""

    def test_asr_label_alignment(self):
        model = _MockSpeechAura()
        N, L_t, D = 5, 4, 128
        embeds, labels, _ = model._build_inputs(
            torch.randn(1, N, D), torch.tensor([N]),
            torch.tensor([[100, 101, 102, 103]]), torch.tensor([L_t]),
            torch.zeros(1, 1, dtype=torch.long), torch.tensor([0]),
            src_languages=["igbo"], tgt_languages=["igbo"], tasks=["asr"],
            device=torch.device("cpu"),
        )
        # Layout: [BOS, audio×N, <|transcribe|>, SRC_LANG, transcript]
        # length = 3 + N + L_t = 12; prompt_len = 2 + N = 7
        # lab[7..10] = transcript; lab[11] = EOS
        assert labels.shape == (1, 12)
        expected = torch.full((12,), -100, dtype=torch.long)
        expected[7:11] = torch.tensor([100, 101, 102, 103])
        expected[11] = 1  # eos
        assert torch.equal(labels[0], expected)

    def test_cot_label_alignment(self):
        model = _MockSpeechAura()
        N, L_t, L_r, D = 5, 3, 2, 128
        embeds, labels, _ = model._build_inputs(
            torch.randn(1, N, D), torch.tensor([N]),
            torch.tensor([[100, 101, 102]]), torch.tensor([L_t]),
            torch.tensor([[200, 201]]), torch.tensor([L_r]),
            src_languages=["igbo"], tgt_languages=["english"], tasks=["cot"],
            device=torch.device("cpu"),
        )
        # Layout: [BOS, audio×N, <|transcribe|>, SRC_LANG, transcript,
        #         <|translate|>, TGT_LANG, translation]
        # length = 5 + N + L_t + L_r = 15; prompt_len = 2 + N = 7
        assert labels.shape == (1, 15)
        expected = torch.full((15,), -100, dtype=torch.long)
        expected[7:10]  = torch.tensor([100, 101, 102])  # transcript at prompt_len..prompt_len+L_t
        expected[10]    = 15                               # TRANSLATE_START at prompt_len+L_t
        # Position 11 is TGT_LANG: english = LANG_MAP["english"] = 8
        expected[11]    = 8                                # TGT_LANG
        expected[12:14] = torch.tensor([200, 201])         # translation
        expected[14]    = 1                                # eos
        assert torch.equal(labels[0], expected)

    def test_cot_predicting_positions(self):
        """Embeddings at supervised positions must be the previous token's embedding."""
        model = _MockSpeechAura()
        N, L_t, L_r, D = 5, 3, 2, 128
        embeds, labels, _ = model._build_inputs(
            torch.randn(1, N, D), torch.tensor([N]),
            torch.tensor([[100, 101, 102]]), torch.tensor([L_t]),
            torch.tensor([[200, 201]]), torch.tensor([L_r]),
            src_languages=["igbo"], tgt_languages=["english"], tasks=["cot"],
            device=torch.device("cpu"),
        )
        # prompt_len = 7, so transcript spans 7..9 and TLS sits at position 10.
        # Position 10: embedding is t3=102 (last transcript token), label is TLS=15.
        t3_emb = model.aura._embed(torch.tensor([102])).squeeze(0)
        assert torch.allclose(embeds[0, 10], t3_emb)
        assert labels[0, 10].item() == 15
        # Position 11: embedding is TLS, label is TGT_LANG.
        tls_emb = model.aura._embed(torch.tensor([15])).squeeze(0)
        assert torch.allclose(embeds[0, 11], tls_emb)
        assert labels[0, 11].item() == 8   # english
        # Position 12: embedding is TGT_LANG, label is r1=200.
        tgt_emb = model.aura._embed(torch.tensor([8])).squeeze(0)
        assert torch.allclose(embeds[0, 12], tgt_emb)
        assert labels[0, 12].item() == 200

    def test_st_label_alignment(self):
        """ST layout: [BOS, TGT_LANG, audio×N, <|translate|>, translation]"""
        model = _MockSpeechAura()
        N, L_r, D = 5, 3, 128
        embeds, labels, _ = model._build_inputs(
            torch.randn(1, N, D), torch.tensor([N]),
            torch.zeros(1, 1, dtype=torch.long), torch.tensor([0]),  # no transcript used
            torch.tensor([[200, 201, 202]]), torch.tensor([L_r]),
            src_languages=["english"],
            tgt_languages=["swahili"],
            tasks=["st"],
            device=torch.device("cpu"),
        )
        # length = 3 + N + L_r = 11
        # positions: 0=BOS, 1=TGT_LANG, 2..6=audio, 7=<|translate|>, 8..10=translation
        # translate_pos = 2 + N = 7
        # lab[7..9] = translation; lab[10] = EOS
        assert labels.shape == (1, 11)
        expected = torch.full((11,), -100, dtype=torch.long)
        expected[7:10] = torch.tensor([200, 201, 202])
        expected[10] = 1  # eos
        assert torch.equal(labels[0], expected)

    def test_st_predicting_positions(self):
        """The <|translate|> embedding at position 2+N must predict the first
        translation token."""
        model = _MockSpeechAura()
        N, L_r, D = 5, 3, 128
        embeds, labels, _ = model._build_inputs(
            torch.randn(1, N, D), torch.tensor([N]),
            torch.zeros(1, 1, dtype=torch.long), torch.tensor([0]),
            torch.tensor([[200, 201, 202]]), torch.tensor([L_r]),
            src_languages=["english"],
            tgt_languages=["swahili"],
            tasks=["st"],
            device=torch.device("cpu"),
        )
        # Position 7: embedding is <|translate|>, label is r1=200
        translate_emb = model.aura._embed(torch.tensor([15])).squeeze(0)
        assert torch.allclose(embeds[0, 7], translate_emb)
        assert labels[0, 7].item() == 200
        # Position 1: embedding is TGT_LANG (swahili — needs LANG_MAP, mock uses english=8)
        # Skip detailed check — mock doesn't import LANG_MAP fully.