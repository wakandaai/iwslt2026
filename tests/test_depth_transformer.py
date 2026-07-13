"""
Tests for the TTS depth decoder and SpeechAuraTTS sequence assembly.

The two load-bearing invariants (promoted from the WIP checks in the proposal):

  Causality  — the depth transformer predicts codebook k from h_t and codes
               0..k-1 ONLY. Perturbing a finer code must not move a coarser
               logit, or teacher forcing during training leaks future codes.

  Parity     — the parallel teacher-forced forward() and the sequential
               generate_frame() must compute the SAME distribution for a given
               prefix. If they diverge, training optimizes a model the decoder
               never actually runs.

Plus structural checks on CodecEmbeddings, the prompt layout, and the masking
of padded frames out of the loss.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from core.aura import LANG_MAP
from tts.models.depth_transformer import CodecEmbeddings, DepthTransformer
from tts.models.speech_aura_tts import SpeechAuraTTS

# Tiny dims for fast, deterministic tests.
K, CARD, D, DEPTH_DIM = 4, 8, 16, 24
SPK_DIM = 12


def _make_depth(seed: int = 0) -> tuple[CodecEmbeddings, DepthTransformer]:
    torch.manual_seed(seed)
    codec = CodecEmbeddings(n_codebooks=K, cardinality=CARD,
                            depth_dim=DEPTH_DIM, aura_dim=D)
    depth = DepthTransformer(
        n_codebooks=K, cardinality=CARD, aura_dim=D, depth_dim=DEPTH_DIM,
        n_layers=2, n_heads=2, codec_emb=codec, dropout=0.0,
    )
    codec.eval()
    depth.eval()
    return codec, depth


# ============================================================================

class TestCodecEmbeddings:
    def test_frame_embedding_is_sum_of_tables(self):
        codec, _ = _make_depth()
        codes = torch.tensor([[1, 2, 3, 4]])             # (1, K)
        got = codec.frame_embedding(codes)               # (1, D)
        # Reference: sum the K per-codebook embeddings, then project.
        acc = sum(codec.tables[k](codes[:, k]) for k in range(K))
        ref = codec.to_aura(acc)
        assert got.shape == (1, D)
        assert torch.allclose(got, ref, atol=1e-6)

    def test_codebook_embedding_uses_table_k(self):
        codec, _ = _make_depth()
        ids = torch.tensor([5, 6])
        for k in range(K):
            got = codec.codebook_embedding(ids, k)
            assert torch.allclose(got, codec.tables[k](ids), atol=1e-6)


# ============================================================================

class TestDepthCausality:
    """logits[:, k] depends on codes 0..k-1 only — perturbing column j leaves
    output positions 0..j untouched."""

    def test_perturbing_finer_code_leaves_coarser_logits(self):
        _, depth = _make_depth()
        B = 3
        h = torch.randn(B, D)
        codes = torch.randint(0, CARD, (B, K))
        base = depth(h, codes)                            # (B, K, CARD)

        for j in range(K):
            bumped = codes.clone()
            bumped[:, j] = (codes[:, j] + 1) % CARD
            out = depth(h, bumped)
            # Output positions 0..j must be identical (they never see column j).
            assert torch.allclose(base[:, : j + 1], out[:, : j + 1], atol=1e-6), \
                f"changing codebook {j} moved a logit at position <= {j}"

    def test_finer_code_actually_used(self):
        """Sanity: column j SHOULD influence at least the next position (j+1),
        otherwise 'causality' would pass trivially by ignoring the input."""
        _, depth = _make_depth()
        h = torch.randn(2, D)
        codes = torch.randint(0, CARD, (2, K))
        base = depth(h, codes)
        for j in range(K - 1):
            bumped = codes.clone()
            bumped[:, j] = (codes[:, j] + 3) % CARD
            out = depth(h, bumped)
            assert not torch.allclose(base[:, j + 1], out[:, j + 1], atol=1e-6), \
                f"codebook {j} had no effect on position {j + 1}"


# ============================================================================

class TestTrainInferParity:
    """Teacher-forced forward() and sequential generate_frame() must agree."""

    def test_greedy_decode_matches_teacher_forced_logits(self):
        _, depth = _make_depth(seed=1)
        B = 4
        h = torch.randn(B, D)

        emitted = depth.generate_frame(h, greedy=True).clone()    # (B, K)
        # Feed the emitted codes back as teacher input: forward at position k
        # conditions on emitted[:, :k] — the exact prefix generate_frame used.
        with torch.no_grad():
            tf_logits = depth(h, emitted)                 # (B, K, CARD)
        tf_pred = tf_logits.argmax(dim=-1)                # (B, K)
        assert torch.equal(tf_pred, emitted), \
            "AR decode diverged from the teacher-forced distribution"

    def test_generate_frame_deterministic_when_greedy(self):
        _, depth = _make_depth(seed=2)
        h = torch.randn(2, D)
        a = depth.generate_frame(h, greedy=True)
        b = depth.generate_frame(h, greedy=True)
        assert torch.equal(a, b)

    def test_eos_allowed_only_in_codebook_0(self):
        """With eos_id set, EOS may win codebook 0 but is masked everywhere else,
        so a real (non-EOS) frame can never carry EOS in a finer codebook."""
        codec, depth = _make_depth(seed=5)
        eos = CARD - 1                       # pick an id and force every head to it
        with torch.no_grad():
            for head in depth.heads:
                head.weight.zero_()
                head.bias.fill_(-10.0)
                head.bias[eos] = 10.0        # logits peak at eos for every k
        codes = depth.generate_frame(torch.randn(3, D), greedy=True, eos_id=eos)
        assert (codes[:, 0] == eos).all()    # codebook 0 may emit EOS
        assert (codes[:, 1:] != eos).all()   # finer codebooks never do


# ============================================================================
# SpeechAuraTTS — needs a stub temporal transformer (the real Aura-1B is heavy
# and its transformer internals are already exercised on the ST side).
# ============================================================================

class _StubAura(nn.Module):
    def __init__(self, hidden_size: int = D, vocab: int = 200):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = 2
        self.bos_id = 0
        self.task_tts_id = 31
        self.speech_start_id = 42
        self._embed = nn.Embedding(vocab, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self._lora_layers = None
        self.model = nn.Module()

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def get_embed_layer(self):
        return self._embed

    def forward_hidden(self, x, position_ids=None, use_cache=False, cache=None):
        return self.proj(x)


def _make_tts(seed: int = 0) -> SpeechAuraTTS:
    torch.manual_seed(seed)
    return SpeechAuraTTS(
        aura=_StubAura(), n_codebooks=K, cardinality=CARD, speaker_dim=SPK_DIM,
        depth_dim=DEPTH_DIM, depth_layers=2, depth_heads=2, freeze_llm=True,
    ).eval()


class TestTTSSequenceAssembly:
    def test_prompt_layout_and_positions(self):
        m = _make_tts()
        L, T = 3, 5
        codes = torch.randint(0, CARD, (1, T, K))
        code_lengths = torch.tensor([T])
        spk = torch.randn(1, SPK_DIM)
        text_ids = torch.tensor([[7, 8, 9]])
        text_lengths = torch.tensor([L])
        lang = "swahili"

        embeds, position_ids, prompt_lens = m._build_inputs(
            codes, code_lengths, [lang], torch.device("cpu"),
            speaker_vecs=spk, text_ids=text_ids, text_lengths=text_lengths)

        # Layout: [BOS, SPK, synth, LANG, text×L, SPEECH_START, frame×T]
        assert prompt_lens.tolist() == [5 + L]
        assert embeds.shape == (1, 5 + L + T, D)
        # Contiguous position ids over the (unpadded) sequence.
        assert torch.equal(position_ids[0], torch.arange(5 + L + T))

        emb = m.aura.get_embed_layer()
        tok = lambda i: emb(torch.tensor([i])).squeeze(0)  # noqa: E731
        assert torch.allclose(embeds[0, 0], tok(m.aura.bos_id))
        assert torch.allclose(embeds[0, 1], m.spk_proj(spk)[0])      # speaker token
        assert torch.allclose(embeds[0, 2], tok(m.aura.task_tts_id))
        assert torch.allclose(embeds[0, 3], tok(LANG_MAP[lang]))
        assert torch.allclose(embeds[0, 4:4 + L], emb(text_ids[0]))
        assert torch.allclose(embeds[0, 4 + L], tok(m.aura.speech_start_id))
        # Frame slots hold summed-codebook frame embeddings.
        frames = m.codec_emb.frame_embedding(codes[0])
        assert torch.allclose(embeds[0, 5 + L:], frames, atol=1e-6)

    def test_unknown_language_falls_back_to_eng(self):
        m = _make_tts()
        codes = torch.randint(0, CARD, (1, 2, K))
        embeds, _, _ = m._build_inputs(
            codes, torch.tensor([2]), ["klingon"], torch.device("cpu"),
            speaker_vecs=torch.randn(1, SPK_DIM),
            text_ids=torch.tensor([[7]]), text_lengths=torch.tensor([1]))
        emb = m.aura.get_embed_layer()
        assert torch.allclose(embeds[0, 3], emb(torch.tensor([LANG_MAP["eng"]])).squeeze(0))


class TestStage0Continuation:
    """conditioning=False: the unconditional [BOS, LANG, SPEECH_START, frame×T]
    prefix used to pretrain Aura + depth on speech dynamics before TTS."""

    def test_continuation_prefix_layout(self):
        m = _make_tts()
        T = 5
        codes = torch.randint(0, CARD, (1, T, K))
        embeds, position_ids, prompt_lens = m._build_inputs(
            codes, torch.tensor([T]), ["swahili"], torch.device("cpu"),
            conditioning=False)

        # Layout: [BOS, LANG, SPEECH_START, frame×T] — prompt_len = 3.
        assert prompt_lens.tolist() == [3]
        assert embeds.shape == (1, 3 + T, D)
        assert torch.equal(position_ids[0], torch.arange(3 + T))

        emb = m.aura.get_embed_layer()
        tok = lambda i: emb(torch.tensor([i])).squeeze(0)  # noqa: E731
        assert torch.allclose(embeds[0, 0], tok(m.aura.bos_id))
        assert torch.allclose(embeds[0, 1], tok(LANG_MAP["swahili"]))
        assert torch.allclose(embeds[0, 2], tok(m.aura.speech_start_id))
        frames = m.codec_emb.frame_embedding(codes[0])
        assert torch.allclose(embeds[0, 3:], frames, atol=1e-6)

    def test_continuation_forward_supervises_eos_without_text_or_speaker(self):
        m = _make_tts(seed=5).train()
        lens = [4, 2]
        T_max = max(lens)
        codes = torch.randint(0, CARD, (len(lens), T_max, K))
        seen = {}
        orig = m.depth.forward

        def spy(h_t, codes_BK):
            seen["N"] = h_t.size(0)
            seen["eos_rows"] = int((codes_BK == m.eos_id).all(dim=1).sum())
            return orig(h_t, codes_BK)

        m.depth.forward = spy
        out = m(codes, torch.tensor(lens), ["swahili", "swahili"],
                conditioning=False)
        m.depth.forward = orig

        assert torch.isfinite(out["loss"])
        # Same EOS-extended supervision as TTS: real frames + one EOS per utt.
        assert seen["N"] == sum(lens) + len(lens)
        assert seen["eos_rows"] == len(lens)
        out["loss"].backward()
        assert m.depth.heads[0].weight.grad is not None


class TestTTSForwardLoss:
    def _batch(self, code_lengths):
        B = len(code_lengths)
        T_max = int(max(code_lengths))
        codes = torch.randint(0, CARD, (B, T_max, K))
        return {
            "codes": codes,
            "code_lengths": torch.tensor(code_lengths),
            "speaker_vecs": torch.randn(B, SPK_DIM),
            "text_ids": torch.randint(4, 30, (B, 3)),
            "text_lengths": torch.tensor([3] * B),
            "languages": ["swahili"] * B,
        }

    def test_loss_finite_and_backprops_to_heads(self):
        m = _make_tts(seed=3).train()
        batch = self._batch([5, 3, 1])
        out = m(**batch)
        assert torch.isfinite(out["loss"])
        assert torch.isfinite(out["depth_loss"])
        assert out["loss"].item() == out["depth_loss"].item()   # no stop term
        out["loss"].backward()
        assert m.depth.heads[0].weight.grad is not None
        # The EOS row of each codebook head must receive gradient (it is the only
        # target at the appended EOS frame).
        assert m.depth.heads[0].weight.grad[m.eos_id].abs().sum() > 0
        # Frozen backbone receives no gradient.
        assert all(p.grad is None for p in m.aura.parameters())

    def test_eos_is_supervised_at_the_boundary_frame(self):
        """forward() must append an all-EOS target frame at f == code_length:
        zeroing the EOS-row logits' grad contribution is detectable, and the
        valid-frame count is sum(code_lengths) + B (one EOS per utterance)."""
        m = _make_tts(seed=7).train()
        lens = [4, 2]
        batch = self._batch(lens)
        # Spy on how many frames the depth decoder is asked to score.
        seen = {}
        orig = m.depth.forward

        def spy(h_t, codes_BK):
            seen["N"] = h_t.size(0)
            seen["eos_rows"] = int((codes_BK == m.eos_id).all(dim=1).sum())
            return orig(h_t, codes_BK)

        m.depth.forward = spy
        m(**batch)
        assert seen["N"] == sum(lens) + len(lens)      # real frames + 1 EOS each
        assert seen["eos_rows"] == len(lens)           # exactly one EOS frame/utt

    def test_generate_stops_on_eos_and_drops_eos_frame(self):
        m = _make_tts(seed=8)
        # Force EOS in codebook 0 on the 4th frame; the first three are real.
        calls = {"n": 0}

        def fake_frame(h_t, greedy=True, temperature=1.0,
                       top_k=None, top_p=None, eos_id=None):
            calls["n"] += 1
            f = torch.zeros(h_t.size(0), K, dtype=torch.long)
            if calls["n"] == 4:
                f[:, 0] = m.eos_id          # EOS fires on the 4th step
            else:
                f[:, 0] = 1                 # arbitrary real code
            return f

        m.depth.generate_frame = fake_frame
        codes = m.generate(
            speaker_vec=torch.randn(SPK_DIM), text_ids=torch.tensor([7, 8]),
            language="swahili", max_frames=50, greedy=True)
        assert codes.shape == (3, K)         # 3 real frames, EOS frame dropped
        assert (codes != m.eos_id).all()     # no EOS id ever reaches the decoder

    def test_padded_frames_do_not_affect_loss(self):
        m = _make_tts(seed=4)
        torch.manual_seed(10)
        batch = self._batch([2, 3])     # sample 0 has 1 padded frame (index 2)
        with torch.no_grad():
            loss_a = m(**batch)["loss"].item()
            # Corrupt ONLY the padded region of sample 0; loss must not move.
            batch["codes"][0, 2, :] = (batch["codes"][0, 2, :] + 1) % CARD
            loss_b = m(**batch)["loss"].item()
        assert loss_a == pytest.approx(loss_b, abs=1e-6)
