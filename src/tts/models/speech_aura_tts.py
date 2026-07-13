"""
SpeechAuraTTS: zero-shot text-to-speech on the Aura-1B backbone.

Architecture (RQ-Transformer factoring, mirrors SpeechAura's structure):

    text + speaker  ──►  Aura temporal transformer  ──►  h_t per frame
                         (forward_hidden, no lm_head)        │
                                                             ▼
                                              DepthTransformer  ──►  K DAC codes / frame

The TEMPORAL transformer carries all long-range dependency; one position per
DAC frame (~50 Hz). The DEPTH transformer (small) autoregresses over the K RVQ
codebooks within a frame, conditioned on h_t. The ×K blow-up never enters
Aura's context window. See depth_transformer.py for the depth decomposition.

Conditioning layout (the prefix the user approved — speaker + text as a single
AR prefix, no cross-attention):

    [BOS, SPK, <|synthesize|>, LANG, text_1..text_L, SPEECH_START,
     frame_0, frame_1, ..., frame_{T-1}]

  - SPK            one token: the RawNet3 speaker vector projected to Aura dim.
  - <|synthesize|> the TTS task marker (core.aura.TASK_TTS_ID).
  - LANG           language token of the text (core.aura.LANG_MAP), so the model
                   knows the target language — cheap, matches the ST design.
  - SPEECH_START   boundary into the acoustic stream (core.aura.SPEECH_START_ID).
  - frame_t        Σ_k codebook_emb(c_t^k) projected to Aura dim (CodecEmbeddings).

    prompt_len = 5 + L          (BOS, SPK, synth, LANG, text×L, SPEECH_START)
    frame_t input sits at position prompt_len + t.

Next-token alignment: the hidden at position (prompt_len - 1 + t) predicts
frame_t (depth loss) for t in 0..T-1. The hidden one step past the last frame
(position prompt_len - 1 + T) predicts a synthetic EOS frame: every codebook
takes the dedicated EOS id (= cardinality, i.e. the "1025th" code), which is not
a real DAC code and is never sent to the decoder. Generation stops when the
coarsest codebook (k=0, predicted first) decodes to EOS. The EOS frame is a
target only — it is never fed back as a temporal input.

Modelling EOS as an in-vocab code (learned by the same depth softmax CE as every
other code) avoids the severe class imbalance of a separate binary stop head
(1 positive vs ~hundreds of negatives per utterance), which previously collapsed
to "never stop".

Loss = depth CE over frames 0..T (T = EOS frame), summed over K codebooks,
       optionally per-codebook weighted.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.aura import AuraLLM, LANG_MAP
from tts.models.depth_transformer import CodecEmbeddings, DepthTransformer

log = logging.getLogger(__name__)


class SpeechAuraTTS(nn.Module):
    """Text + speaker → DAC codes, on the frozen/finetuned Aura backbone.

    Args:
        aura:            AuraLLM wrapper (loaded + frozen/unfrozen as desired).
        n_codebooks:     K — DAC RVQ codebooks (matches CodeStore.K).
        cardinality:     per-codebook vocab (matches CodeStore.cardinality).
        speaker_dim:     dim of the cached speaker vector (RawNet3 = 192).
        depth_dim:       width of the depth transformer.
        depth_layers:    number of depth-transformer layers.
        depth_heads:     attention heads in the depth transformer.
        depth_dropout:   dropout inside the depth transformer.
        codebook_weights: optional length-K weights for the per-codebook CE
                          (codebook 0 dominates perceptual quality). None =
                          uniform.
        freeze_llm:      freeze the Aura backbone (Stage-1 TTS trains the codec
                         embeddings + depth transformer + speaker projector on
                         top of a frozen LM).
    """

    def __init__(
        self,
        aura: AuraLLM,
        n_codebooks: int,
        cardinality: int,
        speaker_dim: int = 192,
        depth_dim: int = 1024,
        depth_layers: int = 6,
        depth_heads: int = 8,
        depth_dropout: float = 0.0,
        codebook_weights: list[float] | None = None,
        freeze_llm: bool = True,
    ):
        super().__init__()
        self.aura = aura
        self.K = n_codebooks
        self.card = cardinality          # real DAC codes: ids 0..card-1
        self.eos_id = cardinality        # the "1025th" code: end-of-sequence
        self.vocab = cardinality + 1     # depth tables / heads span real + EOS
        # Architecture dims — persisted to meta.json so inference can rebuild
        # the exact model without re-reading the training config.
        self.speaker_dim  = speaker_dim
        self.depth_dim    = depth_dim
        self.depth_layers = depth_layers
        self.depth_heads  = depth_heads

        if freeze_llm:
            self.aura.freeze()
        else:
            self.aura.unfreeze()

        D = aura.hidden_size

        # Codec embeddings (shared by the temporal frame input and the depth
        # transformer's per-codebook inputs) + the depth decoder. Both span
        # self.vocab = card + 1 so the extra EOS id has an embedding row and a
        # logit in every codebook head.
        self.codec_emb = CodecEmbeddings(
            n_codebooks=n_codebooks, cardinality=self.vocab,
            depth_dim=depth_dim, aura_dim=D,
        )
        self.depth = DepthTransformer(
            n_codebooks=n_codebooks, cardinality=self.vocab, aura_dim=D,
            depth_dim=depth_dim, n_layers=depth_layers, n_heads=depth_heads,
            codec_emb=self.codec_emb, dropout=depth_dropout,
        )

        # Speaker conditioning: one prefix token. LayerNorm tames the raw
        # RawNet3 vector's scale before the projection into Aura's hidden space.
        self.spk_proj = nn.Sequential(
            nn.LayerNorm(speaker_dim),
            nn.Linear(speaker_dim, D),
        )

        # Per-codebook CE weights.
        if codebook_weights is None:
            w = torch.ones(n_codebooks)
        else:
            if len(codebook_weights) != n_codebooks:
                raise ValueError(
                    f"codebook_weights has {len(codebook_weights)} entries, "
                    f"expected n_codebooks={n_codebooks}")
            w = torch.tensor(codebook_weights, dtype=torch.float32)
        self.register_buffer("codebook_weights", w, persistent=False)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        log.info(f"SpeechAuraTTS: K={n_codebooks} card={cardinality} "
                 f"depth_dim={depth_dim} layers={depth_layers}")
        log.info(f"  {total:,} params total, {trainable:,} trainable")

    # ------------------------------------------------------------------
    # Sequence assembly
    # ------------------------------------------------------------------

    def _build_inputs(
        self,
        codes: torch.Tensor,            # (B, T_max, K) long, padded
        code_lengths: torch.Tensor,    # (B,) true frame counts T_i
        languages: list[str],
        device: torch.device,
        speaker_vecs: torch.Tensor | None = None,   # (B, speaker_dim) float
        text_ids: torch.Tensor | None = None,       # (B, L_max) long, padded
        text_lengths: torch.Tensor | None = None,  # (B,) true text lengths L_i
        conditioning: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build (inputs_embeds, position_ids, prompt_lens) for teacher forcing.

        With conditioning (Stage 1 — TTS), the temporal sequence per sample is
            [BOS, SPK, synth, LANG, text×L, SPEECH_START, frame×T]   prompt_len = 5 + L
        Without conditioning (Stage 0 — speech continuation), the prefix drops
        the speaker token, the synth task marker, and the text, leaving an
        unconditional acoustic stream
            [BOS, LANG, SPEECH_START, frame×T]                       prompt_len = 3
        so the model learns next-frame dynamics decoupled from text/speaker.
        LANG is kept (cheap per-language prior). speaker_vecs / text_ids /
        text_lengths are unused when conditioning is False.

        Frame embeddings are summed codebook embeddings projected to Aura dim.
        Sequences are right-padded with the BOS embedding (padding is causal-safe
        — it sits past every supervised position and is gathered out by the
        conditioning index in forward()).
        """
        embed_layer = self.aura.get_embed_layer()
        B = codes.size(0)

        spk_tokens = (
            self.spk_proj(speaker_vecs.to(device)) if conditioning else None)   # (B, D)

        seqs: list[torch.Tensor] = []
        prompt_lens: list[int] = []

        synth_emb = embed_layer(
            torch.tensor([self.aura.task_tts_id], dtype=torch.long, device=device))
        speech_emb = embed_layer(
            torch.tensor([self.aura.speech_start_id], dtype=torch.long, device=device))
        bos_emb = embed_layer(
            torch.tensor([self.aura.bos_id], dtype=torch.long, device=device))

        for i in range(B):
            T = int(code_lengths[i].item())

            lang_id = LANG_MAP.get(languages[i], LANG_MAP["eng"])
            lang_emb = embed_layer(
                torch.tensor([lang_id], dtype=torch.long, device=device))
            frame_emb = self.codec_emb.frame_embedding(
                codes[i, :T].to(device))                            # (T, D)

            if conditioning:
                L = int(text_lengths[i].item())
                text_emb = embed_layer(text_ids[i, :L].to(device))  # (L, D)
                seq = torch.cat([
                    bos_emb,
                    spk_tokens[i].unsqueeze(0),
                    synth_emb,
                    lang_emb,
                    text_emb,
                    speech_emb,
                    frame_emb,
                ], dim=0)
                prompt_lens.append(5 + L)   # BOS,SPK,synth,LANG,text×L,SPEECH_START
            else:
                seq = torch.cat([
                    bos_emb,
                    lang_emb,
                    speech_emb,
                    frame_emb,
                ], dim=0)
                prompt_lens.append(3)       # BOS, LANG, SPEECH_START
            seqs.append(seq)

        S_max = max(s.size(0) for s in seqs)
        pad_emb = bos_emb.squeeze(0)

        inputs_embeds = torch.stack([
            torch.cat([s, pad_emb.unsqueeze(0).expand(S_max - s.size(0), -1)], dim=0)
            if s.size(0) < S_max else s
            for s in seqs
        ])

        position_ids = torch.zeros(B, S_max, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            real = s.size(0)
            position_ids[i, :real] = torch.arange(real, device=device)
            if real < S_max:
                position_ids[i, real:] = real - 1

        prompt_lens_t = torch.tensor(prompt_lens, dtype=torch.long, device=device)
        return inputs_embeds, position_ids, prompt_lens_t

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------

    def forward(
        self,
        codes: torch.Tensor,
        code_lengths: torch.Tensor,
        languages: list[str],
        speaker_vecs: torch.Tensor | None = None,
        text_ids: torch.Tensor | None = None,
        text_lengths: torch.Tensor | None = None,
        conditioning: bool = True,
        **_unused,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forced depth-CE loss.

        conditioning=True  → Stage-1 TTS (text + speaker prefix).
        conditioning=False → Stage-0 speech continuation (unconditional acoustic
        stream); speaker_vecs / text_ids / text_lengths are ignored. The EOS
        frame, gather indexing, and depth CE are identical in both stages — only
        the prefix length (prompt_lens) differs.
        """
        device = codes.device
        B, T_max, K = codes.shape
        assert K == self.K, f"codes has K={K}, model expects {self.K}"

        inputs_embeds, position_ids, prompt_lens = self._build_inputs(
            codes, code_lengths, languages, device,
            speaker_vecs=speaker_vecs, text_ids=text_ids,
            text_lengths=text_lengths, conditioning=conditioning,
        )

        h = self.aura.forward_hidden(inputs_embeds, position_ids)   # (B, S_max, D)
        S_max = h.size(1)

        # Gather the conditioning hidden states. cond_h[:, f] is the hidden at
        # position (prompt_len - 1 + f); it predicts frame f for f < T_i, and at
        # f == T_i it predicts the synthetic EOS frame. The extra column (f up to
        # T_max) is exactly that post-last-frame hidden.
        ar = torch.arange(T_max + 1, device=device)                 # (T_max+1,)
        gather_idx = (prompt_lens - 1).unsqueeze(1) + ar.unsqueeze(0)
        gather_idx = gather_idx.clamp(max=S_max - 1)                 # (B, T_max+1)
        cond_h = h.gather(
            1, gather_idx.unsqueeze(-1).expand(-1, -1, h.size(-1))
        )                                                            # (B, T_max+1, D)

        # ---- Build EOS-extended targets: frames 0..T_i-1 are the real codes,
        # frame T_i is all-EOS, frames > T_i are padding (masked out). ----
        codes_ext = torch.cat(
            [codes, codes.new_zeros(B, 1, self.K)], dim=1)             # (B, T_max+1, K)
        codes_ext[torch.arange(B, device=device), code_lengths] = self.eos_id

        f_idx = torch.arange(T_max + 1, device=device)
        valid = f_idx.unsqueeze(0) <= code_lengths.unsqueeze(1)        # (B, T_max+1)
        h_flat = cond_h[valid]                                        # (N, D)
        codes_flat = codes_ext[valid].to(device)                      # (N, K) long
        N = h_flat.size(0)

        if N > 0:
            logits = self.depth(h_flat, codes_flat)                   # (N, K, vocab)
            ce = F.cross_entropy(
                logits.reshape(-1, self.vocab), codes_flat.reshape(-1),
                reduction="none",
            ).view(N, self.K)                                         # (N, K)
            w = self.codebook_weights.to(ce.dtype)
            depth_loss = (ce * w).sum(dim=1).mean() / w.sum()
            metrics = self._frame_metrics(logits, codes_flat, ce)
        else:
            depth_loss = torch.zeros((), device=device)
            metrics = {}

        return {
            "loss":       depth_loss,
            "depth_loss": depth_loss,
            **metrics,
        }

    @torch.no_grad()
    def _frame_metrics(
        self, logits: torch.Tensor, codes_flat: torch.Tensor, ce: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Monitoring scalars: per-codebook CE, codebook-0 acc, EOS recall/FPR.

        The aggregate weighted loss hides whether codebook 0 (which drives
        intelligibility) is converging while the high-entropy fine RVQ levels
        sit near random. EOS recall/FPR track whether the in-vocab stop code is
        learned — a low recall is what makes greedy decoding run to max_frames.
        All returned tensors are detached per-batch scalar means.
        """
        per_cb = ce.mean(dim=0)                                       # (K,)
        out = {f"ce_cb{k}": per_cb[k].detach() for k in range(self.K)}

        pred0 = logits[:, 0].argmax(dim=-1)                           # (N,)
        tgt0  = codes_flat[:, 0]                                      # (N,)
        out["cb0_acc"] = (pred0 == tgt0).float().mean()

        is_eos = tgt0 == self.eos_id
        pred_eos = pred0 == self.eos_id
        if is_eos.any():
            out["eos_recall"] = pred_eos[is_eos].float().mean()
        if (~is_eos).any():
            out["eos_fpr"] = pred_eos[~is_eos].float().mean()
        return out

    # ------------------------------------------------------------------
    # Inference: frame-autoregressive decode
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        speaker_vec: torch.Tensor,      # (speaker_dim,) or (1, speaker_dim)
        text_ids: torch.Tensor,         # (L,) or (1, L) long
        language: str,
        max_frames: int = 1000,
        greedy: bool = True,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Decode DAC codes for one utterance.

        Returns:
            codes: (T, K) long — feed to DAC.decode (via tts_generate.py) for a
            waveform. T is the number of frames emitted before the model decoded
            EOS in codebook 0 (or max_frames). The EOS frame itself is not
            included (it carries no real DAC code).
        """
        from core.kvcache import KVcache

        device = next(self.parameters()).device
        use_cuda = device.type == "cuda"
        embed_layer = self.aura.get_embed_layer()

        if speaker_vec.dim() == 1:
            speaker_vec = speaker_vec.unsqueeze(0)
        if text_ids.dim() == 1:
            text_ids = text_ids.unsqueeze(0)
        text_ids = text_ids.to(device)

        lang_id = LANG_MAP.get(language, LANG_MAP["eng"])

        # ---- Prompt: [BOS, SPK, synth, LANG, text×L, SPEECH_START] ----
        spk_token = self.spk_proj(speaker_vec.to(device))              # (1, D)
        pieces = [
            embed_layer(torch.tensor([self.aura.bos_id], device=device)),
            spk_token,
            embed_layer(torch.tensor([self.aura.task_tts_id], device=device)),
            embed_layer(torch.tensor([lang_id], device=device)),
            embed_layer(text_ids[0]),
            embed_layer(torch.tensor([self.aura.speech_start_id], device=device)),
        ]
        inputs_embeds = torch.cat(pieces, dim=0).unsqueeze(0)          # (1, S, D)
        S = inputs_embeds.size(1)
        position_ids = torch.arange(S, device=device).unsqueeze(0)
        cache = KVcache(self.aura.n_layers)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
            h = self.aura.forward_hidden(
                inputs_embeds, position_ids, use_cache=True, cache=cache)
        h_last = h[:, -1]                                              # (1, D)

        emitted: list[torch.Tensor] = []
        for step in range(max_frames):
            codes_t = self.depth.generate_frame(
                h_last, greedy=greedy, temperature=temperature,
                top_k=top_k, top_p=top_p, eos_id=self.eos_id)          # (1, K)

            # EOS is decided by the coarsest codebook (predicted first). Once it
            # fires, stop without emitting the EOS frame (no real DAC code).
            if int(codes_t[0, 0].item()) == self.eos_id:
                break
            emitted.append(codes_t)

            # Feed the emitted frame back into the temporal transformer.
            frame_emb = self.codec_emb.frame_embedding(codes_t)        # (1, D)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
                h = self.aura.forward_hidden(
                    frame_emb.unsqueeze(1), position_ids=None,
                    use_cache=True, cache=cache)
            h_last = h[:, -1]

        if not emitted:
            return torch.zeros(0, self.K, dtype=torch.long, device=device)
        return torch.cat(emitted, dim=0)                               # (T, K)

    # ------------------------------------------------------------------
    # Checkpoint helpers (TTS heads + optional LLM adapter/full)
    # ------------------------------------------------------------------

    def save_checkpoint(self, directory: str) -> None:
        import json, os
        os.makedirs(directory, exist_ok=True)

        heads = {
            "codec_emb": self.codec_emb.state_dict(),
            "depth":     self.depth.state_dict(),
            "spk_proj":  self.spk_proj.state_dict(),
        }
        torch.save(heads, f"{directory}/tts_heads.pt")

        if self.aura._lora_layers is not None:
            self.aura.save_adapter(f"{directory}/lora.pt")
        elif any(p.requires_grad for p in self.aura.model.parameters()):
            self.aura.save_full(f"{directory}/llm_full.pt")

        meta = {
            "llm_hidden":   self.aura.hidden_size,
            "n_codebooks":  self.K,
            "cardinality":  self.card,
            "speaker_dim":  self.speaker_dim,
            "depth_dim":    self.depth_dim,
            "depth_layers": self.depth_layers,
            "depth_heads":  self.depth_heads,
            "has_lora":     self.aura._lora_layers is not None,
        }
        with open(f"{directory}/meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        log.info(f"TTS checkpoint saved → {directory}")

    def load_checkpoint(self, directory: str) -> None:
        import os
        heads = torch.load(
            f"{directory}/tts_heads.pt", map_location="cpu", weights_only=True)
        self.codec_emb.load_state_dict(heads["codec_emb"])
        self.depth.load_state_dict(heads["depth"])
        self.spk_proj.load_state_dict(heads["spk_proj"])

        lora_path = f"{directory}/lora.pt"
        if os.path.exists(lora_path):
            self.aura.load_adapter(lora_path)
        llm_path = f"{directory}/llm_full.pt"
        if os.path.exists(llm_path):
            self.aura.load_full(llm_path)
        log.info(f"TTS checkpoint loaded ← {directory}")
