"""
SpeechNLLB: end-to-end speech translation with an NLLB-200 decoder.

    audio → SpeechEncoder → (CTCCompressor) → Projector → NLLB → translation

The counterpart to SpeechAura. The difference is structural: SpeechAura splices
projected audio into a decoder-only LM's prompt, so it needs prose templates and
hand-assembled inputs_embeds/labels/position_ids. NLLB is an encoder-decoder — audio
enters through cross-attention and the output language is one forced BOS token — so all
of that template machinery disappears and the loss is plain seq2seq CE.

Two ways to attach the speech encoder (`attach`):

  "encoder_input"   (default) — projected audio is fed to NLLB's *text* encoder as
      inputs_embeds, wrapped as [SRC_LANG] audio×N [</s>]. The pretrained encoder pulls
      the audio representation toward the space the decoder already expects, so this
      works with far less data. (ZeroSwot / CRESS family.)

  "encoder_outputs" — projected audio is handed straight to the decoder as
      encoder_outputs, bypassing the text encoder entirely. This is the SeamlessM4T
      recipe; it is cleaner but the decoder starts out cross-attending to a completely
      out-of-distribution representation, so it needs a lot more data.

Loss:
    CE on the target tokens (from NLLB's own labels handling)
    + ctc_weight * auxiliary CTC loss on the encoder (keeps it phonetically grounded)
    + align_weight * MSE to NLLB's text-encoder output on the paired transcript.
      The alignment term is the ZeroSwot trick: with NLLB frozen, it is what actually
      teaches the projector to land in the text encoder's space. Only meaningful when
      attach="encoder_input" and the CTC compressor is on (so audio length ≈ subword
      length).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from st.data.nllb_lang import to_flores
from st.models.ctc_compressor import CTCCompressor, build_ctc_compressor
from st.models.encoder import SpeechEncoder
from st.models.nllb import NLLBSeq2Seq
from st.models.projector import build_projector

log = logging.getLogger(__name__)

ATTACH_MODES = ("encoder_input", "encoder_outputs")


class SpeechNLLB(nn.Module):
    def __init__(
        self,
        encoder: SpeechEncoder,
        nllb: NLLBSeq2Seq,
        projector_cfg: dict,
        ctc_compress_cfg: dict | None = None,
        ctc_weight: float = 0.0,
        align_weight: float = 0.0,
        attach: str = "encoder_input",
        freeze_encoder: bool = True,
    ):
        super().__init__()

        if attach not in ATTACH_MODES:
            raise ValueError(f"attach={attach!r} not in {ATTACH_MODES}")
        if align_weight > 0.0 and attach != "encoder_input":
            raise ValueError(
                "align_weight > 0 only makes sense with attach='encoder_input' — "
                "there is no text-encoder output to align to otherwise."
            )

        self.encoder      = encoder
        self.nllb         = nllb
        self.attach       = attach
        self.ctc_weight   = ctc_weight
        self.align_weight = align_weight

        if freeze_encoder:
            self.encoder.freeze()
        else:
            self.encoder.unfreeze()

        if ctc_weight > 0.0 and encoder.ctc_head is None:
            raise ValueError(
                "ctc_weight > 0 requires an encoder CTC head (set vocab_size when "
                "loading the Stage 1 encoder checkpoint)."
            )
        if ctc_weight == 0.0 and encoder.ctc_head is not None:
            for p in encoder.ctc_head.parameters():
                p.requires_grad = False

        self.ctc_compressor: CTCCompressor | None = build_ctc_compressor(ctc_compress_cfg)
        if self.ctc_compressor is not None:
            if encoder.ctc_head is None:
                raise ValueError("CTCCompressor requires encoder CTC logits.")
            log.info(f"  CTCCompressor: strategy={self.ctc_compressor.strategy}, "
                     f"remove_blanks={self.ctc_compressor.remove_blanks}")
        else:
            log.info("  CTCCompressor: disabled")

        self.projector = build_projector(
            config=projector_cfg,
            encoder_dim=encoder.get_output_dim(),
            llm_hidden=nllb.hidden_size,
        )
        n = sum(p.numel() for p in self.projector.parameters())
        log.info(f"  Projector ({projector_cfg.get('type', 'mlp')}): {n:,} params "
                 f"→ d_model {nllb.hidden_size}")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        log.info(f"SpeechNLLB ({attach}): {total:,} params, {trainable:,} trainable")

    # ------------------------------------------------------------------
    # Audio encoding
    # ------------------------------------------------------------------

    def encode_audio(
        self, features: torch.Tensor, feature_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """audio → encoder → (compressor) → projector.

        Returns (projected, lengths, ctc_logits, enc_lengths); enc_lengths is the
        pre-compression length the CTC loss needs.
        """
        enc_out     = self.encoder(features, feature_lengths)
        hidden      = enc_out["hidden_states"]
        enc_lengths = enc_out["lengths"]
        ctc_logits  = enc_out.get("ctc_logits")

        lengths = enc_lengths
        if self.ctc_compressor is not None and ctc_logits is not None:
            hidden, lengths = self.ctc_compressor(hidden, ctc_logits, enc_lengths)

        projected = self.projector(hidden, lengths)
        return projected, lengths, ctc_logits, enc_lengths

    # ------------------------------------------------------------------
    # Encoder-side sequence assembly
    # ------------------------------------------------------------------

    def _build_encoder_inputs(
        self,
        audio_embeds: torch.Tensor,
        audio_lens: torch.Tensor,
        src_languages: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wrap projected audio as [SRC_LANG] audio×N [</s>] for NLLB's text encoder.

        Scaling is the subtle part. M2M100 multiplies token embeddings by sqrt(d_model)
        (=32) inside M2M100ScaledWordEmbedding, so embed(ids) is *already* scaled — but
        projected audio never passes through that lookup, so it must be scaled here.
        Skip this and the audio arrives 32x too small, and the sinusoidal position
        embeddings (magnitude ~1) drown it out.

        Returns (inputs_embeds (B,S,D), attention_mask (B,S)).
        """
        B = audio_embeds.size(0)
        embed = self.nllb.get_embed_layer()   # already applies the sqrt(d_model) scale

        eos = torch.tensor([self.nllb.eos_id], dtype=torch.long, device=device)
        eos_emb = embed(eos)                              # (1, D)

        seqs: list[torch.Tensor] = []
        for i in range(B):
            n = int(audio_lens[i].item())
            lang_id = torch.tensor(
                [self.nllb.lang_id(to_flores(src_languages[i]))],
                dtype=torch.long, device=device,
            )
            lang_emb = embed(lang_id)                     # (1, D)
            audio    = audio_embeds[i, :n] * self.nllb.audio_scale   # (n, D)
            seqs.append(torch.cat([lang_emb, audio, eos_emb], dim=0))

        S = max(s.size(0) for s in seqs)
        D = audio_embeds.size(-1)
        inputs_embeds  = audio_embeds.new_zeros(B, S, D)
        attention_mask = torch.zeros(B, S, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            inputs_embeds[i, : s.size(0)] = s
            attention_mask[i, : s.size(0)] = 1

        return inputs_embeds, attention_mask

    @staticmethod
    def _audio_mask(audio_lens: torch.Tensor, T: int, device) -> torch.Tensor:
        return (
            torch.arange(T, device=device).unsqueeze(0) < audio_lens.unsqueeze(1)
        ).long()

    def _nllb_encoder_kwargs(
        self,
        audio_embeds: torch.Tensor,
        audio_lens: torch.Tensor,
        src_languages: list[str],
        device: torch.device,
    ) -> dict:
        """Whichever of inputs_embeds / encoder_outputs this attach mode uses."""
        if self.attach == "encoder_input":
            inputs_embeds, attention_mask = self._build_encoder_inputs(
                audio_embeds, audio_lens, src_languages, device,
            )
            return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}

        # encoder_outputs — straight to the decoder, no text encoder
        from transformers.modeling_outputs import BaseModelOutput
        mask = self._audio_mask(audio_lens, audio_embeds.size(1), device)
        return {
            "encoder_outputs": BaseModelOutput(last_hidden_state=audio_embeds),
            "attention_mask":  mask,
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_lengths: torch.Tensor,
        labels: torch.Tensor,
        src_language: list[str],
        transcript_input_ids: torch.Tensor | None = None,
        transcript_attention_mask: torch.Tensor | None = None,
        ctc_labels: torch.Tensor | None = None,
        ctc_label_lengths: torch.Tensor | None = None,
        **_unused,
    ) -> dict[str, torch.Tensor]:
        device = audio_features.device

        audio_embeds, audio_lens, ctc_logits, enc_lengths = self.encode_audio(
            audio_features, audio_lengths,
        )

        kwargs = self._nllb_encoder_kwargs(
            audio_embeds, audio_lens, src_language, device,
        )
        out = self.nllb.model(**kwargs, labels=labels)
        ce_loss = out.loss

        # --- auxiliary CTC on the speech encoder ---
        ctc_loss = torch.tensor(0.0, device=device)
        if self.ctc_weight > 0.0 and ctc_logits is not None:
            if ctc_labels is None or ctc_label_lengths is None:
                log.warning("ctc_weight > 0 but ctc_labels missing — skipping CTC loss.")
            else:
                log_probs = ctc_logits.log_softmax(dim=-1).transpose(0, 1)
                ctc_loss = F.ctc_loss(
                    log_probs, ctc_labels, enc_lengths, ctc_label_lengths,
                    blank=0, reduction="mean", zero_infinity=True,
                )

        # --- ZeroSwot-style alignment to the text encoder's output space ---
        align_loss = torch.tensor(0.0, device=device)
        if self.align_weight > 0.0 and transcript_input_ids is not None:
            speech_h = self.nllb.model.model.encoder(
                inputs_embeds=kwargs["inputs_embeds"],
                attention_mask=kwargs["attention_mask"],
            ).last_hidden_state
            with torch.no_grad():
                text_h = self.nllb.encode_text(
                    transcript_input_ids, transcript_attention_mask,
                )
            align_loss = self._align_loss(
                speech_h, kwargs["attention_mask"],
                text_h,   transcript_attention_mask,
            )

        loss = ce_loss + self.ctc_weight * ctc_loss + self.align_weight * align_loss
        return {
            "loss":       loss,
            "ce_loss":    ce_loss,
            "ctc_loss":   ctc_loss,
            "align_loss": align_loss,
            "logits":     out.logits,
        }

    @staticmethod
    def _align_loss(
        speech_h: torch.Tensor, speech_mask: torch.Tensor,
        text_h: torch.Tensor,   text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between mean-pooled speech and text encoder states.

        Mean pooling sidesteps the length mismatch between compressed audio and
        subwords; a token-level (optimal-transport) variant is the obvious upgrade.
        """
        def pool(h, m):
            m = m.unsqueeze(-1).to(h.dtype)
            return (h * m).sum(1) / m.sum(1).clamp(min=1.0)

        return F.mse_loss(pool(speech_h, speech_mask), pool(text_h, text_mask))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        audio_features: torch.Tensor,
        audio_lengths: torch.Tensor,
        src_lang: str,
        tgt_lang: str = "english",
        task: str = "st",
        max_new_tokens: int = 256,
        num_beams: int = 4,
        no_repeat_ngram_size: int = 3,
        **_unused,
    ) -> str:
        """Beam-search translation. Returns the decoded string.

        Unlike SpeechAura.generate this is not hand-rolled greedy decoding — NLLB is a
        stock HF seq2seq model, so beam search comes for free.
        """
        device = audio_features.device
        audio_embeds, audio_lens, _, _ = self.encode_audio(audio_features, audio_lengths)

        kwargs = self._nllb_encoder_kwargs(
            audio_embeds, audio_lens, [src_lang], device,
        )
        ids = self.nllb.model.generate(
            **kwargs,
            forced_bos_token_id=self.nllb.lang_id(to_flores(tgt_lang)),
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
        )
        return self.nllb.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]

    # Compatibility with the train_st.py eval loop, which is written against
    # SpeechAura's decoder-only output format.
    def _strip_special_tokens(self, text: str) -> str:
        return text

    def split_cot_output(self, text: str) -> tuple[str, str]:
        """NLLB emits the translation only — there is no transcript segment."""
        return "", text.strip()

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, directory: str) -> None:
        import json, os
        os.makedirs(directory, exist_ok=True)

        torch.save(self.projector.state_dict(), f"{directory}/projector.pt")
        if any(p.requires_grad for p in self.nllb.model.parameters()):
            self.nllb.save_trainable(f"{directory}/nllb_trainable.pt")

        meta = {
            "kind":         "speech_nllb",
            "attach":       self.attach,
            "encoder_dim":  self.encoder.get_output_dim(),
            "d_model":      self.nllb.hidden_size,
            "ctc_weight":   self.ctc_weight,
            "align_weight": self.align_weight,
            "nllb_trainable": self.nllb.trainable,
            "has_ctc_compressor": self.ctc_compressor is not None,
        }
        with open(f"{directory}/meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        log.info(f"Checkpoint saved → {directory}")

    def load_checkpoint(self, directory: str) -> None:
        import os
        self.projector.load_state_dict(
            torch.load(f"{directory}/projector.pt", map_location="cpu", weights_only=True)
        )
        p = f"{directory}/nllb_trainable.pt"
        if os.path.exists(p):
            self.nllb.load_trainable(p)
        log.info(f"Checkpoint loaded ← {directory}")
