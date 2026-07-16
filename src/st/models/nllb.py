"""
NLLB-200 wrapper — the encoder-decoder counterpart to core.aura.AuraLLM.

Where AuraLLM is a decoder-only LM that consumes audio as a soft prompt, NLLB is a
seq2seq model: audio enters through cross-attention and the output language is chosen
by a single forced BOS token. This wrapper exposes the same surface SpeechAura relies
on (hidden_size, freeze/unfreeze, save/load) so SpeechNLLB can mirror SpeechAura.

Target-side token layout (what tokenizer(text_target=...) produces):
    labels        = [TGT_LANG] t_1 ... t_L </s>
    decoder_input = </s> [TGT_LANG] t_1 ... t_L      (HF shifts right internally)
so decoder_start_token_id is </s> and the language token is the first *predicted*
token — which is exactly what forced_bos_token_id pins at generation time.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# Trainable-parameter policies. "none" = fully frozen (projector-only training).
TRAINABLE_MODES = ("none", "cross_attn", "encoder_cross_attn", "decoder", "all")


class NLLBSeq2Seq(nn.Module):
    """Thin wrapper around M2M100ForConditionalGeneration (NLLB-200).

    Args:
        model_path:  Local dir or HF id (e.g. facebook/nllb-200-distilled-600M).
        trainable:   Which NLLB params get gradients — see TRAINABLE_MODES.
                     "cross_attn" trains only the decoder's encoder-attention
                     blocks, the cheapest way to teach a frozen MT decoder to
                     read a new (speech) representation.
        gradient_checkpointing: Trade throughput for activation memory. The
                     256k-vocab softmax dominates memory here.
    """

    def __init__(
        self,
        model_path: str,
        trainable: str = "none",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast

        if trainable not in TRAINABLE_MODES:
            raise ValueError(
                f"trainable={trainable!r} not in {TRAINABLE_MODES}"
            )

        self.tokenizer = NllbTokenizerFast.from_pretrained(model_path)

        # Language codes must resolve to real tokens: an <unk> here does not
        # raise, it silently makes the model translate into the wrong language.
        from st.data.nllb_lang import verify_lang_codes
        verify_lang_codes(self.tokenizer)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        cfg = self.model.config
        self.hidden_size: int = cfg.d_model
        self.vocab_size:  int = cfg.vocab_size
        self.pad_id:      int = self.tokenizer.pad_token_id
        self.eos_id:      int = self.tokenizer.eos_token_id

        # M2M100 scales token embeddings by sqrt(d_model) (=32 here). Since
        # transformers 4.4x that multiply lives *inside* M2M100ScaledWordEmbedding,
        # so embed_tokens(ids) comes back already scaled and there is no
        # encoder.embed_scale attribute to read. Projected audio bypasses the
        # embedding lookup entirely, so it must be scaled by hand — otherwise it
        # reaches the encoder 32x too small and the sinusoidal position embeddings
        # drown it out. Read the scale from the embedding module, falling back to
        # the config so this survives a transformers version that moves it back.
        self.audio_scale: float = float(
            getattr(
                self.get_embed_layer(), "embed_scale",
                cfg.d_model ** 0.5 if getattr(cfg, "scale_embedding", False) else 1.0,
            )
        )

        self.trainable = trainable
        self._set_trainable(trainable)

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            log.info("  Gradient checkpointing enabled on NLLB")

        log.info(
            f"NLLB loaded ← {model_path} (d_model={self.hidden_size}, "
            f"vocab={self.vocab_size}, audio_scale={self.audio_scale:.1f}, "
            f"trainable={trainable})"
        )
        self._log_trainable()

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def _set_trainable(self, mode: str) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

        if mode == "all":
            for p in self.model.parameters():
                p.requires_grad = True
        elif mode == "decoder":
            # embed_tokens is tied to `shared`, so it is also the *encoder's* input
            # embedding and the lm_head. Training it would drift the frozen text
            # encoder that the alignment loss targets, and move the [SRC_LANG]
            # embedding the projector is calibrated against. Leave it frozen.
            embed = self.model.model.decoder.embed_tokens
            for p in self.model.model.decoder.parameters():
                p.requires_grad = True
            for p in embed.parameters():
                p.requires_grad = False
        elif mode == "encoder_cross_attn":
            # Text encoder (consumes the projected audio) + the decoder's
            # cross-attention (its interface to the encoder). Decoder self-attn/FFN
            # stay frozen so the LM core cannot collapse into an unconditional LM at
            # this step. The tied embedding (shared with the decoder input + lm_head)
            # stays frozen — audio enters as inputs_embeds and bypasses it anyway.
            for p in self.model.model.encoder.parameters():
                p.requires_grad = True
            for p in self.model.model.encoder.embed_tokens.parameters():
                p.requires_grad = False
            for layer in self.model.model.decoder.layers:
                for p in layer.encoder_attn.parameters():
                    p.requires_grad = True
                for p in layer.encoder_attn_layer_norm.parameters():
                    p.requires_grad = True
        elif mode == "cross_attn":
            for layer in self.model.model.decoder.layers:
                for p in layer.encoder_attn.parameters():
                    p.requires_grad = True
                for p in layer.encoder_attn_layer_norm.parameters():
                    p.requires_grad = True
        # "none" → everything stays frozen

    def freeze(self) -> None:
        self._set_trainable("none")

    def unfreeze(self) -> None:
        self._set_trainable("all")

    def _log_trainable(self) -> None:
        t = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n = sum(p.numel() for p in self.model.parameters())
        log.info(f"  NLLB: {n:,} params, {t:,} trainable ({100 * t / n:.1f}%)")

    # ------------------------------------------------------------------
    # Encoder-side entry points
    # ------------------------------------------------------------------

    def get_embed_layer(self) -> nn.Embedding:
        """Token embeddings. NOTE: the returned module already applies the
        sqrt(d_model) scaling — do not scale its output again."""
        return self.model.model.encoder.embed_tokens

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run NLLB's *text* encoder — used as the alignment target (ZeroSwot-style)."""
        return self.model.model.encoder(
            input_ids=input_ids, attention_mask=attention_mask,
        ).last_hidden_state

    def lang_id(self, flores_code: str) -> int:
        tok = self.tokenizer.convert_tokens_to_ids(flores_code)
        if tok == self.tokenizer.unk_token_id:
            raise ValueError(f"{flores_code!r} is <unk> in the NLLB tokenizer")
        return tok

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_full(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load_full(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)

    def save_trainable(self, path: str) -> None:
        """Save only the params that carry gradients (e.g. cross-attention)."""
        state = {
            k: v for k, v in self.model.state_dict().items()
            if dict(self.model.named_parameters()).get(k) is not None
            and dict(self.model.named_parameters())[k].requires_grad
        }
        torch.save(state, path)

    def load_trainable(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            log.warning(f"NLLB trainable-load unexpected keys: {unexpected}")
