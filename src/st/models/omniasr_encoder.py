"""
Live, trainable wrapper around Meta's omniASR_CTC_1B (fairseq2 Wav2Vec2AsrModel).

Unlike scripts/extract_omniasr_features.py (frozen inference only, cached
features), this module supports full backprop into the encoder's weights —
used for Stage 4-equivalent training (encoder + CTCCompressor + projector
trainable, Aura-1B LLM frozen).

Implements the same duck-typed contract as SpeechEncoder (src/st/models/encoder.py):
    forward(features, lengths) -> {"hidden_states", "lengths", "ctc_logits"}
    .freeze() / .unfreeze()
    .get_output_dim() -> int
    .ctc_head attribute

MUST run under the isolated torch==2.8.0+cu128 env (iwslt2026/.envs/omniasr_extract),
never under the main training env (Aura_base/env, torch==2.6.0+cu124) — fairseq2n
requires torch 2.8. All fairseq2 imports are therefore local to methods, not at
module level, so this file stays safely importable (never instantiated) from the
main env too.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

CTC_VOCAB_SIZE = 9812


class OmniASREncoder(nn.Module):
    """Live omniASR_CTC_1B encoder — trainable, fp32 master weights.

    Args:
        checkpoint_path:      Path to omniASR-CTC-1B.pt.
        dropout_p:            Encoder dropout (0.0 in the frozen extraction
                              script; nonzero here for real training).
        attn_dropout_p:       Attention dropout.
        ffn_inner_dropout_p:  FFN dropout.
        layer_drop_p:         LayerDrop (stochastic depth). Default 0.0 —
                              parameter-free-ness not yet exercised in training,
                              keep at 0.0 until the smoke test confirms it.
        final_dropout_p:      Dropout before the CTC head.
        freeze_ctc_head:      Keep final_proj frozen even after unfreeze().
                              Default True — with ctc_weight=0.0 and
                              ctc_compress.strategy="avg" (non-differentiable
                              argmax), final_proj gets zero gradient. Leaving
                              it trainable would let AdamW's decoupled weight
                              decay silently shrink the pretrained CTC head
                              every step for no benefit.
        gradient_checkpointing_every_n: Wrap every Nth transformer layer with
                              activation checkpointing (fairseq2's
                              apply_layerwise_ac, backed by
                              torch.distributed.algorithms._checkpoint —
                              confirmed to preserve state_dict key names, so
                              this can be toggled without affecting checkpoint
                              compatibility). 0 = disabled. Trades ~20-30% more
                              compute per step for much lower activation
                              memory, allowing larger batches — use this to
                              raise max_batch_duration/max_batch_size and
                              lower grad_accum instead of the reverse.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dropout_p: float = 0.1,
        attn_dropout_p: float = 0.1,
        ffn_inner_dropout_p: float = 0.1,
        layer_drop_p: float = 0.0,
        final_dropout_p: float = 0.1,
        freeze_ctc_head: bool = True,
        gradient_checkpointing_every_n: int = 0,
    ):
        super().__init__()
        self.encoder_dim = 1280
        self._freeze_ctc_head = freeze_ctc_head

        self._model = self._build_and_load(
            checkpoint_path, dropout_p, attn_dropout_p,
            ffn_inner_dropout_p, layer_drop_p, final_dropout_p,
        )
        self.ctc_head = self._model.final_proj  # non-None so SpeechAura's
                                                  # `encoder.ctc_head is None` guards pass
        if freeze_ctc_head:
            for p in self.ctc_head.parameters():
                p.requires_grad = False

        if gradient_checkpointing_every_n > 0:
            from fairseq2.models.utils.ac import apply_layerwise_ac
            apply_layerwise_ac(self._model.encoder.layers, gradient_checkpointing_every_n)
            log.info(
                f"Gradient checkpointing enabled: every {gradient_checkpointing_every_n} "
                f"of {len(self._model.encoder.layers)} encoder layers"
            )

    @staticmethod
    def _build_and_load(
        checkpoint_path: str,
        dropout_p: float,
        attn_dropout_p: float,
        ffn_inner_dropout_p: float,
        layer_drop_p: float,
        final_dropout_p: float,
    ):
        from fairseq2.models.transformer import TransformerNormOrder
        from fairseq2.models.wav2vec2 import Wav2Vec2EncoderConfig
        from fairseq2.models.wav2vec2.asr.config import Wav2Vec2AsrConfig
        from fairseq2.models.wav2vec2.asr.factory import Wav2Vec2AsrFactory

        encoder_config = Wav2Vec2EncoderConfig(
            model_dim=1280,
            feature_dim=512,
            num_encoder_layers=48,
            num_encoder_attn_heads=16,
            ffn_inner_dim=5120,
            feature_extractor_layer_descs=[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2,
            feature_extractor_bias=True,
            feature_extractor_layer_norm_convs=True,
            layer_norm_features=False,
            pos_encoder_type="conv",
            pos_conv_kernel_size=128,
            num_pos_conv_groups=16,
            norm_order=TransformerNormOrder.PRE,
            dropout_p=dropout_p,
            attn_dropout_p=attn_dropout_p,
            ffn_inner_dropout_p=ffn_inner_dropout_p,
            layer_drop_p=layer_drop_p,
        )
        config = Wav2Vec2AsrConfig(
            encoder_config=encoder_config,
            target_vocab_size=CTC_VOCAB_SIZE,
            use_masking=False,
            final_dropout_p=final_dropout_p,
        )

        model = Wav2Vec2AsrFactory(config).create_model()

        # weights_only=False: our own fine-tuned checkpoints (see
        # st.training.pretrain_omniasr_ctc.save_checkpoint) bundle optimizer/
        # scheduler state alongside the weights, which weights_only=True's
        # unpickling allowlist doesn't cover. Same trust model already used by
        # pretrain_omniasr_ctc.py's own --resume_from path — these are our own
        # locally-produced checkpoints, not untrusted downloads.
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            # Our own fine-tuned checkpoint format (see
            # st.training.pretrain_omniasr_ctc.save_checkpoint) — saved via
            # OmniASREncoder.state_dict(), so keys carry a "_model." prefix
            # (this class's inner fairseq2 model lives at self._model) that
            # the raw fairseq2 `model` object below doesn't have. Also drop
            # the top-level "ctc_head.*" keys: OmniASREncoder.ctc_head is
            # just `self._model.final_proj` re-registered under a second
            # attribute name (for SpeechAura's `encoder.ctc_head is None`
            # guard), so those keys are exact duplicates of
            # "_model.final_proj.*" already covered below — keeping both
            # would make the raw model's strict=True load see an
            # "unexpected key" it has no attribute for.
            state = {
                k.removeprefix("_model."): v
                for k, v in ckpt["model_state_dict"].items()
                if k.startswith("_model.")
            }
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]              # base pretrained omniASR-CTC-1B.pt format
        else:
            state = ckpt                       # bare state dict
        # strict=True on purpose — dropout/layer_drop have no learned params, so a
        # load failure here is still the loudest, cheapest signal of a config mistake.
        model.load_state_dict(state, strict=True)
        # No .to(dtype=bf16) — keep fp32 master weights for AdamW; the outer training
        # loop wraps forward passes in torch.amp.autocast(dtype=bf16), same as
        # AuraLLM/projector.
        log.info(f"Loaded live omniASR_CTC_1B ← {checkpoint_path} (strict=True, trainable)")
        return model

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            features: (B, T_samples) raw waveform, 16kHz, per-utterance
                      layer_norm-normalized (see RawAudioDataset).
            lengths:  (B,) valid sample counts.

        Returns dict with:
            hidden_states: (B, T, 1280)
            lengths:       (B,) post-frontend frame counts
            ctc_logits:    (B, T, 9812)
        """
        from fairseq2.nn import BatchLayout

        layout = BatchLayout.of(features, seq_lens=lengths.tolist())

        seqs, layout, _ = self._model.encoder_frontend.extract_features(features, layout)
        seqs, _ = self._model.encoder_frontend.process_features(seqs, layout, masker=None)
        hidden_states = self._model.encoder(seqs, layout)      # (B, T, 1280)
        ctc_logits = self._model.final_proj(hidden_states)      # (B, T, 9812)

        out_lengths = (
            torch.as_tensor(layout.seq_lens, dtype=torch.long, device=hidden_states.device)
            if layout.padded
            else torch.full(
                (hidden_states.size(0),), hidden_states.size(1),
                dtype=torch.long, device=hidden_states.device,
            )
        )

        return {
            "hidden_states": hidden_states,
            "lengths": out_lengths,
            "ctc_logits": ctc_logits,
        }

    def get_output_dim(self) -> int:
        return self.encoder_dim

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
        if self._freeze_ctc_head:
            for p in self.ctc_head.parameters():
                p.requires_grad = False


def build_omniasr_encoder_from_config(config: dict) -> OmniASREncoder:
    """Build an OmniASREncoder from a config dict (cfg["encoder"] when
    cfg["encoder"]["type"] == "omniasr_live"). Mirrors
    load_encoder_from_checkpoint()'s config-dict-in style.

    Config keys: checkpoint, dropout_p, attn_dropout_p, ffn_inner_dropout_p,
    layer_drop_p, final_dropout_p, freeze_ctc_head, gradient_checkpointing_every_n.
    """
    return OmniASREncoder(
        checkpoint_path=config["checkpoint"],
        dropout_p=config.get("dropout_p", 0.1),
        attn_dropout_p=config.get("attn_dropout_p", 0.1),
        ffn_inner_dropout_p=config.get("ffn_inner_dropout_p", 0.1),
        layer_drop_p=config.get("layer_drop_p", 0.0),
        final_dropout_p=config.get("final_dropout_p", 0.1),
        freeze_ctc_head=config.get("freeze_ctc_head", True),
        gradient_checkpointing_every_n=config.get("gradient_checkpointing_every_n", 0),
    )
