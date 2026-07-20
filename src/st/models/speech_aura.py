"""
SpeechAura: end-to-end speech translation model.

Wires:
    audio → SpeechEncoder → (CTCCompressor) → Projector → AuraLLM → logits

Training modes (controlled by config, not separate classes):
    Stage 1 — CTC pretraining:    handled by pretrain_ctc.py, not this file.
    Stage 2 — Projector only:     freeze_encoder=True, freeze_llm=True, lora_rank=0
    Stage 3 — Projector + LoRA:   freeze_encoder=True, freeze_llm=True, lora_rank>0
    Stage 4 — Full fine-tune:     freeze_encoder=False, freeze_llm=False, lora_rank=0

Loss:
    CE loss on target tokens (always).
    + auxiliary CTC loss weighted by ctc_weight (when ctc_weight > 0 and encoder has CTC head).
    CTC auxiliary loss keeps the encoder phonetically grounded during ST training.
    Ref: Chimera (Tang et al. 2021), ESPnet-ST.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from st.models.encoder import SpeechEncoder
from st.models.projector import build_projector
from st.models.ctc_compressor import CTCCompressor, build_ctc_compressor
from st.models.aura import AuraLLM, LANG_MAP

log = logging.getLogger(__name__)


class SpeechAura(nn.Module):
    """
    Encoder → (CTC Compressor) → Projector → Aura-1B.

    Args:
        encoder:        Pretrained SpeechEncoder, or None for cached-features mode
                        (see encode_audio_cached/forward_cached/generate_cached) —
                        in which case encoder_output_dim must be given instead.
        aura:           AuraLLM wrapper (already loaded + frozen/unfrozen as desired).
        projector_cfg:  Dict passed to build_projector().
        ctc_compress_cfg: Dict passed to build_ctc_compressor(). None = disabled.
        ctc_weight:     Weight for auxiliary CTC loss (0.0 = disabled). Must be 0.0
                        when encoder is None (cached mode has no live ctc_logits).
        freeze_encoder: Freeze encoder weights. Ignored when encoder is None.
        freeze_llm:     Freeze LLM weights (set False for full fine-tune).
        encoder_output_dim: Encoder hidden dim, required when encoder is None
                        (cached mode) since there's no live encoder to query.
    """

    def __init__(
        self,
        encoder: SpeechEncoder | None,
        aura: AuraLLM,
        projector_cfg: dict,
        ctc_compress_cfg: dict | None = None,
        ctc_weight: float = 0.0,
        freeze_encoder: bool = True,
        freeze_llm: bool = True,
        encoder_output_dim: int | None = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.aura    = aura
        self.ctc_weight = ctc_weight

        if encoder is not None:
            # Freeze / unfreeze components
            if freeze_encoder:
                self.encoder.freeze()
            else:
                self.encoder.unfreeze()

            # Validate CTC auxiliary loss requirements
            if ctc_weight > 0.0 and encoder.ctc_head is None:
                raise ValueError(
                    "ctc_weight > 0 requires encoder to have a CTC head (vocab_size must be set). "
                    "Load the encoder checkpoint with vocab_size from Stage 1."
                )
            if ctc_compress_cfg is not None and ctc_compress_cfg.get("enabled", True) \
                    and encoder.ctc_head is None:
                raise ValueError(
                    "CTCCompressor requires encoder CTC logits (encoder.ctc_head must exist). "
                    "Set vocab_size when loading the encoder."
                )
            self._encoder_output_dim = encoder.get_output_dim()
        else:
            # Cached-features mode: no live encoder, no live ctc_logits.
            if encoder_output_dim is None:
                raise ValueError("encoder_output_dim is required when encoder is None.")
            if ctc_weight > 0.0:
                raise ValueError(
                    "ctc_weight > 0 requires a live encoder with ctc_logits — cached-features "
                    "mode only stores argmax predicted_ids, not per-frame logits. Set ctc_weight=0.0."
                )
            self._encoder_output_dim = encoder_output_dim

        if freeze_llm:
            self.aura.freeze()
        else:
            self.aura.unfreeze()

        # CTC compressor (optional)
        self.ctc_compressor: CTCCompressor | None = build_ctc_compressor(ctc_compress_cfg)
        if self.ctc_compressor is not None:
            log.info(f"  CTCCompressor enabled: strategy={self.ctc_compressor.strategy}, "
                     f"remove_blanks={self.ctc_compressor.remove_blanks}")
        else:
            log.info("  CTCCompressor: disabled")

        # Projector
        self.projector = build_projector(
            config=projector_cfg,
            encoder_dim=self._encoder_output_dim,
            llm_hidden=aura.hidden_size,
        )
        n = sum(p.numel() for p in self.projector.parameters())
        log.info(f"  Projector ({projector_cfg.get('type', 'mlp')}): {n:,} params")
        self.projector_dtype = next(self.projector.parameters()).dtype

        # Summary
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        log.info(f"SpeechAura: {total:,} params total, {trainable:,} trainable")

    # ------------------------------------------------------------------
    # Audio encoding (encoder → compressor → projector)
    # ------------------------------------------------------------------

    def encode_audio(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Run audio through encoder → (compressor) → projector.

        Returns:
            projected:    (B, T', llm_hidden)
            lengths:      (B,) valid token counts AFTER compression
            ctc_logits:   (B, T_enc, vocab) or None — pre-compression, for CTC loss
            enc_lengths:  (B,) encoder output lengths BEFORE compression, or None
                          Same as lengths when compressor is disabled.
        """
        enc_out    = self.encoder(features, feature_lengths)
        hidden     = enc_out["hidden_states"]   # (B, T_enc, D)
        enc_lengths = enc_out["lengths"]         # (B,) — pre-compression
        ctc_logits = enc_out.get("ctc_logits")  # (B, T_enc, V) or None

        # Optional CTC compression — enc_lengths stays at pre-compression value
        # so the CTC loss uses the correct uncompressed sequence lengths
        lengths = enc_lengths
        if self.ctc_compressor is not None and ctc_logits is not None:
            hidden, lengths = self.ctc_compressor(hidden, ctc_logits, enc_lengths)

        projected = self.projector(hidden, lengths)
        return projected, lengths, ctc_logits, enc_lengths

    def encode_audio_cached(
        self,
        hidden_states: torch.Tensor,
        predicted_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same as encode_audio(), but for precomputed encoder features (see
        CachedFeatureDataset) — skips the live encoder entirely, using cached
        hidden_states + argmax predicted_ids instead of running self.encoder(...)
        and computing ctc_logits.

        Args:
            hidden_states: (B, T, D) cached encoder hidden states
            predicted_ids: (B, T) cached argmax CTC predictions per frame
            lengths:       (B,) valid frame counts per utterance

        Returns:
            projected: (B, T', llm_hidden)
            lengths:   (B,) valid token counts AFTER compression
        """
        # Cached hidden_states are stored as fp16 (see CachedFeatureCollator). The live
        # encoder path always produces fp32, so callers outside an autocast context
        # (e.g. generate_cached, which calls this before entering its autocast block)
        # would otherwise hit a dtype mismatch against the projector's fp32 weights.
        hidden_states = hidden_states.to(self.projector_dtype)

        if self.ctc_compressor is not None:
            hidden_states, lengths = self.ctc_compressor.forward_cached(
                hidden_states, predicted_ids, lengths
            )
        projected = self.projector(hidden_states, lengths)
        return projected, lengths

    # ------------------------------------------------------------------
    # Sequence assembly (called from forward and generate)
    # ------------------------------------------------------------------

    def _build_inputs(
        self,
        audio_embeds: torch.Tensor,   # (B, T_audio, D)
        audio_lens: torch.Tensor,     # (B,)
        target_ids: torch.Tensor,     # (B, L_target) — padded target token ids
        target_lengths: torch.Tensor, # (B,)
        languages: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build inputs_embeds, labels, and position_ids for the LLM.

        Sequence per sample:
            [BOS, LANG, audio×N_i, TRANSCRIPT_START, t1...tL, EOS]

        Labels:
            [-100, -100, -100×N_i, -100, t1...tL, EOS]

        Samples are padded to the longest sequence in the batch with EOS id.
        Label padding positions are -100.

        Returns:
            inputs_embeds: (B, S_max, D)
            labels:        (B, S_max)  with -100 on non-target positions
            position_ids:  (B, S_max)
        """
        embed_layer = self.aura.get_embed_layer()
        B = audio_embeds.size(0)

        seqs:   list[torch.Tensor] = []  # list of (S_i, D) embed tensors
        labels_list: list[torch.Tensor] = []  # list of (S_i,) label tensors

        for i in range(B):
            n_audio  = int(audio_lens[i].item())
            lang_id  = LANG_MAP.get(languages[i], LANG_MAP["eng"])
            n_target = int(target_lengths[i].item())
            tgt      = target_ids[i, :n_target]  # (L,)

            # --- build token id sequence for non-audio positions ---
            prefix_ids = torch.tensor(
                [self.aura.bos_id, lang_id],
                dtype=torch.long, device=device
            )
            suffix_ids = torch.tensor(
                [self.aura.task_token_id],
                dtype=torch.long, device=device
            )

            # --- build embeddings ---
            prefix_emb  = embed_layer(prefix_ids)           # (2, D)
            audio_emb   = audio_embeds[i, :n_audio]         # (N, D)
            suffix_emb  = embed_layer(suffix_ids)            # (1, D)
            target_emb  = embed_layer(tgt.to(device))        # (L, D)

            seq_emb = torch.cat([prefix_emb, audio_emb, suffix_emb, target_emb], dim=0)
            seqs.append(seq_emb)

            # --- build labels ---
            # No explicit shift: logits[i] predicts the token AFTER position i,
            # so the "next token" is stored directly at labels[i]. The sequence
            # is [BOS, LANG, audio x n_audio, TRANSCRIPT_START, target...]
            # (0-indexed), so TRANSCRIPT_START sits at index 2 + n_audio; the
            # logit there predicts the first target token, so target labels
            # start at that same index: prompt_len = 1(LANG) + n_audio(audio) + 1(TRANSCRIPT_START).
            prompt_len = 1 + n_audio + 1
            lab = torch.full((seq_emb.size(0),), -100, dtype=torch.long, device=device)
            lab[prompt_len : prompt_len + n_target] = tgt.to(device)
            lab[prompt_len + n_target] = self.aura.eos_id   # loss on EOS
            labels_list.append(lab)

        # --- pad to max sequence length ---
        S_max = max(s.size(0) for s in seqs)

        # Guard against silently running past the LLM's real position range —
        # RoPE has no built-in length cap, so an over-long sequence wouldn't
        # error, it would just extrapolate into positions the model never
        # saw during pretraining and degrade quality invisibly.
        model_max_seq_len = self.aura.model.config.max_seq_len
        if S_max > model_max_seq_len:
            raise ValueError(
                f"Assembled sequence length {S_max} exceeds the LLM's max_seq_len="
                f"{model_max_seq_len}. Reduce max_duration / increase CTC compression, "
                f"or this batch would silently run past the model's pretrained "
                f"position range."
            )

        D     = audio_embeds.size(-1)
        eos_emb_pad = embed_layer(
            torch.tensor([self.aura.eos_id], dtype=torch.long, device=device)
        ).squeeze(0)  # (D,) — pad embeddings with EOS embedding

        inputs_embeds = torch.stack([
            torch.cat([s, eos_emb_pad.unsqueeze(0).expand(S_max - s.size(0), -1)], dim=0)
            if s.size(0) < S_max else s
            for s in seqs
        ])  # (B, S_max, D)

        labels = torch.stack([
            torch.cat([l, torch.full((S_max - l.size(0),), -100, dtype=torch.long, device=device)])
            if l.size(0) < S_max else l
            for l in labels_list
        ])  # (B, S_max)

        position_ids = torch.zeros(B, S_max, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            real_len = s.size(0)
            position_ids[i, :real_len] = torch.arange(real_len, device=device)
            if real_len < S_max:
                position_ids[i, real_len:] = real_len - 1

        return inputs_embeds, labels, position_ids

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        audio_features: torch.Tensor,   # (B, T_mel, 80)
        audio_lengths: torch.Tensor,    # (B,) mel frame counts
        target_ids: torch.Tensor,       # (B, L_target) padded target token ids
        target_lengths: torch.Tensor,   # (B,) actual target lengths
        languages: list[str],           # (B,) language codes
        ctc_labels: torch.Tensor | None = None,         # (B, L) for aux CTC loss
        ctc_label_lengths: torch.Tensor | None = None,  # (B,)
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict with:
            loss:      scalar — CE loss + ctc_weight * CTC loss
            ce_loss:   scalar
            ctc_loss:  scalar (0.0 if disabled)
            logits:    (B, S, vocab_size)
        """
        device = audio_features.device

        # 1. Encode audio — single forward, get actual post-compression lengths
        audio_embeds, audio_lens, ctc_logits, enc_lengths = self.encode_audio(
            audio_features, audio_lengths
        )

        # 2. Build inputs_embeds, labels, position_ids (no placeholders, no mismatch)
        inputs_embeds, labels, position_ids = self._build_inputs(
            audio_embeds, audio_lens,
            target_ids, target_lengths,
            languages, device,
        )

        # 3. LLM forward
        logits = self.aura(inputs_embeds, position_ids)  # (B, S, V)

        # 4. CE loss
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # 5. Auxiliary CTC loss
        ctc_loss = torch.tensor(0.0, device=device)
        if self.ctc_weight > 0.0 and ctc_logits is not None:
            if ctc_labels is None or ctc_label_lengths is None:
                log.warning("ctc_weight > 0 but ctc_labels not provided — skipping CTC loss.")
            else:
                log_probs = ctc_logits.log_softmax(dim=-1).transpose(0, 1)  # (T_enc, B, V)
                ctc_loss = F.ctc_loss(
                    log_probs, ctc_labels, enc_lengths, ctc_label_lengths,
                    blank=0, reduction="mean", zero_infinity=True,
                )

        loss = ce_loss + self.ctc_weight * ctc_loss

        return {
            "loss":     loss,
            "ce_loss":  ce_loss,
            "ctc_loss": ctc_loss,
            "logits":   logits,
        }

    def forward_cached(
        self,
        hidden_states: torch.Tensor,     # (B, T, D) cached encoder hidden states
        predicted_ids: torch.Tensor,     # (B, T) cached argmax CTC predictions
        audio_lengths: torch.Tensor,     # (B,) valid cached-feature frame counts
        target_ids: torch.Tensor,        # (B, L_target) padded target token ids
        target_lengths: torch.Tensor,    # (B,) actual target lengths
        languages: list[str],            # (B,) language codes
    ) -> dict[str, torch.Tensor]:
        """Same as forward(), but for precomputed encoder features (see
        CachedFeatureDataset/CachedFeatureCollator). No auxiliary CTC loss —
        cached mode has no per-frame ctc_logits to supervise it from (enforced
        at __init__ time: ctc_weight must be 0.0 when encoder is None).

        Returns dict with:
            loss:    scalar — CE loss only
            ce_loss: scalar
            logits:  (B, S, vocab_size)
        """
        device = hidden_states.device

        # 1. Encode audio — single call, get actual post-compression lengths
        audio_embeds, audio_lens = self.encode_audio_cached(
            hidden_states, predicted_ids, audio_lengths
        )

        # 2. Build inputs_embeds, labels, position_ids
        inputs_embeds, labels, position_ids = self._build_inputs(
            audio_embeds, audio_lens,
            target_ids, target_lengths,
            languages, device,
        )

        # 3. LLM forward
        logits = self.aura(inputs_embeds, position_ids)  # (B, S, V)

        # 4. CE loss
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        return {
            "loss":     ce_loss,
            "ce_loss":  ce_loss,
            "ctc_loss": torch.tensor(0.0, device=device),
            "logits":   logits,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        audio_features: torch.Tensor,   # (1, T_mel, 80)
        audio_lengths: torch.Tensor,    # (1,)
        target_lang: str = "eng",
        max_new_tokens: int = 256,
    ) -> str:
        """Autoregressive decoding with KV cache."""
        from st.models.kvcache import KVcache

        device = audio_features.device
        audio_embeds, audio_lens, _, _ = self.encode_audio(audio_features, audio_lengths)
        n_audio = int(audio_lens[0].item())

        # Build prompt embeddings: [BOS, LANG, audio×N, TRANSCRIPT_START]
        embed_layer = self.aura.get_embed_layer()
        lang_id  = LANG_MAP.get(target_lang, LANG_MAP["eng"])
        prefix_ids = torch.tensor(
            [self.aura.bos_id, lang_id, self.aura.task_token_id],
            dtype=torch.long, device=device,
        )
        prefix_emb = embed_layer(prefix_ids)  # (3, D)
        audio_emb  = audio_embeds[0, :n_audio]  # (N, D)

        # [BOS, LANG] + audio + [TRANSCRIPT_START]
        inputs_embeds = torch.cat([
            prefix_emb[:2],   # BOS, LANG
            audio_emb,
            prefix_emb[2:],   # TRANSCRIPT_START
        ], dim=0).unsqueeze(0)  # (1, S, D)

        S = inputs_embeds.size(1)
        model_max_seq_len = self.aura.model.config.max_seq_len
        if S >= model_max_seq_len:
            raise ValueError(
                f"Prompt length {S} already exceeds the LLM's max_seq_len="
                f"{model_max_seq_len} before generation even starts."
            )
        max_new_tokens = min(max_new_tokens, model_max_seq_len - S)

        position_ids = torch.arange(S, device=device).unsqueeze(0)
        cache = KVcache(self.aura.n_layers)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=(device.type == "cuda")):
            h = inputs_embeds
            for layer in self.aura.model.model.layers:
                h = layer(h, position_ids=position_ids, use_cache=True, cache=cache)
            h      = self.aura.model.model.norm(h)
            logits = self.aura.model.lm_head(h).float()

        generated  = []
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)

        for step in range(max_new_tokens):
            tok = int(next_token.item())
            if tok == self.aura.eos_id:
                break
            generated.append(tok)

            pos = torch.tensor([[S + step]], device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                h = embed_layer(next_token)
                for layer in self.aura.model.model.layers:
                    h = layer(h, position_ids=pos, use_cache=True, cache=cache)
                h      = self.aura.model.model.norm(h)
                logits = self.aura.model.lm_head(h).float()
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        return self.aura.tokenizer.decode(generated, skip_special_tokens=True)

    @torch.inference_mode()
    def generate_cached(
        self,
        hidden_states: torch.Tensor,   # (1, T, D) cached encoder hidden states
        predicted_ids: torch.Tensor,   # (1, T) cached argmax CTC predictions
        audio_lengths: torch.Tensor,   # (1,)
        target_lang: str = "eng",
        max_new_tokens: int = 256,
    ) -> str:
        """Same as generate(), but for precomputed encoder features."""
        from st.models.kvcache import KVcache

        device = hidden_states.device
        audio_embeds, audio_lens = self.encode_audio_cached(
            hidden_states, predicted_ids, audio_lengths
        )
        n_audio = int(audio_lens[0].item())

        embed_layer = self.aura.get_embed_layer()
        lang_id  = LANG_MAP.get(target_lang, LANG_MAP["eng"])
        prefix_ids = torch.tensor(
            [self.aura.bos_id, lang_id, self.aura.task_token_id],
            dtype=torch.long, device=device,
        )
        prefix_emb = embed_layer(prefix_ids)  # (3, D)
        audio_emb  = audio_embeds[0, :n_audio]  # (N, D)

        inputs_embeds = torch.cat([
            prefix_emb[:2],   # BOS, LANG
            audio_emb,
            prefix_emb[2:],   # TRANSCRIPT_START
        ], dim=0).unsqueeze(0)  # (1, S, D)

        S = inputs_embeds.size(1)
        model_max_seq_len = self.aura.model.config.max_seq_len
        if S >= model_max_seq_len:
            raise ValueError(
                f"Prompt length {S} already exceeds the LLM's max_seq_len="
                f"{model_max_seq_len} before generation even starts."
            )
        max_new_tokens = min(max_new_tokens, model_max_seq_len - S)

        position_ids = torch.arange(S, device=device).unsqueeze(0)
        cache = KVcache(self.aura.n_layers)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=(device.type == "cuda")):
            h = inputs_embeds
            for layer in self.aura.model.model.layers:
                h = layer(h, position_ids=position_ids, use_cache=True, cache=cache)
            h      = self.aura.model.model.norm(h)
            logits = self.aura.model.lm_head(h).float()

        generated  = []
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)

        for step in range(max_new_tokens):
            tok = int(next_token.item())
            if tok == self.aura.eos_id:
                break
            generated.append(tok)

            pos = torch.tensor([[S + step]], device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                h = embed_layer(next_token)
                for layer in self.aura.model.model.layers:
                    h = layer(h, position_ids=pos, use_cache=True, cache=cache)
                h      = self.aura.model.model.norm(h)
                logits = self.aura.model.lm_head(h).float()
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        return self.aura.tokenizer.decode(generated, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Checkpoint helpers (projector + optional LLM adapter/full)
    # ------------------------------------------------------------------

    def save_checkpoint(self, directory: str) -> None:
        import json, os
        os.makedirs(directory, exist_ok=True)

        torch.save(self.projector.state_dict(), f"{directory}/projector.pt")

        if self.encoder is not None and any(p.requires_grad for p in self.encoder.parameters()):
            torch.save(self.encoder.state_dict(), f"{directory}/encoder.pt")

        if self.aura._lora_layers is not None:
            self.aura.save_adapter(f"{directory}/lora.pt")
        elif any(p.requires_grad for p in self.aura.model.parameters()):
            self.aura.save_full(f"{directory}/llm_full.pt")

        meta = {
            "encoder_dim":  self._encoder_output_dim,
            "llm_hidden":   self.aura.hidden_size,
            "ctc_weight":   self.ctc_weight,
            "has_lora":     self.aura._lora_layers is not None,
            "has_ctc_compressor": self.ctc_compressor is not None,
        }
        with open(f"{directory}/meta.json", "w") as f:
            import json
            json.dump(meta, f, indent=2)

        log.info(f"Checkpoint saved → {directory}")

    def load_checkpoint(self, directory: str) -> None:
        import os
        self.projector.load_state_dict(
            torch.load(f"{directory}/projector.pt", map_location="cpu", weights_only=True)
        )
        encoder_path = f"{directory}/encoder.pt"
        if self.encoder is not None and os.path.exists(encoder_path):
            self.encoder.load_state_dict(
                torch.load(encoder_path, map_location="cpu", weights_only=True)
            )
            log.info(f"Encoder loaded ← {encoder_path}")
        lora_path = f"{directory}/lora.pt"
        if os.path.exists(lora_path):
            self.aura.load_adapter(lora_path)
        llm_path = f"{directory}/llm_full.pt"
        if os.path.exists(llm_path):
            self.aura.load_full(llm_path)
        log.info(f"Checkpoint loaded ← {directory}")