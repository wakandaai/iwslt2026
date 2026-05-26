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
from st.models.aura import (
    AuraLLM,
    TRANSLATE_START_ID, TASK_ASR_ID, TASK_COT_ID,
    LANG_MAP,
)
log = logging.getLogger(__name__)


class SpeechAura(nn.Module):
    """
    Encoder → (CTC Compressor) → Projector → Aura-1B.

    Args:
        encoder:        Pretrained SpeechEncoder.
        aura:           AuraLLM wrapper (already loaded + frozen/unfrozen as desired).
        projector_cfg:  Dict passed to build_projector().
        ctc_compress_cfg: Dict passed to build_ctc_compressor(). None = disabled.
        ctc_weight:     Weight for auxiliary CTC loss (0.0 = disabled).
        freeze_encoder: Freeze encoder weights.
        freeze_llm:     Freeze LLM weights (set False for full fine-tune).
    """

    def __init__(
        self,
        encoder: SpeechEncoder,
        aura: AuraLLM,
        projector_cfg: dict,
        ctc_compress_cfg: dict | None = None,
        ctc_weight: float = 0.0,
        freeze_encoder: bool = True,
        freeze_llm: bool = True,
    ):
        super().__init__()

        self.encoder = encoder
        self.aura    = aura
        self.ctc_weight = ctc_weight

        # Freeze / unfreeze components
        if freeze_encoder:
            self.encoder.freeze()
        else:
            self.encoder.unfreeze()

        if freeze_llm:
            self.aura.freeze()
        else:
            self.aura.unfreeze()

        # Validate CTC auxiliary loss requirements
        if ctc_weight > 0.0 and encoder.ctc_head is None:
            raise ValueError(
                "ctc_weight > 0 requires encoder to have a CTC head (vocab_size must be set). "
                "Load the encoder checkpoint with vocab_size from Stage 1."
            )
        
        if self.ctc_weight == 0.0 and encoder.ctc_head is not None:
            for p in encoder.ctc_head.parameters():
                p.requires_grad = False

        # CTC compressor (optional)
        self.ctc_compressor: CTCCompressor | None = build_ctc_compressor(ctc_compress_cfg)
        if self.ctc_compressor is not None:
            if encoder.ctc_head is None:
                raise ValueError(
                    "CTCCompressor requires encoder CTC logits (encoder.ctc_head must exist). "
                    "Set vocab_size when loading the encoder."
                )
            log.info(f"  CTCCompressor enabled: strategy={self.ctc_compressor.strategy}, "
                     f"remove_blanks={self.ctc_compressor.remove_blanks}")
        else:
            log.info("  CTCCompressor: disabled")

        # Projector
        self.projector = build_projector(
            config=projector_cfg,
            encoder_dim=encoder.get_output_dim(),
            llm_hidden=aura.hidden_size,
        )
        n = sum(p.numel() for p in self.projector.parameters())
        log.info(f"  Projector ({projector_cfg.get('type', 'mlp')}): {n:,} params")

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

    # ------------------------------------------------------------------
    # Sequence assembly (called from forward and generate)
    # ------------------------------------------------------------------

    def _build_inputs(
        self,
        audio_embeds: torch.Tensor,
        audio_lens: torch.Tensor,
        transcript_ids: torch.Tensor,
        transcript_lengths: torch.Tensor,
        translation_ids: torch.Tensor,
        translation_lengths: torch.Tensor,
        src_languages: list[str],
        tgt_languages: list[str],
        tasks: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build inputs_embeds, labels, and position_ids for the LLM.

        Layout uses <|transcribe|> and <|translate|> as semantic segment
        markers, each followed by a language token (matching the Aura
        pretraining `<lang> text` adjacency). ASR is structurally a CoT
        with the translation segment omitted.

        - src_languages[i]: language of the audio / transcript
        - tgt_languages[i]: language of the translation (CoT only; ignored for ASR)

        ASR layout (length S = 3 + N + L_t):
            [BOS, audio×N, <|transcribe|>, SRC_LANG, t_1...t_{L_t}]

            prompt_len = 2 + N    (position of SRC_LANG, predicts t_1)
            lab[prompt_len .. prompt_len+L_t-1] = transcript
            lab[prompt_len + L_t]               = EOS

        CoT layout (length S = 5 + N + L_t + L_r):
            [BOS, audio×N, <|transcribe|>, SRC_LANG, t_1...t_{L_t},
             <|translate|>, TGT_LANG, r_1...r_{L_r}]

            prompt_len = 2 + N
            lab[prompt_len .. prompt_len+L_t-1]         = transcript
            lab[prompt_len + L_t]                       = TRANSLATE_START_ID
            lab[prompt_len + L_t + 1]                   = TGT_LANG
            lab[prompt_len + L_t + 2 .. +L_t+1+L_r]     = translation
            lab[prompt_len + L_t + 2 + L_r]             = EOS
        """
        embed_layer = self.aura.get_embed_layer()
        B = audio_embeds.size(0)

        seqs:        list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []

        for i in range(B):
            n_audio = int(audio_lens[i].item())
            n_t     = int(transcript_lengths[i].item())
            n_r     = int(translation_lengths[i].item())
            task    = tasks[i]
            if task not in ("asr", "cot", "st"):
                raise ValueError(f"Unknown task '{task}' (expected 'asr', 'cot', or 'st')")

            src_lang_id = LANG_MAP.get(src_languages[i], LANG_MAP["eng"])
            tgt_lang_id = LANG_MAP.get(tgt_languages[i], LANG_MAP["eng"])

            bos_emb = embed_layer(
                torch.tensor([self.aura.bos_id], dtype=torch.long, device=device)
            )
            audio_emb = audio_embeds[i, :n_audio]
            transcribe_emb = embed_layer(
                torch.tensor([self.aura.task_asr_id], dtype=torch.long, device=device)
            )
            src_lang_emb = embed_layer(
                torch.tensor([src_lang_id], dtype=torch.long, device=device)
            )
            transcript = transcript_ids[i, :n_t].to(device)
            transcript_emb = embed_layer(transcript)

            prompt_len = 2 + n_audio  # position of SRC_LANG

            if task == "asr":
                seq_emb = torch.cat(
                    [bos_emb, audio_emb, transcribe_emb, src_lang_emb, transcript_emb],
                    dim=0,
                )
                S = 3 + n_audio + n_t
                lab = torch.full((S,), -100, dtype=torch.long, device=device)
                if n_t > 0:
                    lab[prompt_len : prompt_len + n_t] = transcript
                lab[prompt_len + n_t] = self.aura.eos_id

            elif task == "cot":
                translate_emb = embed_layer(
                    torch.tensor([self.aura.translate_start_id],
                                 dtype=torch.long, device=device)
                )
                tgt_lang_emb = embed_layer(
                    torch.tensor([tgt_lang_id], dtype=torch.long, device=device)
                )
                translation = translation_ids[i, :n_r].to(device)
                translation_emb = embed_layer(translation)

                seq_emb = torch.cat([
                    bos_emb, audio_emb, transcribe_emb, src_lang_emb, transcript_emb,
                    translate_emb, tgt_lang_emb, translation_emb,
                ], dim=0)

                S = 5 + n_audio + n_t + n_r
                lab = torch.full((S,), -100, dtype=torch.long, device=device)
                if n_t > 0:
                    lab[prompt_len : prompt_len + n_t] = transcript
                lab[prompt_len + n_t]     = self.aura.translate_start_id
                lab[prompt_len + n_t + 1] = tgt_lang_id
                if n_r > 0:
                    lab[prompt_len + n_t + 2 : prompt_len + n_t + 2 + n_r] = translation
                lab[prompt_len + n_t + 2 + n_r] = self.aura.eos_id

            else:  # st — direct speech translation
                # Layout: [BOS, TGT_LANG, audio×N, <|translate|>, translation]
                # Mirrors Aura-MT-1B SFT prompt:
                #   [BOS, TGT_LANG, "Translate ... English: <text> Swahili:\n"]
                # The <|translate|> structural token substitutes for the prose
                # "Source-lang: ... Target-lang:" delimiter — no clean prose
                # equivalent exists between audio embeddings and output text.
                translate_emb = embed_layer(
                    torch.tensor([self.aura.translate_start_id],
                                dtype=torch.long, device=device)
                )
                tgt_lang_emb_st = embed_layer(
                    torch.tensor([tgt_lang_id], dtype=torch.long, device=device)
                )
                translation = translation_ids[i, :n_r].to(device)
                translation_emb = embed_layer(translation)

                seq_emb = torch.cat([
                    bos_emb, tgt_lang_emb_st, audio_emb, translate_emb, translation_emb,
                ], dim=0)

                # length = 3 + n_audio + n_r  (BOS + TGT_LANG + audio + <|translate|> + translation)
                # Position of <|translate|> = 2 + n_audio; it predicts the first
                # translation token. translation_emb[k] predicts translation[k+1] for
                # k < n_r-1; translation_emb[n_r-1] predicts EOS.
                S = 3 + n_audio + n_r
                lab = torch.full((S,), -100, dtype=torch.long, device=device)
                translate_pos = 2 + n_audio  # position of <|translate|>
                if n_r > 0:
                    lab[translate_pos : translate_pos + n_r] = translation
                lab[translate_pos + n_r] = self.aura.eos_id

            seqs.append(seq_emb)
            labels_list.append(lab)

        # Pad to max sequence length
        S_max = max(s.size(0) for s in seqs)
        eos_emb_pad = embed_layer(
            torch.tensor([self.aura.eos_id], dtype=torch.long, device=device)
        ).squeeze(0)

        inputs_embeds = torch.stack([
            torch.cat([s, eos_emb_pad.unsqueeze(0).expand(S_max - s.size(0), -1)], dim=0)
            if s.size(0) < S_max else s
            for s in seqs
        ])

        labels = torch.stack([
            torch.cat([l, torch.full((S_max - l.size(0),), -100, dtype=torch.long, device=device)])
            if l.size(0) < S_max else l
            for l in labels_list
        ])

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
        audio_features: torch.Tensor,
        audio_lengths: torch.Tensor,
        transcript_ids: torch.Tensor,
        transcript_lengths: torch.Tensor,
        translation_ids: torch.Tensor,
        translation_lengths: torch.Tensor,
        src_language: list[str],
        tgt_language: list[str],
        task: list[str],
        ctc_labels: torch.Tensor | None = None,
        ctc_label_lengths: torch.Tensor | None = None,
        **_unused,
    ) -> dict[str, torch.Tensor]:
        device = audio_features.device

        audio_embeds, audio_lens, ctc_logits, enc_lengths = self.encode_audio(
            audio_features, audio_lengths
        )

        inputs_embeds, labels, position_ids = self._build_inputs(
            audio_embeds, audio_lens,
            transcript_ids, transcript_lengths,
            translation_ids, translation_lengths,
            src_language, tgt_language, task, device,
        )

        logits = self.aura(inputs_embeds, position_ids)

        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        ctc_loss = torch.tensor(0.0, device=device)
        if self.ctc_weight > 0.0 and ctc_logits is not None:
            if ctc_labels is None or ctc_label_lengths is None:
                log.warning("ctc_weight > 0 but ctc_labels not provided — skipping CTC loss.")
            else:
                log_probs = ctc_logits.log_softmax(dim=-1).transpose(0, 1)
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

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        audio_features: torch.Tensor,
        audio_lengths: torch.Tensor,
        src_lang: str,
        tgt_lang: str = "eng",
        task: str = "asr",
        max_new_tokens: int = 256,
    ) -> str:
        """Greedy generation.

        Args:
            src_lang: language of the audio (used in the transcript segment).
            tgt_lang: language of the translation (CoT only; ignored for ASR).
            task:     "asr" or "cot".

        For CoT the model emits <|translate|><tgt_lang> mid-stream; ASR stops
        at <|translate|> defensively.
        """
        from st.models.kvcache import KVcache

        if task not in ("asr", "cot", "st"):
            raise ValueError(f"Unknown task '{task}' (expected 'asr', 'cot', or 'st')")

        device = audio_features.device
        audio_embeds, audio_lens, _, _ = self.encode_audio(audio_features, audio_lengths)
        n_audio = int(audio_lens[0].item())

        embed_layer = self.aura.get_embed_layer()
        src_lang_id = LANG_MAP.get(src_lang, LANG_MAP["eng"])
        tgt_lang_id = LANG_MAP.get(tgt_lang, LANG_MAP["eng"])

        bos_emb = embed_layer(
            torch.tensor([self.aura.bos_id], dtype=torch.long, device=device)
        )
        audio_emb = audio_embeds[0, :n_audio]

        if task in ("asr", "cot"):
            # Prompt: [BOS, audio×N, <|transcribe|>, SRC_LANG]
            transcribe_emb = embed_layer(
                torch.tensor([self.aura.task_asr_id], dtype=torch.long, device=device)
            )
            src_lang_emb = embed_layer(
                torch.tensor([src_lang_id], dtype=torch.long, device=device)
            )
            inputs_embeds = torch.cat(
                [bos_emb, audio_emb, transcribe_emb, src_lang_emb], dim=0
            ).unsqueeze(0)
        else:  # st
            # Prompt: [BOS, TGT_LANG, audio×N, <|translate|>]
            # Model generates translation tokens until EOS.
            tgt_lang_emb_st = embed_layer(
                torch.tensor([tgt_lang_id], dtype=torch.long, device=device)
            )
            translate_emb = embed_layer(
                torch.tensor([self.aura.translate_start_id], dtype=torch.long, device=device)
            )
            inputs_embeds = torch.cat(
                [bos_emb, tgt_lang_emb_st, audio_emb, translate_emb], dim=0
            ).unsqueeze(0)

        S = inputs_embeds.size(1)
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
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # ASR: stop at <|translate|> (boundary into translation) or EOS.
        # CoT: stop only at EOS — <|translate|> is part of the expected output mid-stream.
        # ST:  stop only at EOS — <|translate|> is already in the prompt, won't appear again.
        stop_ids = {self.aura.eos_id}
        if task == "asr":
            stop_ids.add(self.aura.translate_start_id)

        for step in range(max_new_tokens):
            tok = int(next_token.item())
            if tok in stop_ids:
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

        # NOTE: skip_special_tokens=False so TRANSLATE_START survives for split_cot_output
        return self.aura.tokenizer.decode(generated, skip_special_tokens=False)

    def split_cot_output(self, text: str) -> tuple[str, str]:
        """Split CoT decoded text into (transcript, translation) on TRANSLATE_START."""
        sep = self.aura.tokenizer.decode([self.aura.translate_start_id])
        if sep in text:
            transcript, translation = text.split(sep, 1)
        else:
            transcript, translation = text, ""
        return self._strip_special_tokens(transcript).strip(), \
               self._strip_special_tokens(translation).strip()

    def _strip_special_tokens(self, text: str) -> str:
        import re
        return re.sub(r"<\|[^|>]*\|>", "", text)

    # ------------------------------------------------------------------
    # Checkpoint helpers (projector + optional LLM adapter/full)
    # ------------------------------------------------------------------

    def save_checkpoint(self, directory: str) -> None:
        import json, os
        os.makedirs(directory, exist_ok=True)

        torch.save(self.projector.state_dict(), f"{directory}/projector.pt")

        if self.aura._lora_layers is not None:
            self.aura.save_adapter(f"{directory}/lora.pt")
        elif any(p.requires_grad for p in self.aura.model.parameters()):
            self.aura.save_full(f"{directory}/llm_full.pt")

        meta = {
            "encoder_dim":  self.encoder.get_output_dim(),
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
        lora_path = f"{directory}/lora.pt"
        if os.path.exists(lora_path):
            self.aura.load_adapter(lora_path)
        llm_path = f"{directory}/llm_full.pt"
        if os.path.exists(llm_path):
            self.aura.load_full(llm_path)
        log.info(f"Checkpoint loaded ← {directory}")