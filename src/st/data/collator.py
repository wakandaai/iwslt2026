"""
Collator for SpeechAura training.

Intentionally simple — the collator does NOT know about the encoder,
compressor, or LLM token format. It just:
  1. Pads mel features
  2. Tokenizes target text → target_ids
  3. Optionally encodes CTC labels

Sequence assembly (input_ids, labels, audio placeholders) happens inside
SpeechAura.forward() after encoding, when actual post-compression lengths
are known.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import torch

log = logging.getLogger(__name__)

# The omniASR_CTC_1B SentencePiece vocab has no tokens for these characters —
# encoding them maps to <unk>, which corrupts both CTC training targets and
# WER references (decodes back as a literal "⁇" placeholder word). Strip them
# before CTC tokenization only; "-" and "'" are real vocab pieces and are left
# alone. Aura-1B's own tokenizer (used elsewhere in this file) has full
# punctuation support, so this must NOT be applied to its targets.
_CTC_UNSUPPORTED_PUNCT_RE = re.compile(r"[,.;:!?]")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_ctc_unsupported_punct(text: str) -> str:
    text = _CTC_UNSUPPORTED_PUNCT_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _tokenize_targets(
    batch: list[dict[str, Any]],
    tokenizer: Any,
    max_target_tokens: int,
) -> tuple[list[int], list[torch.Tensor]]:
    """Tokenize batch[i]['text'] for each sample, dropping ones whose target
    exceeds max_target_tokens. Shared between AuraCollator and
    CachedFeatureCollator — target tokenization is orthogonal to whether
    audio is mel or cached encoder features.

    Returns:
        keep:            indices into `batch` that were kept.
        target_ids_list: tokenized target ids for kept samples, same order as `keep`.
    """
    keep: list[int] = []
    target_ids_list: list[torch.Tensor] = []

    for i, b in enumerate(batch):
        ids = tokenizer.encode(b["text"], add_special_tokens=False)
        if len(ids) > max_target_tokens:
            log.debug(
                f"Dropping sample {b.get('audio_id', i)}: "
                f"target {len(ids)} tokens > max_target_tokens={max_target_tokens}"
            )
            continue
        target_ids_list.append(torch.tensor(ids, dtype=torch.long))
        keep.append(i)

    return keep, target_ids_list


def _pad_target_ids(target_ids_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of 1D target-id tensors to (B, L_target). Returns (padded, lengths)."""
    target_lens = torch.tensor([t.size(0) for t in target_ids_list], dtype=torch.long)
    max_target  = int(target_lens.max().item())
    target_pad  = torch.zeros(len(target_ids_list), max_target, dtype=torch.long)
    for j, t in enumerate(target_ids_list):
        target_pad[j, : t.size(0)] = t
    return target_pad, target_lens


@dataclass
class AuraCollator:
    """Collator for SpeechAura batches.

    Args:
        tokenizer:         Aura tokenizer (PreTrainedTokenizerFast).
        vocab:             Optional char->id CTC vocab. If provided, also
                           returns ctc_labels and ctc_label_lengths.
        max_target_tokens: Drop samples whose target exceeds this token count.
    """

    tokenizer:         Any
    vocab:             dict[str, int] | None = None
    max_target_tokens: int = 256

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any] | None:
        """
        Args:
            batch: List of dicts from SpeechDataset.__getitem__().

        Returns:
            Dict with tensors + language list, or None if all samples dropped.
        """
        # 1. Tokenize targets, drop samples that are too long
        keep, target_ids_list = _tokenize_targets(batch, self.tokenizer, self.max_target_tokens)

        if not keep:
            return None

        # 2. Pad mel features
        mel_lens = torch.tensor([batch[i]["mel_len"] for i in keep], dtype=torch.long)
        max_mel  = int(mel_lens.max().item())
        mel_pad  = torch.zeros(len(keep), max_mel, 80)
        for j, i in enumerate(keep):
            b = batch[i]
            mel_pad[j, : b["mel_len"]] = b["mel"]

        # 3. Pad target token ids
        target_pad, target_lens = _pad_target_ids(target_ids_list)

        out: dict[str, Any] = {
            "audio_features": mel_pad,               # (B, T_mel, 80)
            "audio_lengths":  mel_lens,               # (B,)
            "target_ids":     target_pad,             # (B, L_target)
            "target_lengths": target_lens,            # (B,)
            "language":       [batch[i]["language"] for i in keep],
        }

        # 4. Optional CTC labels
        if self.vocab is not None:
            ctc_list:    list[torch.Tensor] = []
            ctc_lengths: list[int]           = []
            for i in keep:
                text    = batch[i]["text"]
                encoded = []
                for c in text:
                    if c in self.vocab:
                        encoded.append(self.vocab[c])
                    elif " " in self.vocab:
                        encoded.append(self.vocab[" "])
                # if neither the character nor space exists, omit it to avoid crashing the embedder
                ctc_list.append(torch.tensor(encoded, dtype=torch.long))
                ctc_lengths.append(len(encoded))

            max_ctc = max(len(t) for t in ctc_list) if ctc_list else 0
            ctc_pad = torch.zeros(len(keep), max_ctc, dtype=torch.long)
            for j, lab in enumerate(ctc_list):
                ctc_pad[j, : lab.size(0)] = lab

            out["ctc_labels"]        = ctc_pad
            out["ctc_label_lengths"] = torch.tensor(ctc_lengths, dtype=torch.long)

        return out


@dataclass
class CachedFeatureCollator:
    """Collator for CachedFeatureDataset batches (precomputed omniASR_CTC_1B
    encoder features instead of raw audio + mel). Pads hidden_states/
    predicted_ids instead of mel; never produces ctc_labels since cached-mode
    training always runs with ctc_weight=0.0 (see SpeechAura.forward_cached —
    there are no real per-frame ctc_logits to supervise an aux CTC loss from,
    only the argmax predicted_ids needed by CTCCompressor.forward_cached).

    Args:
        tokenizer:         Aura tokenizer (PreTrainedTokenizerFast).
        max_target_tokens: Drop samples whose target exceeds this token count.
    """

    tokenizer:         Any
    max_target_tokens: int = 256

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any] | None:
        """
        Args:
            batch: List of dicts from CachedFeatureDataset.__getitem__().

        Returns:
            Dict with tensors + language list, or None if all samples dropped.
        """
        keep, target_ids_list = _tokenize_targets(batch, self.tokenizer, self.max_target_tokens)

        if not keep:
            return None

        # Pad cached hidden_states / predicted_ids
        feature_lens = torch.tensor([batch[i]["feature_len"] for i in keep], dtype=torch.long)
        max_len = int(feature_lens.max().item())
        hidden_dim = batch[keep[0]]["hidden_states"].size(-1)

        hidden_pad = torch.zeros(len(keep), max_len, hidden_dim, dtype=torch.float16)
        pred_pad   = torch.zeros(len(keep), max_len, dtype=torch.long)
        for j, i in enumerate(keep):
            b = batch[i]
            n = b["feature_len"]
            hidden_pad[j, :n] = b["hidden_states"]
            pred_pad[j, :n]   = b["predicted_ids"].long()

        target_pad, target_lens = _pad_target_ids(target_ids_list)

        return {
            "encoder_hidden_states": hidden_pad,                       # (B, T, D) fp16
            "encoder_lengths":       feature_lens,                     # (B,)
            "ctc_predicted_ids":     pred_pad,                         # (B, T)
            "target_ids":            target_pad,                       # (B, L_target)
            "target_lengths":        target_lens,                      # (B,)
            "language":              [batch[i]["language"] for i in keep],
        }


@dataclass
class RawAudioCollator:
    """Collator for RawAudioDataset batches (live/unfrozen omniASR_CTC_1B
    encoder — see src/st/models/omniasr_encoder.py). Pads raw waveforms
    instead of mel; output keys deliberately match AuraCollator's naming
    exactly ("audio_features"/"audio_lengths") even though the tensor now
    holds a raw waveform, not mel — SpeechAura.encode_audio()/forward() are
    generic about what "features" contains, so this lets train_st.py's
    existing non-cached run_forward() branch work completely unchanged.
    Never produces ctc_labels: this mode always runs with ctc_weight=0.0
    (omniASR's own CTC head outputs over a 9812-piece SentencePiece vocab,
    incompatible with this project's char-level CTC vocab).

    Args:
        tokenizer:         Aura tokenizer (PreTrainedTokenizerFast).
        max_target_tokens: Drop samples whose target exceeds this token count.
    """

    tokenizer:         Any
    max_target_tokens: int = 256

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any] | None:
        """
        Args:
            batch: List of dicts from RawAudioDataset.__getitem__().

        Returns:
            Dict with tensors + language list, or None if all samples dropped.
        """
        keep, target_ids_list = _tokenize_targets(batch, self.tokenizer, self.max_target_tokens)

        if not keep:
            return None

        wave_lens = torch.tensor([batch[i]["waveform_len"] for i in keep], dtype=torch.long)
        max_len   = int(wave_lens.max().item())
        wave_pad  = torch.zeros(len(keep), max_len)
        for j, i in enumerate(keep):
            b = batch[i]
            wave_pad[j, : b["waveform_len"]] = b["waveform"]

        target_pad, target_lens = _pad_target_ids(target_ids_list)

        return {
            "audio_features": wave_pad,                                # (B, T_samples)
            "audio_lengths":  wave_lens,                                # (B,)
            "target_ids":     target_pad,                               # (B, L_target)
            "target_lengths": target_lens,                              # (B,)
            "language":       [batch[i]["language"] for i in keep],
        }


@dataclass
class RawAudioAuxCTCCollator:
    """Same as RawAudioCollator, plus proj_ctc_labels for SpeechAura's
    aux_ctc_weight (projector CTC head) — separate from ctc_weight, which
    supervises the encoder's own CTC head and is off for omniasr_live.

    proj_ctc_labels use the same SentencePiece tokenizer (omniASR_tokenizer.model)
    as the encoder's own pretrained CTC head.

    Kept separate from RawAudioCollator so configs with aux_ctc_weight=0 skip
    the extra tokenization cost.

    Args:
        tokenizer:         Aura tokenizer — for target_ids.
        sp_tokenizer:      SentencePieceProcessor — for proj_ctc_labels.
        max_target_tokens: Drop samples whose Aura target exceeds this token count.
    """

    tokenizer:         Any
    sp_tokenizer:      Any
    max_target_tokens: int = 256

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any] | None:
        keep, target_ids_list = _tokenize_targets(batch, self.tokenizer, self.max_target_tokens)

        if not keep:
            return None

        wave_lens = torch.tensor([batch[i]["waveform_len"] for i in keep], dtype=torch.long)
        max_len   = int(wave_lens.max().item())
        wave_pad  = torch.zeros(len(keep), max_len)
        for j, i in enumerate(keep):
            b = batch[i]
            wave_pad[j, : b["waveform_len"]] = b["waveform"]

        target_pad, target_lens = _pad_target_ids(target_ids_list)

        proj_ctc_list: list[torch.Tensor] = []
        for i in keep:
            text = _strip_ctc_unsupported_punct(batch[i]["text"])
            ids  = self.sp_tokenizer.encode(text, out_type=int)
            proj_ctc_list.append(torch.tensor(ids, dtype=torch.long))
        proj_ctc_pad, proj_ctc_lens = _pad_target_ids(proj_ctc_list)

        return {
            "audio_features": wave_pad,                                # (B, T_samples)
            "audio_lengths":  wave_lens,                                # (B,)
            "target_ids":     target_pad,                               # (B, L_target)
            "target_lengths": target_lens,                              # (B,)
            "language":       [batch[i]["language"] for i in keep],
            "proj_ctc_labels":        proj_ctc_pad,                     # (B, L_ctc)
            "proj_ctc_label_lengths": proj_ctc_lens,                    # (B,)
        }


@dataclass
class CTCRawAudioCollator:
    """Collator for RawAudioDataset batches feeding a standalone CTC loss
    against omniASR_CTC_1B's own 9812-piece SentencePiece vocab (Stage 1
    encoder-only pretraining — see st/training/pretrain_omniasr_ctc.py).

    Unlike RawAudioCollator (which tokenizes targets with Aura's BPE
    tokenizer for the downstream LLM), this tokenizes with the SentencePiece
    processor that matches OmniASREncoder's ctc_head output space, and
    produces ctc_labels/ctc_label_lengths instead of target_ids/target_lengths
    — there's no LLM in this training path at all.

    Args:
        sp_tokenizer:      Loaded `sentencepiece.SentencePieceProcessor` for
                           omniASR_tokenizer.model (9812 pieces).
        max_target_tokens: Drop samples whose CTC label sequence exceeds this length.
    """

    sp_tokenizer:      Any
    max_target_tokens: int = 400

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any] | None:
        keep: list[int] = []
        label_list: list[torch.Tensor] = []

        for i, b in enumerate(batch):
            text = _strip_ctc_unsupported_punct(b["text"])
            ids = self.sp_tokenizer.encode(text, out_type=int)
            if not ids or len(ids) > self.max_target_tokens:
                continue
            label_list.append(torch.tensor(ids, dtype=torch.long))
            keep.append(i)

        if not keep:
            return None

        wave_lens = torch.tensor([batch[i]["waveform_len"] for i in keep], dtype=torch.long)
        max_len   = int(wave_lens.max().item())
        wave_pad  = torch.zeros(len(keep), max_len)
        for j, i in enumerate(keep):
            b = batch[i]
            wave_pad[j, : b["waveform_len"]] = b["waveform"]

        label_lens = torch.tensor([t.size(0) for t in label_list], dtype=torch.long)
        max_lab    = int(label_lens.max().item())
        label_pad  = torch.zeros(len(keep), max_lab, dtype=torch.long)
        for j, lab in enumerate(label_list):
            label_pad[j, : lab.size(0)] = lab

        return {
            "audio_features":  wave_pad,                                # (B, T_samples)
            "audio_lengths":   wave_lens,                                # (B,)
            "ctc_labels":      label_pad,                                # (B, L_label)
            "ctc_label_lengths": label_lens,                             # (B,)
            "language":        [batch[i]["language"] for i in keep],
        }