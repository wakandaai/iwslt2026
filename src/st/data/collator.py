"""
Collators for speech translation training.

AuraCollator (SpeechAura, decoder-only): pads mels, tokenizes transcript and
translation into raw id tensors, and leaves all sequence assembly to
SpeechAura.forward(), which needs post-compression audio lengths to place the
audio embeddings.

NllbCollator (SpeechNLLB, encoder-decoder): pads mels and builds NLLB's own
seq2seq `labels`. NLLB assembles the decoder sequence itself, so there is no
prompt to lay out here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

log = logging.getLogger(__name__)


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
        # 1. Tokenize transcript (always) and translation (CoT only).
        keep: list[int] = []
        transcript_ids_list: list[torch.Tensor] = []
        translation_ids_list: list[torch.Tensor] = []

        for i, b in enumerate(batch):
            t_ids = self.tokenizer.encode(b["transcript"], add_special_tokens=False)
            if b["task"] in ("cot", "st"):
                tr_ids = self.tokenizer.encode(b["translation"], add_special_tokens=False)
            else:
                tr_ids = []

            if len(t_ids) + len(tr_ids) > self.max_target_tokens:
                log.debug(
                    f"Dropping sample {b.get('audio_id', i)}: "
                    f"target {len(t_ids) + len(tr_ids)} > max_target_tokens"
                )
                continue

            transcript_ids_list.append(torch.tensor(t_ids, dtype=torch.long))
            translation_ids_list.append(torch.tensor(tr_ids, dtype=torch.long))
            keep.append(i)

        if not keep:
            return None

        # 2. Pad mel features
        mel_lens = torch.tensor([batch[i]["mel_len"] for i in keep], dtype=torch.long)
        max_mel  = int(mel_lens.max().item())
        mel_pad  = torch.zeros(len(keep), max_mel, 80)
        for j, i in enumerate(keep):
            mel_pad[j, : batch[i]["mel_len"]] = batch[i]["mel"]

        # 3. Pad transcript ids
        transcript_lens = torch.tensor(
            [t.size(0) for t in transcript_ids_list], dtype=torch.long
        )
        max_t = max(int(transcript_lens.max().item()), 1)
        transcript_pad = torch.zeros(len(keep), max_t, dtype=torch.long)
        for j, t in enumerate(transcript_ids_list):
            if t.size(0) > 0:
                transcript_pad[j, : t.size(0)] = t

        # 4. Pad translation ids (zero-length for ASR samples)
        translation_lens = torch.tensor(
            [t.size(0) for t in translation_ids_list], dtype=torch.long
        )
        max_r = max(int(translation_lens.max().item()), 1)
        translation_pad = torch.zeros(len(keep), max_r, dtype=torch.long)
        for j, t in enumerate(translation_ids_list):
            if t.size(0) > 0:
                translation_pad[j, : t.size(0)] = t

        out: dict[str, Any] = {
            "audio_features":      mel_pad,
            "audio_lengths":       mel_lens,
            "transcript_ids":      transcript_pad,
            "transcript_lengths":  transcript_lens,
            "translation_ids":     translation_pad,
            "translation_lengths": translation_lens,
            "src_language":        [batch[i]["src_language"] for i in keep],
            "tgt_language":        [batch[i]["tgt_language"] for i in keep],
            "task":                [batch[i]["task"]         for i in keep],
        }

        # 5. Optional CTC labels — always built from the SOURCE transcript
        if self.vocab is not None:
            ctc_list:    list[torch.Tensor] = []
            ctc_lengths: list[int]           = []
            for i in keep:
                text = batch[i]["transcript"]
                encoded = []
                for c in text:
                    if c in self.vocab:
                        encoded.append(self.vocab[c])
                    elif " " in self.vocab:
                        encoded.append(self.vocab[" "])
                ctc_list.append(torch.tensor(encoded, dtype=torch.long))
                ctc_lengths.append(len(encoded))

            max_ctc = max(max(len(t) for t in ctc_list), 1) if ctc_list else 1
            ctc_pad = torch.zeros(len(keep), max_ctc, dtype=torch.long)
            for j, lab in enumerate(ctc_list):
                if lab.size(0) > 0:
                    ctc_pad[j, : lab.size(0)] = lab

            out["ctc_labels"]        = ctc_pad
            out["ctc_label_lengths"] = torch.tensor(ctc_lengths, dtype=torch.long)

        return out


@dataclass
class NllbCollator:
    """Collator for SpeechNLLB batches.

    Produces NLLB's native seq2seq targets. tokenizer(text_target=...) already emits
        labels = [TGT_LANG] t_1 ... t_L </s>
    and HF shifts that right off decoder_start_token_id (</s>) to build the decoder
    input — so the language token must NOT be prepended a second time here. Padding
    becomes -100 so it is ignored by the CE loss.

    transcript_input_ids (source-language encoding of the transcript) is only needed
    when the alignment loss is on; pass need_transcript=False to skip it.

    Args:
        tokenizer:         NllbTokenizerFast.
        vocab:             Optional char->id CTC vocab (adds ctc_labels).
        max_target_tokens: Drop samples whose target exceeds this token count.
        need_transcript:   Emit transcript_input_ids for the alignment loss.
    """

    tokenizer:         Any
    vocab:             dict[str, int] | None = None
    max_target_tokens: int = 256
    need_transcript:   bool = False

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any] | None:
        from st.data.nllb_lang import to_flores

        pad_id = self.tokenizer.pad_token_id

        keep:        list[int] = []
        label_list:  list[torch.Tensor] = []
        src_ids_list: list[torch.Tensor] = []

        for i, b in enumerate(batch):
            target = b["translation"] if b["task"] in ("cot", "st") else b["transcript"]
            if not target:
                continue

            # Per-sample: a batch can mix translation directions, and src_lang/tgt_lang
            # are tokenizer *state*, not call arguments.
            self.tokenizer.src_lang = to_flores(b["src_language"])
            self.tokenizer.tgt_lang = to_flores(b["tgt_language"])

            lab = self.tokenizer(text_target=target, add_special_tokens=True)["input_ids"]
            if len(lab) > self.max_target_tokens:
                log.debug(
                    f"Dropping {b.get('audio_id', i)}: target {len(lab)} tokens "
                    f"> max_target_tokens"
                )
                continue

            label_list.append(torch.tensor(lab, dtype=torch.long))
            if self.need_transcript:
                src_ids_list.append(torch.tensor(
                    self.tokenizer(b["transcript"], add_special_tokens=True)["input_ids"],
                    dtype=torch.long,
                ))
            keep.append(i)

        if not keep:
            return None

        # Mels
        mel_lens = torch.tensor([batch[i]["mel_len"] for i in keep], dtype=torch.long)
        max_mel  = int(mel_lens.max().item())
        mel_pad  = torch.zeros(len(keep), max_mel, 80)
        for j, i in enumerate(keep):
            mel_pad[j, : batch[i]["mel_len"]] = batch[i]["mel"]

        # Labels — pad with -100 so CE ignores them
        max_l  = max(t.size(0) for t in label_list)
        labels = torch.full((len(keep), max_l), -100, dtype=torch.long)
        for j, t in enumerate(label_list):
            labels[j, : t.size(0)] = t

        out: dict[str, Any] = {
            "audio_features": mel_pad,
            "audio_lengths":  mel_lens,
            "labels":         labels,
            "src_language":   [batch[i]["src_language"] for i in keep],
            "tgt_language":   [batch[i]["tgt_language"] for i in keep],
            "task":           [batch[i]["task"]         for i in keep],
        }

        if self.need_transcript:
            max_s = max(t.size(0) for t in src_ids_list)
            src_ids  = torch.full((len(keep), max_s), pad_id, dtype=torch.long)
            src_mask = torch.zeros(len(keep), max_s, dtype=torch.long)
            for j, t in enumerate(src_ids_list):
                src_ids[j, : t.size(0)]  = t
                src_mask[j, : t.size(0)] = 1
            out["transcript_input_ids"]      = src_ids
            out["transcript_attention_mask"] = src_mask

        # Optional CTC labels — always from the SOURCE transcript
        if self.vocab is not None:
            ctc_list:    list[torch.Tensor] = []
            ctc_lengths: list[int]          = []
            for i in keep:
                encoded = [
                    self.vocab[c] if c in self.vocab else self.vocab[" "]
                    for c in batch[i]["transcript"]
                    if c in self.vocab or " " in self.vocab
                ]
                ctc_list.append(torch.tensor(encoded, dtype=torch.long))
                ctc_lengths.append(len(encoded))

            max_ctc = max(max((len(t) for t in ctc_list), default=1), 1)
            ctc_pad = torch.zeros(len(keep), max_ctc, dtype=torch.long)
            for j, lab in enumerate(ctc_list):
                if lab.size(0) > 0:
                    ctc_pad[j, : lab.size(0)] = lab
            out["ctc_labels"]        = ctc_pad
            out["ctc_label_lengths"] = torch.tensor(ctc_lengths, dtype=torch.long)

        return out