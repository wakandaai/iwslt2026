"""
Collator for SpeechAuraTTS training.

Like AuraCollator (ST side) it stays dumb about the model: it tokenizes text,
pads the variable-length (T, K) DAC code grids, and stacks speaker vectors. The
prompt/sequence assembly (BOS, speaker token, SPEECH_START, frame embeddings)
happens inside SpeechAuraTTS._build_inputs.

Code padding uses id 0; padded frames sit past code_lengths and are masked out
of both the depth loss and the stop loss in forward(), so the pad value is
irrelevant to the gradient.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

log = logging.getLogger(__name__)


@dataclass
class TTSCollator:
    """Collator for SpeechAuraTTS batches.

    Args:
        tokenizer:        Aura tokenizer (PreTrainedTokenizerFast).
        max_text_tokens:  Drop samples whose tokenized text exceeds this.
        max_frames:       Drop samples with more DAC frames than this. None = no cap.
                          (Primary length control is the sampler; this is a guard.)
    """

    tokenizer:       Any
    max_text_tokens: int = 256
    max_frames:      int | None = None

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any] | None:
        keep: list[int] = []
        text_ids_list: list[torch.Tensor] = []

        for i, b in enumerate(batch):
            ids = self.tokenizer.encode(b["text"], add_special_tokens=False)
            if len(ids) == 0 or len(ids) > self.max_text_tokens:
                continue
            if self.max_frames is not None and b["frame_count"] > self.max_frames:
                continue
            text_ids_list.append(torch.tensor(ids, dtype=torch.long))
            keep.append(i)

        if not keep:
            return None

        K = batch[keep[0]]["codes"].size(1)

        # ---- Pad DAC codes → (B, T_max, K) ----
        code_lengths = torch.tensor(
            [batch[i]["frame_count"] for i in keep], dtype=torch.long)
        T_max = int(code_lengths.max().item())
        codes = torch.zeros(len(keep), T_max, K, dtype=torch.long)
        for j, i in enumerate(keep):
            c = batch[i]["codes"]
            codes[j, : c.size(0)] = c

        # ---- Pad text ids → (B, L_max) ----
        text_lengths = torch.tensor([t.size(0) for t in text_ids_list], dtype=torch.long)
        L_max = int(text_lengths.max().item())
        text_ids = torch.zeros(len(keep), L_max, dtype=torch.long)
        for j, t in enumerate(text_ids_list):
            text_ids[j, : t.size(0)] = t

        # ---- Stack speaker vectors → (B, spk_dim) ----
        speaker_vecs = torch.stack([batch[i]["speaker_vec"] for i in keep], dim=0)

        return {
            "codes":        codes,
            "code_lengths": code_lengths,
            "speaker_vecs": speaker_vecs,
            "text_ids":     text_ids,
            "text_lengths": text_lengths,
            "languages":    [batch[i]["language"] for i in keep],
            "audio_id":     [batch[i]["audio_id"] for i in keep],
        }


@dataclass
class ContinuationCollator:
    """Collator for Stage-0 speech continuation (unconditional next-frame).

    Strips text and speaker — emits only the padded (T, K) DAC grids and the
    language, the inputs SpeechAuraTTS.forward(conditioning=False) needs. The
    same TTSDataset (and its index CSV) feeds this; text/speaker fields in each
    example are simply ignored.

    Overflow handling mirrors TTSCollator: utterances longer than max_frames are
    DROPPED, not cropped. A whole utterance keeps its true end, so the synthetic
    EOS frame appended in forward() always marks a real stop — cropping a window
    out of the middle would teach the depth transformer to halt mid-utterance.
    If long files dominate the data, raise the Aura context (max_seq_len) instead
    of cropping. Returns None if nothing in the batch survives the guard.

    Args:
        max_frames: Drop samples with more DAC frames than this. None = no cap
                    (the DurationBucketSampler is the primary length control).
    """

    max_frames: int | None = None

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any] | None:
        keep = [
            i for i, b in enumerate(batch)
            if self.max_frames is None or b["frame_count"] <= self.max_frames
        ]
        if not keep:
            return None

        K = batch[keep[0]]["codes"].size(1)

        code_lengths = torch.tensor(
            [batch[i]["frame_count"] for i in keep], dtype=torch.long)
        T_max = int(code_lengths.max().item())
        codes = torch.zeros(len(keep), T_max, K, dtype=torch.long)
        for j, i in enumerate(keep):
            c = batch[i]["codes"]
            codes[j, : c.size(0)] = c

        return {
            "codes":        codes,
            "code_lengths": code_lengths,
            "languages":    [batch[i]["language"] for i in keep],
            "audio_id":     [batch[i]["audio_id"] for i in keep],
            # The batch dict is SpeechAuraTTS.forward()'s kwargs by contract, so
            # the stage selector rides along here: model(**batch) → the
            # unconditional [BOS, LANG, SPEECH_START, frame×T] prefix.
            "conditioning": False,
        }
