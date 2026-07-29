"""
CTC-based adaptive frame compression.

Sits between the encoder and projector. Uses the encoder's CTC predictions
to merge consecutive frames predicted as the same token, producing a
content-adaptive compression of the encoder hidden states.

Flow:
    encoder output
        ├── hidden_states  (B, T, D)
        └── ctc_logits     (B, T, V)
                │
                ▼
        CTCCompressor  →  compressed hidden_states (B, T', D)
                │
                ▼
           Projector  →  LLM

Three merge strategies:
    avg:      Equal weight to all frames in a CTC segment.
    weighted: Weight by CTC posterior probability of the predicted token.
    softmax:  Softmax-normalized posteriors as weights.

Blank removal:
    With remove_blanks=True (default), segments predicted as blank are
    discarded entirely. This achieves strong compression on silence/noise
    regions. With remove_blanks=False, blanks are merged into a single
    representative frame.

Literature context:
    Auxiliary CTC loss during ST training (ctc_weight ~0.3) keeps the
    encoder's representations phonetically grounded, which in turn makes
    the CTC predictions used here more reliable.
    Ref: Chimera (Tang et al. 2021), ESPnet-ST.
"""

from __future__ import annotations

import logging
from itertools import groupby

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class CTCCompressor(nn.Module):
    """CTC-based frame compression via batched weight matrix.

    Builds a (B, T, T') weight matrix and does a single bmm, making the
    operation fully differentiable and GPU-efficient.

    Args:
        strategy:      "avg" | "weighted" | "softmax"
        blank_id:      CTC blank token index (default 0).
        remove_blanks: Drop blank segments from output (default True).
    """

    def __init__(
        self,
        strategy: str = "avg",
        blank_id: int = 0,
        remove_blanks: bool = True,
    ):
        super().__init__()
        if strategy not in ("avg", "weighted", "softmax"):
            raise ValueError(f"Unknown strategy '{strategy}'. Choose: avg, weighted, softmax")
        self.strategy = strategy
        self.blank_id = blank_id
        self.remove_blanks = remove_blanks

    def _build_segments(
        self,
        predictions: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[list[list[tuple[int, int, int]]], list[int]]:
        """Group consecutive same-token frames into (token_id, offset, count) segments.

        Args:
            predictions: (B, T) predicted token id per frame (argmax over vocab).
            lengths:     (B,) valid frame counts per utterance.

        Returns:
            batch_segments: per-utterance list of (token_id, offset, count) segments.
            new_lengths:    per-utterance segment count.
        """
        B = predictions.size(0)
        batch_segments: list[list[tuple[int, int, int]]] = []
        new_lengths: list[int] = []

        for b in range(B):
            seq_len = int(lengths[b].item())
            pred_seq = predictions[b, :seq_len].tolist()

            segments = []
            offset = 0
            for token_id, group in groupby(pred_seq):
                count = sum(1 for _ in group)
                if self.remove_blanks and token_id == self.blank_id:
                    offset += count
                    continue
                segments.append((token_id, offset, count))
                offset += count

            # Edge case: all frames were blank
            if not segments:
                segments = [(self.blank_id, 0, min(1, seq_len))]

            batch_segments.append(segments)
            new_lengths.append(len(segments))

        return batch_segments, new_lengths

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctc_logits: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, T, D) encoder hidden states
            ctc_logits:    (B, T, V) CTC logits (pre-softmax)
            lengths:       (B,) valid frame counts per utterance

        Returns:
            compressed:    (B, T', D) compressed hidden states (zero-padded)
            new_lengths:   (B,) compressed lengths
        """
        B, T, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        ctc_probs = ctc_logits.softmax(dim=-1)   # (B, T, V)
        predictions = ctc_logits.argmax(dim=-1)  # (B, T)

        batch_segments, new_lengths = self._build_segments(predictions, lengths)
        new_max_len = max(new_lengths)

        # Build weight matrix (B, T, T') — sparse by construction
        weights = torch.zeros(B, T, new_max_len, dtype=dtype, device=device)

        for b, segments in enumerate(batch_segments):
            for seg_idx, (token_id, offset, count) in enumerate(segments):
                end = min(offset + count, int(lengths[b].item()))
                actual_count = end - offset
                if actual_count <= 0:
                    continue

                if self.strategy == "avg":
                    weights[b, offset:end, seg_idx] = 1.0 / actual_count

                elif self.strategy == "weighted":
                    seg_probs = ctc_probs[b, offset:end, token_id]
                    w = seg_probs / (seg_probs.sum() + 1e-10)
                    weights[b, offset:end, seg_idx] = w

                elif self.strategy == "softmax":
                    seg_probs = ctc_probs[b, offset:end, token_id]
                    weights[b, offset:end, seg_idx] = F.softmax(seg_probs, dim=0)

        # Batched matmul: (B, T', T) @ (B, T, D) = (B, T', D)
        compressed = torch.bmm(weights.transpose(1, 2), hidden_states)
        new_lengths_t = torch.tensor(new_lengths, dtype=torch.long, device=device)

        return compressed, new_lengths_t

    def forward_cached(
        self,
        hidden_states: torch.Tensor,
        predicted_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same as forward(), but takes precomputed argmax predictions directly
        instead of deriving them from ctc_logits — for use with cached encoder
        features (see CachedFeatureDataset) where only predicted_ids were
        stored, not the full per-frame logits/probabilities.

        Only supports strategy="avg": "weighted"/"softmax" need the real
        per-frame CTC posterior probability of the predicted token, which
        isn't available here by design (storing full ctc_logits would be a
        ~7x storage blowup vs storing just the argmax id per frame, and no
        experiment config in this repo uses anything but "avg").

        Args:
            hidden_states:  (B, T, D) cached encoder hidden states
            predicted_ids:  (B, T) cached argmax CTC predictions per frame
            lengths:        (B,) valid frame counts per utterance

        Returns:
            compressed:  (B, T', D) compressed hidden states (zero-padded)
            new_lengths: (B,) compressed lengths
        """
        if self.strategy != "avg":
            raise ValueError(
                f"CTCCompressor.forward_cached only supports strategy='avg' "
                f"(got '{self.strategy}') — cached features only store argmax "
                f"predicted_ids, not the per-frame probabilities 'weighted'/'softmax' need."
            )

        B, T, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        batch_segments, new_lengths = self._build_segments(predicted_ids, lengths)
        new_max_len = max(new_lengths)

        weights = torch.zeros(B, T, new_max_len, dtype=dtype, device=device)
        for b, segments in enumerate(batch_segments):
            for seg_idx, (_token_id, offset, count) in enumerate(segments):
                end = min(offset + count, int(lengths[b].item()))
                actual_count = end - offset
                if actual_count <= 0:
                    continue
                weights[b, offset:end, seg_idx] = 1.0 / actual_count

        compressed = torch.bmm(weights.transpose(1, 2), hidden_states)
        new_lengths_t = torch.tensor(new_lengths, dtype=torch.long, device=device)

        return compressed, new_lengths_t

    def compression_ratio(self, lengths: torch.Tensor, new_lengths: torch.Tensor) -> float:
        """Mean compression ratio across the batch. Useful for logging."""
        return (new_lengths.float() / lengths.float().clamp(min=1)).mean().item()


def build_ctc_compressor(config: dict | None) -> CTCCompressor | None:
    """Build a CTCCompressor from config, or None if disabled.

    Config format:
        ctc_compress:
          enabled: true
          strategy: avg       # avg | weighted | softmax
          remove_blanks: true
          blank_id: 0

    Returns None if config is None or enabled=False.
    """
    if config is None:
        return None
    if not config.get("enabled", True):
        return None

    return CTCCompressor(
        strategy=config.get("strategy", "avg"),
        blank_id=config.get("blank_id", 0),
        remove_blanks=config.get("remove_blanks", True),
    )
