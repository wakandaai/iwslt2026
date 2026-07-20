"""
Audio projectors: map encoder hidden states → LLM embedding space.

Two variants:
  - MLPProjector:         Fast, per-frame. Good baseline.
  - TransformerProjector: Cross-frame attention. Better at merging
                          redundant frames and modelling context.

Use build_projector() to construct from a config dict.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """2-layer MLP: encoder_dim → GELU → llm_hidden.

    Matches the Qwen3-ASR projector pattern. Fast and parameter-efficient.
    Final layer zero-initialized so training starts from identity-like behavior.
    """

    def __init__(self, encoder_dim: int, llm_hidden: int):
        super().__init__()
        self.proj1 = nn.Linear(encoder_dim, llm_hidden)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(llm_hidden, llm_hidden)
        nn.init.zeros_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """lengths unused — accepted for API compatibility with TransformerProjector."""
        return self.proj2(self.act(self.proj1(x)))


class TransformerProjector(nn.Module):
    """Bidirectional Transformer over audio frames → LLM embedding space.

    Cross-frame attention lets the projector pool redundant frames and
    emphasize phoneme boundaries. More expressive than per-frame MLP,
    especially after CTC compression has already removed blanks.

    Architecture:
        encoder_dim → LayerNorm(Linear) → TransformerEncoder(pre-norm) → Linear → llm_hidden

    Output projection zero-initialized (stable training start).
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_hidden: int,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        ffn_dim = ffn_dim or llm_hidden * 4

        self.input_proj = nn.Linear(encoder_dim, llm_hidden)
        self.input_norm = nn.LayerNorm(llm_hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=llm_hidden,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # pre-norm (norm_first=True) doesn't support it
        )

        self.output_proj = nn.Linear(llm_hidden, llm_hidden)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:       (B, T, encoder_dim)
            lengths: (B,) valid frame counts — used to build padding mask.

        Returns:
            (B, T, llm_hidden)
        """
        x = self.input_norm(self.input_proj(x))

        mask = None
        if lengths is not None:
            max_len = x.size(1)
            # TransformerEncoder: True = ignore this position
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)

        x = self.transformer(x, src_key_padding_mask=mask)
        return self.output_proj(x)


def build_projector(config: dict, encoder_dim: int, llm_hidden: int) -> nn.Module:
    """Build a projector from a config dict.

    Config keys:
        type:       "mlp" | "transformer"
        num_layers: (transformer only) default 2
        num_heads:  (transformer only) default 8
        ffn_dim:    (transformer only) default llm_hidden * 4
        dropout:    (transformer only) default 0.1

    Example:
        projector:
          type: transformer
          num_layers: 2
          num_heads: 8
    """
    ptype = config.get("type", "mlp")

    if ptype == "mlp":
        proj = MLPProjector(encoder_dim, llm_hidden)
    elif ptype == "transformer":
        proj = TransformerProjector(
            encoder_dim=encoder_dim,
            llm_hidden=llm_hidden,
            num_layers=config.get("num_layers", 2),
            num_heads=config.get("num_heads", 8),
            ffn_dim=config.get("ffn_dim", None),
            dropout=config.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown projector type '{ptype}'. Choose: mlp, transformer")

    return proj
