"""
Speech encoder: ConvSubsampler (4x) + torchaudio Conformer + optional CTC head.

The CTC head is kept active during ST training to support:
  - Auxiliary CTC loss (keeps encoder phonetically grounded)
  - CTCCompressor (reads ctc_logits to merge redundant frames)

Set vocab_size=None to disable the CTC head entirely.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


class ConvSubsampler(nn.Module):
    """4x convolutional subsampling on log-mel features.

    Two stride-2 Conv2d layers reduce the time dimension by 4x and project
    to encoder_dim. Standard pre-Conformer front-end.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.linear = nn.Linear(32 * (input_dim // 4), output_dim)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:       (B, T, F) log-mel features
            lengths: (B,) frame counts

        Returns:
            out:     (B, T//4, encoder_dim)
            lengths: (B,) subsampled lengths
        """
        x = x.unsqueeze(1)                          # (B, 1, T, F)
        x = self.conv(x)                             # (B, 32, T//4, F//4)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        x = self.linear(x)                           # (B, T//4, encoder_dim)
        # Applied twice deliberately: one length update per stride-2 conv layer.
        lengths = ((lengths - 1) // 2 + 1)
        lengths = ((lengths - 1) // 2 + 1)
        return x, lengths


class SpeechEncoder(nn.Module):
    """Conformer encoder with optional CTC head.

    Args:
        input_dim:                 Mel bins (default 80).
        encoder_dim:               Conformer hidden dim.
        num_heads:                 Attention heads.
        ffn_dim:                   Feed-forward dim.
        num_layers:                Conformer blocks.
        depthwise_conv_kernel_size: Depthwise conv kernel.
        dropout:                   Dropout rate.
        vocab_size:                CTC vocab size. None = no CTC head.
    """

    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        num_layers: int = 12,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.1,
        vocab_size: int | None = None,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim

        self.subsampler = ConvSubsampler(input_dim, encoder_dim)
        self.conformer = torchaudio.models.Conformer(
            input_dim=encoder_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
        )
        self.ctc_head = nn.Linear(encoder_dim, vocab_size) if vocab_size else None

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            features: (B, T, 80) log-mel spectrogram
            lengths:  (B,) mel frame counts

        Returns dict with:
            hidden_states: (B, T', encoder_dim)
            lengths:       (B,) output lengths
            ctc_logits:    (B, T', vocab_size)  — only if ctc_head exists
        """
        x, lengths = self.subsampler(features, lengths)
        x, lengths = self.conformer(x, lengths)

        out: dict[str, torch.Tensor] = {
            "hidden_states": x,
            "lengths": lengths,
        }
        if self.ctc_head is not None:
            out["ctc_logits"] = self.ctc_head(x)

        return out

    def get_output_dim(self) -> int:
        return self.encoder_dim

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


def load_encoder_from_checkpoint(
    config: dict,
    checkpoint_path: str | None = None,
    vocab_size: int | None = None,
    strict: bool = False,
) -> SpeechEncoder:
    """Build a SpeechEncoder from a config dict and optional checkpoint.

    Args:
        config:          Encoder config dict (keys match SpeechEncoder args).
        checkpoint_path: Path to a Stage-1 checkpoint (.pt).
                         Handles formats: raw state_dict, {"model_state_dict": ...},
                         {"encoder": ...}.
        vocab_size:      CTC vocab size. Override to re-attach head with a
                         different vocab after loading.
        strict:          Passed to load_state_dict (False = ignore missing ctc_head).
    """
    encoder = SpeechEncoder(
        input_dim=config.get("input_dim", 80),
        encoder_dim=config.get("encoder_dim", 512),
        num_heads=config.get("num_heads", 8),
        ffn_dim=config.get("ffn_dim", 2048),
        num_layers=config.get("num_layers", 12),
        depthwise_conv_kernel_size=config.get("depthwise_conv_kernel_size", 31),
        dropout=config.get("dropout", 0.1),
        vocab_size=vocab_size,
    )

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        # Unwrap common checkpoint formats
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "encoder" in state:
            state = state["encoder"]

        # Strip "encoder." prefix if present (some checkpoints wrap it)
        state = {
            k.replace("encoder.", "", 1) if k.startswith("encoder.") else k: v
            for k, v in state.items()
        }

        # Drop CTC head weights if we're not using a CTC head
        if vocab_size is None:
            state = {k: v for k, v in state.items() if not k.startswith("ctc_head")}

        encoder.load_state_dict(state, strict=strict)

    return encoder
