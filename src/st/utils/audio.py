"""Audio processing utilities."""

from __future__ import annotations

import torch
import torchaudio


def build_feature_extractor(
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    win_length: int = 400,
) -> torch.nn.Module:
    """Build a log-mel spectrogram feature extractor.

    Default settings: 25ms window, 10ms hop at 16kHz, 80 mel bins.

    Returns a module: (T_samples,) → (n_mels, T_frames).
    Transpose to get (T_frames, n_mels) for the encoder.
    """
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
    )

    class LogMelExtractor(torch.nn.Module):
        def __init__(self, mel_transform):
            super().__init__()
            self.mel = mel_transform

        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            return torch.clamp(self.mel(waveform), min=1e-10).log10()

    return LogMelExtractor(mel)
