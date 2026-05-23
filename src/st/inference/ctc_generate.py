"""
CTC inference: transcribe audio with the pretrained encoder (Stage 1).

Loads a Stage 1 encoder checkpoint (which bundles the vocab and encoder
config) and runs greedy CTC decoding. No LLM, no projector — just the
encoder's CTC head.

Usage:
    python -m st.inference.ctc_generate \
        --checkpoint runs/stage1_23_lang/encoder_step96000.pt \
        --audio test.wav

    # Beam search (requires pyctcdecode or torchaudio decoder; greedy by default)
    python -m st.inference.ctc_generate \
        --checkpoint runs/stage1_23_lang/encoder_step96000.pt \
        --audio test.wav \
        --decode greedy
"""

from __future__ import annotations

import argparse
import logging

import torch

from st.inference.generate import load_audio, audio_to_mel
from st.models.encoder import SpeechEncoder
from st.training.pretrain_encoder import ctc_greedy_decode

log = logging.getLogger(__name__)


def load_encoder_for_inference(
    checkpoint: str,
    device: torch.device,
) -> tuple[SpeechEncoder, dict[str, int]]:
    """Load a Stage 1 encoder checkpoint with its bundled vocab + config.

    Stage 1 checkpoints (from pretrain_encoder.save_checkpoint) carry
    `encoder_config` and `vocab` alongside the weights, so we don't need a
    YAML file to rebuild the encoder.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    if "encoder_config" not in ckpt or "vocab" not in ckpt:
        raise ValueError(
            f"Checkpoint {checkpoint} is missing encoder_config or vocab. "
            f"Use a Stage 1 checkpoint saved by pretrain_encoder.py."
        )

    enc_cfg = ckpt["encoder_config"]
    vocab   = ckpt["vocab"]

    encoder = SpeechEncoder(
        input_dim=enc_cfg.get("input_dim", 80),
        encoder_dim=enc_cfg.get("encoder_dim", 512),
        num_heads=enc_cfg.get("num_heads", 8),
        ffn_dim=enc_cfg.get("ffn_dim", 2048),
        num_layers=enc_cfg.get("num_layers", 12),
        depthwise_conv_kernel_size=enc_cfg.get("depthwise_conv_kernel_size", 31),
        dropout=enc_cfg.get("dropout", 0.1),
        vocab_size=len(vocab),
    )
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder = encoder.to(device).eval()

    log.info(
        f"Loaded encoder ← {checkpoint} "
        f"(step={ckpt.get('step', '?')}, vocab={len(vocab)} tokens, "
        f"params={sum(p.numel() for p in encoder.parameters()):,})"
    )
    return encoder, vocab


def run_ctc_inference(
    checkpoint: str,
    audio_path: str,
    device_str: str = "cuda",
) -> dict[str, str]:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    encoder, vocab = load_encoder_for_inference(checkpoint, device)
    idx_to_char = {v: k for k, v in vocab.items()}

    waveform, sr = load_audio(audio_path)
    mel = audio_to_mel(waveform, sr).unsqueeze(0).to(device)   # (1, T, 80)
    mel_len = torch.tensor([mel.size(1)], device=device)

    with torch.inference_mode():
        out = encoder(mel, mel_len)
        # ctc_greedy_decode argmaxes internally — pass logits directly
        hyps = ctc_greedy_decode(
            out["ctc_logits"], out["lengths"], idx_to_char, blank_id=0,
        )

    return {"transcript": hyps[0].strip()}


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="CTC inference (pretrained encoder)")
    parser.add_argument("--checkpoint", required=True,
                        help="Stage 1 encoder checkpoint .pt (must contain vocab + encoder_config)")
    parser.add_argument("--audio",      required=True, help="Audio file path")
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    output = run_ctc_inference(
        checkpoint=args.checkpoint,
        audio_path=args.audio,
        device_str=args.device,
    )

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Audio:      {args.audio}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{sep}")
    print(f"Transcript: {output['transcript']}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()