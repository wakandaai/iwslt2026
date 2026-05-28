"""
Inference: generate transcription or translation from audio.

Usage:
    python -m st.inference.generate \
        --config configs/experiment/stage3.yaml \
        --checkpoint runs/stage3/checkpoint_step20000 \
        --audio test.wav \
        --language igbo \
        --task transcribe
"""

from __future__ import annotations

import argparse
import logging

import torch
import torchaudio
import soundfile as sf

from st.utils.config import load_config

log = logging.getLogger(__name__)


def load_audio(path: str, sample_rate: int = 16000) -> tuple[torch.Tensor, int]:
    """Load audio file, convert to mono, return (waveform, sr)."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    waveform = torch.from_numpy(data)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform, sample_rate


def audio_to_mel(waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=80,
    )
    mel = mel_transform(waveform)
    mel = torch.clamp(mel, min=1e-10).log10()
    return mel.T   # (T, 80)


def build_model_for_inference(cfg: dict, checkpoint: str, device: torch.device):
    """Build SpeechAura from config and load checkpoint weights."""
    from st.training.train_st import build_model
    model = build_model(cfg).to(device)
    model.load_checkpoint(checkpoint)
    model.eval()
    return model


def run_inference(
    cfg: dict,
    checkpoint: str,
    audio_path: str,
    src_language: str,
    tgt_language: str,
    task: str,
    max_new_tokens: int = 256,
    device_str: str = "cuda",
) -> dict[str, str]:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model  = build_model_for_inference(cfg, checkpoint, device)

    waveform, sr = load_audio(audio_path)
    mel = audio_to_mel(waveform, sr).unsqueeze(0).to(device)
    mel_len = torch.tensor([mel.size(1)], device=device)

    with torch.inference_mode():
        output = model.generate(
            audio_features=mel,
            audio_lengths=mel_len,
            src_lang=src_language,
            tgt_lang=tgt_language,
            task=task,
            max_new_tokens=max_new_tokens,
        )

    if task == "asr":
        return {"transcript": model._strip_special_tokens(output).strip()}
    elif task == "cot":
        transcript, translation = model.split_cot_output(output)
        return {"transcript": transcript, "translation": translation}
    else:  # st — output is translation only, no transcript in the generation
        return {"translation": model._strip_special_tokens(output).strip()}


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="SpeechAura inference")
    parser.add_argument("--config",         required=True,  help="Experiment YAML config")
    parser.add_argument("--checkpoint",     required=True,  help="Checkpoint directory")
    parser.add_argument("--audio",          required=True,  help="Audio file path")
    parser.add_argument("--src_language",   required=True,  help="Source language code (e.g. igbo)")
    parser.add_argument("--tgt_language", default="english", help="Target language for cot/st (ignored for ASR)")
    parser.add_argument("--task",           default="asr", choices=["asr", "cot", "st"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device",         default="cuda")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    output = run_inference(
        cfg=cfg,
        checkpoint=args.checkpoint,
        audio_path=args.audio,
        src_language=args.src_language,
        tgt_language=args.tgt_language,
        task=args.task,
        max_new_tokens=args.max_new_tokens,
        device_str=args.device,
    )

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Audio:    {args.audio}")
    if args.task == "asr":
        print(f"Source: {args.src_language}  |  Task: {args.task}")
    else:
        print(f"Source: {args.src_language}  |  Target: {args.tgt_language}  |  Task: {args.task}")
    print(f"{sep}")
    if "transcript" in output:
        print(f"Transcript:  {output['transcript']}")
    if "translation" in output:
        print(f"Translation: {output['translation']}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()