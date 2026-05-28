"""
mic.py — live microphone transcription / translation.

Records from the default (or chosen) input device, builds log-mel features,
and runs them through either:

  encoder       Stage 1 CTC encoder — greedy CTC transcription. No LLM.
  speech_aura   Stage 4 SpeechAura  — ASR / CoT / direct ST via the LLM.

Recording modes (both subcommands):
  default            Press Enter to START, Enter again to STOP.
  --seconds N        Fixed-length capture of N seconds, no prompts.
  --loop             Keep capturing in a loop until Ctrl-C.

Examples:
    # Stage 1 encoder, push-Enter capture
    speech-aura-mic encoder --checkpoint encoder.pt

    # Stage 4 SpeechAura, 5-second clips on a loop, Yoruba → English ST
    speech-aura-mic speech_aura \\
        --config config.yaml --checkpoint . \\
        --src-lang yoruba --tgt-lang english --task st \\
        --seconds 5 --loop

    # Pick a specific input device
    speech-aura-mic --list-devices
    speech-aura-mic encoder --checkpoint encoder.pt --input-device 3

Requires the optional `mic` extra:  pip install "iwslt2026[mic]"
(sounddevice also needs the system PortAudio library, e.g. `apt install
libportaudio2` or `brew install portaudio`.)
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from typing import Optional

import torch

log = logging.getLogger(__name__)

TARGET_SR = 16000


# ============================================================================
# Recording
# ============================================================================

def _import_sounddevice():
    try:
        import sounddevice as sd  # noqa
        return sd
    except (ImportError, OSError) as e:
        # ImportError: package missing. OSError: PortAudio shared lib missing.
        print(
            "[error] could not load sounddevice. Install the mic extra and "
            "PortAudio:\n"
            '  pip install "iwslt2026[mic]"\n'
            "  # then the system lib, e.g.\n"
            "  sudo apt install libportaudio2   # Debian/Ubuntu\n"
            "  brew install portaudio           # macOS\n"
            f"(underlying error: {e})",
            file=sys.stderr,
        )
        sys.exit(1)


def list_devices() -> None:
    sd = _import_sounddevice()
    print(sd.query_devices())


def _to_mono_16k(audio: torch.Tensor, sr: int) -> torch.Tensor:
    """(samples,) or (samples, channels) float32 → mono (samples,) @ 16 kHz."""
    if audio.ndim > 1:
        audio = audio.mean(dim=1)
    if sr != TARGET_SR:
        import torchaudio.functional as AF
        audio = AF.resample(audio, sr, TARGET_SR)
    return audio.contiguous()


def record_fixed(seconds: float, input_device: Optional[int]) -> torch.Tensor:
    """Block for `seconds`, return mono float32 waveform at 16 kHz."""
    sd = _import_sounddevice()
    sr = TARGET_SR
    print(f"[rec] capturing {seconds:.1f}s ...", flush=True)
    frames = sd.rec(
        int(seconds * sr), samplerate=sr, channels=1,
        dtype="float32", device=input_device,
    )
    sd.wait()
    wav = torch.from_numpy(frames).squeeze(-1)
    print("[rec] done.", flush=True)
    return _to_mono_16k(wav, sr)


def record_until_enter(input_device: Optional[int]) -> torch.Tensor:
    """Press Enter to start, Enter again to stop. Returns 16 kHz mono float32."""
    sd = _import_sounddevice()
    import numpy as np

    try:
        input("[rec] press Enter to START recording ...")
    except (EOFError, KeyboardInterrupt):
        print()
        return torch.zeros(0)

    sr = TARGET_SR
    chunks: list = []
    stop = threading.Event()

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            log.debug(f"input stream status: {status}")
        chunks.append(indata.copy())

    def wait_for_enter():
        try:
            input("[rec] recording... press Enter to STOP.\n")
        except (EOFError, KeyboardInterrupt):
            pass
        stop.set()

    waiter = threading.Thread(target=wait_for_enter, daemon=True)
    waiter.start()

    with sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                        device=input_device, callback=callback):
        while not stop.is_set():
            sd.sleep(50)

    if not chunks:
        return torch.zeros(0)
    wav = torch.from_numpy(np.concatenate(chunks, axis=0)).squeeze(-1)
    dur = wav.numel() / sr
    print(f"[rec] captured {dur:.1f}s.", flush=True)
    return _to_mono_16k(wav, sr)


def capture_one(args) -> torch.Tensor:
    """Single capture honoring --seconds / interactive mode."""
    if args.seconds is not None:
        return record_fixed(args.seconds, args.input_device)
    return record_until_enter(args.input_device)


# ============================================================================
# Model runners
# ============================================================================

def _waveform_to_mel(waveform: torch.Tensor, device: torch.device):
    """Reuse the same mel front-end as file inference. Returns (mel, mel_len)."""
    from st.inference.generate import audio_to_mel
    mel = audio_to_mel(waveform, TARGET_SR).unsqueeze(0).to(device)   # (1, T, 80)
    mel_len = torch.tensor([mel.size(1)], device=device)
    return mel, mel_len


def run_encoder_loop(args) -> None:
    """Stage 1 CTC encoder: record → greedy CTC transcript."""
    from st.inference.ctc_generate import load_encoder_for_inference
    from st.training.pretrain_encoder import ctc_greedy_decode

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    encoder, vocab = load_encoder_for_inference(args.checkpoint, device)
    idx_to_char = {v: k for k, v in vocab.items()}

    def transcribe(waveform: torch.Tensor) -> None:
        if waveform.numel() == 0:
            print("[warn] empty recording, skipping.")
            return
        mel, mel_len = _waveform_to_mel(waveform, device)
        with torch.inference_mode():
            out = encoder(mel, mel_len)
            hyp = ctc_greedy_decode(
                out["ctc_logits"], out["lengths"], idx_to_char, blank_id=0,
            )[0].strip()
        _print_result(transcript=hyp)

    _drive(args, transcribe)


def run_speech_aura_loop(args) -> None:
    """Stage 4 SpeechAura: record → ASR / CoT / ST via the LLM."""
    from st.inference.generate import build_model_for_inference
    from st.utils.config import load_config

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    cfg = load_config(args.config)
    model = build_model_for_inference(cfg, args.checkpoint, device)

    def transcribe(waveform: torch.Tensor) -> None:
        if waveform.numel() == 0:
            print("[warn] empty recording, skipping.")
            return
        mel, mel_len = _waveform_to_mel(waveform, device)
        with torch.inference_mode():
            output = model.generate(
                audio_features=mel,
                audio_lengths=mel_len,
                src_lang=args.src_lang,
                tgt_lang=args.tgt_lang,
                task=args.task,
                max_new_tokens=args.max_new_tokens,
            )
        if args.task == "asr":
            _print_result(transcript=model._strip_special_tokens(output).strip())
        elif args.task == "cot":
            transcript, translation = model.split_cot_output(output)
            _print_result(transcript=transcript, translation=translation)
        else:  # st — translation only
            _print_result(translation=model._strip_special_tokens(output).strip())

    _drive(args, transcribe)


# ============================================================================
# Loop driver + output
# ============================================================================

def _drive(args, transcribe_fn) -> None:
    """Run transcribe_fn once, or forever if --loop."""
    if args.loop:
        print("[loop] Ctrl-C to quit.\n")
        try:
            while True:
                transcribe_fn(capture_one(args))
        except KeyboardInterrupt:
            print("\n[loop] bye.")
    else:
        transcribe_fn(capture_one(args))


def _print_result(transcript: str | None = None, translation: str | None = None) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    if transcript is not None:
        print(f"Transcript:  {transcript}")
    if translation is not None:
        print(f"Translation: {translation}")
    print(f"{sep}\n")


# ============================================================================
# CLI
# ============================================================================

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input-device", type=int, default=None,
                   help="Input device index (see --list-devices). Default: system default.")
    p.add_argument("--seconds", type=float, default=None,
                   help="Fixed capture length. Omit for press-Enter-to-start/stop.")
    p.add_argument("--loop", action="store_true",
                   help="Keep capturing until Ctrl-C.")
    p.add_argument("--device", default="cuda",
                   help="Compute device: cuda / cpu (default: cuda if available).")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Live microphone transcription / translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list-devices", action="store_true",
                        help="Print available audio devices and exit.")
    sub = parser.add_subparsers(dest="mode")

    p_enc = sub.add_parser("encoder", help="Stage 1 CTC encoder (no LLM)")
    p_enc.add_argument("--checkpoint", required=True,
                       help="Stage 1 encoder .pt (bundles vocab + encoder_config)")
    _add_common(p_enc)

    p_sa = sub.add_parser("speech_aura", help="Stage 4 SpeechAura (ASR / CoT / ST)")
    p_sa.add_argument("--config", required=True, help="Inference YAML (e.g. exported config.yaml)")
    p_sa.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    p_sa.add_argument("--src-lang", required=True, help="Source language (e.g. yoruba)")
    p_sa.add_argument("--tgt-lang", default="english",
                      help="Target language for cot/st (ignored for asr)")
    p_sa.add_argument("--task", default="asr", choices=["asr", "cot", "st"])
    p_sa.add_argument("--max-new-tokens", type=int, default=256)
    _add_common(p_sa)

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.mode == "encoder":
        run_encoder_loop(args)
    elif args.mode == "speech_aura":
        run_speech_aura_loop(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()