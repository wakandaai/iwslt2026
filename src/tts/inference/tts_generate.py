"""
Inference: synthesize speech from text + a reference speaker.

Pipeline:
    text ──tokenize──► SpeechAuraTTS.generate ──► DAC codes (T,K)
                                                      │
                          reference wav ──RawNet3──► speaker vector (192-d)
                                                      │
                              DAC.decode(codes) ──► waveform ──► .wav

The model is rebuilt from the TTS checkpoint's meta.json (codec + depth dims)
plus the Aura paths in the config — meta.json is the source of truth for
architecture, so no train/inference config drift can corrupt the load.

Speaker conditioning comes from one of:
  --ref_wav   a reference clip → RawNet3 embedding (needs --spk_model_dir)
  --ref_npy   a precomputed (192,) .npy speaker vector
  --ref_id    an audio_id to look up in a SpeakerStore cache (--spk_cache)

Usage:
    python -m tts.inference.tts_generate \
        --config configs/experiment/tts_stage1.yaml \
        --checkpoint runs/tts_stage1/checkpoint_step20000 \
        --text "Habari ya asubuhi." --language swahili \
        --ref_wav ref_speaker.wav --spk_model_dir ./models/voxcelebs12_rawnet3 \
        --out out.wav
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from core.aura import AuraLLM
from core.utils.config import load_config
from tts.models import SpeechAuraTTS

log = logging.getLogger(__name__)

TARGET_SR = 16000


# ============================================================================
# Model
# ============================================================================

def build_model_for_inference(
    cfg: dict, checkpoint: str, device: torch.device,
) -> SpeechAuraTTS:
    """Rebuild SpeechAuraTTS from the checkpoint's meta.json + config Aura paths."""
    meta = json.loads((Path(checkpoint) / "meta.json").read_text())
    aura_cfg = cfg["aura"]

    aura = AuraLLM(
        ckpt_path=aura_cfg["checkpoint"],
        tokenizer_path=aura_cfg["tokenizer"],
        size=aura_cfg.get("size", "1b"),
        freeze=True,
        lora_rank=cfg["training"].get("lora_rank", 0) if meta.get("has_lora") else 0,
        lora_alpha=cfg["training"].get("lora_alpha", 32),
        lora_targets=cfg["training"].get("lora_targets", ["q_proj", "v_proj"]),
    )
    model = SpeechAuraTTS(
        aura=aura,
        n_codebooks=meta["n_codebooks"],
        cardinality=meta["cardinality"],
        speaker_dim=meta["speaker_dim"],
        depth_dim=meta["depth_dim"],
        depth_layers=meta["depth_layers"],
        depth_heads=meta["depth_heads"],
        freeze_llm=True,
    ).to(device)
    model.load_checkpoint(checkpoint)
    model.eval()
    return model


# ============================================================================
# Speaker conditioning
# ============================================================================

def load_ref_vector(args, device: torch.device) -> torch.Tensor:
    """Resolve the (speaker_dim,) conditioning vector from the chosen source."""
    if args.ref_npy:
        vec = np.load(args.ref_npy).astype(np.float32).reshape(-1)
        return torch.from_numpy(vec).to(device)

    if args.ref_id:
        from tts.data import SpeakerStore
        store = SpeakerStore(args.spk_cache)
        if args.ref_id not in store:
            raise KeyError(f"{args.ref_id!r} not in speaker cache {args.spk_cache}")
        return store[args.ref_id].to(device)

    if args.ref_wav:
        return _embed_ref_wav(args.ref_wav, args.spk_model_dir, device)

    raise ValueError("Provide one of --ref_wav, --ref_npy, or --ref_id.")


def _embed_ref_wav(path: str, spk_model_dir: str, device: torch.device) -> torch.Tensor:
    """RawNet3 embedding of a reference clip (matches precompute_spk_embeddings)."""
    import soundfile as sf
    import torchaudio.functional as AF
    from espnet2.bin.spk_inference import Speech2Embedding

    if not spk_model_dir:
        raise ValueError("--ref_wav requires --spk_model_dir (RawNet3 snapshot).")

    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    wav = torch.from_numpy(np.ascontiguousarray(data))
    if sr != TARGET_SR:
        wav = AF.resample(wav, sr, TARGET_SR)
    wav = wav.numpy().astype(np.float32, copy=False)

    mdir = Path(spk_model_dir)
    config = next(mdir.rglob("config.yaml"))
    weights = next(mdir.rglob("*.pth"))
    spk_model = Speech2Embedding(
        train_config=str(config), model_file=str(weights), device=str(device))
    with torch.no_grad():
        emb = spk_model(wav).squeeze(0).detach().to(device).float()
    return emb


# ============================================================================
# DAC decode
# ============================================================================

def load_dac(model_type: str, device: torch.device):
    import dac
    model = dac.DAC.load(dac.utils.download(model_type=model_type)).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.inference_mode()
def decode_codes(dac_model, codes_TK: torch.Tensor, device: torch.device) -> torch.Tensor:
    """(T, K) codes → 1-D waveform tensor on CPU via DAC.quantizer + decode."""
    # DAC expects (B, K, T) long codes.
    codes = codes_TK.t().unsqueeze(0).to(device).long()      # (1, K, T)
    out = dac_model.quantizer.from_codes(codes)
    z = out[0] if isinstance(out, (tuple, list)) else out    # z_q (version-tolerant)
    wav = dac_model.decode(z)                                # (1, 1, samples)
    return wav.squeeze().detach().float().cpu()


# ============================================================================
# Driver
# ============================================================================

def run(args) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    model = build_model_for_inference(cfg, args.checkpoint, device)
    spk_vec = load_ref_vector(args, device)

    text_ids = torch.tensor(
        model.aura.tokenizer.encode(args.text, add_special_tokens=False),
        dtype=torch.long, device=device,
    )
    if text_ids.numel() == 0:
        raise ValueError("Text tokenized to zero tokens.")

    codes = model.generate(
        speaker_vec=spk_vec,
        text_ids=text_ids,
        language=args.language,
        max_frames=args.max_frames,
        # top_k/top_p only matter when sampling, so requesting either turns
        # sampling on (greedy ignores them and reliably collapses to silence).
        greedy=not (args.sample or args.top_k or args.top_p),
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    log.info(f"Generated {codes.size(0)} frames × {codes.size(1)} codebooks")
    if codes.size(0) == 0:
        raise RuntimeError("Model emitted 0 frames (EOS decoded immediately).")

    dac_model = load_dac(args.dac_model_type, device)
    wav = decode_codes(dac_model, codes, device)

    import soundfile as sf
    sf.write(args.out, wav.numpy(), dac_model.sample_rate)
    log.info(f"Wrote {wav.numel() / dac_model.sample_rate:.2f}s → {args.out} "
             f"(sr={dac_model.sample_rate})")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="SpeechAuraTTS synthesis")
    p.add_argument("--config",     required=True, help="Experiment YAML (Aura paths)")
    p.add_argument("--checkpoint", required=True, help="TTS checkpoint directory")
    p.add_argument("--text",       required=True, help="Text to synthesize")
    p.add_argument("--language",   required=True, help="Language code/name (e.g. swahili)")
    p.add_argument("--out",        default="out.wav", help="Output wav path")

    # Speaker source (exactly one).
    p.add_argument("--ref_wav", default=None, help="Reference clip → RawNet3 embedding")
    p.add_argument("--ref_npy", default=None, help="Precomputed (192,) speaker vector .npy")
    p.add_argument("--ref_id",  default=None, help="audio_id to look up in --spk_cache")
    p.add_argument("--spk_model_dir", default=None, help="RawNet3 snapshot dir (for --ref_wav)")
    p.add_argument("--spk_cache",     default=None, help="SpeakerStore dir (for --ref_id)")

    # Decode controls.
    p.add_argument("--dac_model_type", default="16khz", help="DAC model type")
    p.add_argument("--max_frames",     type=int,   default=1500)
    p.add_argument("--sample",         action="store_true", help="Sample instead of greedy")
    p.add_argument("--temperature",    type=float, default=1.0)
    p.add_argument("--top_k",          type=int,   default=None,
                   help="Top-k truncation before sampling (e.g. 100). Implies --sample.")
    p.add_argument("--top_p",          type=float, default=None,
                   help="Nucleus (top-p) truncation before sampling (e.g. 0.9). Implies --sample.")
    p.add_argument("--device",         default="cuda")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
