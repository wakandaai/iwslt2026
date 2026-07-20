"""
One-shot smoke test for the cached-features (omniASR_CTC_1B) SpeechAura path.

Builds SpeechAura(encoder=None) from a stage2 cached-features config,
pulls one real batch from CachedFeatureDataset/CachedFeatureCollator, and runs it
through forward_cached(). Confirms: finite loss, sane shapes, sane CTCCompressor
compression ratio. Run via the main training env (Aura_base/env), not the
isolated omniasr_extract env — this needs the real Aura-1B checkpoint + repo's
torch/transformers stack.

Usage (from repo root, on a GPU node):
    PYTHONPATH=src python scripts/smoke_test_cached.py \
        --config configs/experiment/stage2/smoke/stage2_omniasr_cached_smoke.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "/ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1")
sys.path.insert(0, "/ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/src")

import torch
from torch.utils.data import DataLoader

from st.data import CachedFeatureDataset, CachedFeatureCollator, DurationBucketSampler
from st.training.train_st import build_model, run_forward
from st.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device: {device}")

    cfg = load_config(args.config)
    assert cfg.get("cached_features", {}).get("enabled", False), \
        "config must have cached_features.enabled: true"

    log.info("Building model (SpeechAura with encoder=None)...")
    model = build_model(cfg).to(device)
    model.eval()
    log.info(f"[1] Model built OK. encoder={model.encoder}, "
             f"encoder_output_dim={model._encoder_output_dim}")

    data_cfg = cfg["data"]
    cache_dir = cfg["cached_features"]["cache_dir"]
    train_cfg = cfg["training"]

    ds = CachedFeatureDataset(
        index_path=data_cfg["train_index"],
        cache_dir=cache_dir,
        split=data_cfg.get("train_split", "train"),
        languages=data_cfg.get("languages"),
        max_duration=data_cfg.get("max_duration", 20.0),
        lowercase=data_cfg.get("lowercase", False),
    )
    log.info(f"[2] CachedFeatureDataset built OK: {len(ds)} entries")

    sample0 = ds[0]
    log.info(
        f"    sample0: hidden_states={tuple(sample0['hidden_states'].shape)} "
        f"dtype={sample0['hidden_states'].dtype}, "
        f"predicted_ids={tuple(sample0['predicted_ids'].shape)} "
        f"dtype={sample0['predicted_ids'].dtype}, "
        f"feature_len={sample0['feature_len']}, lang={sample0['language']}"
    )

    collator = CachedFeatureCollator(
        tokenizer=model.aura.tokenizer,
        max_target_tokens=train_cfg.get("max_target_tokens", 256),
    )

    sampler = DurationBucketSampler(
        dataset=ds,
        target_duration=train_cfg.get("max_batch_duration", 240.0),
        max_batch_size=args.batch_size,
        shuffle=True,
        shuffle_buckets=True,
    )
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0, collate_fn=collator)

    batch = next(iter(loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    B, T, D = batch["encoder_hidden_states"].shape
    log.info(
        f"[3] Batch loaded OK: B={B}, T={T}, D={D}, "
        f"encoder_lengths={batch['encoder_lengths'].tolist()}, "
        f"target_ids={tuple(batch['target_ids'].shape)}, "
        f"languages={batch['language']}"
    )
    assert D == 1280, f"expected hidden dim 1280, got {D}"

    with torch.no_grad(), torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
    ):
        # Peek at post-compression lengths via encode_audio_cached directly,
        # to report the compression ratio before running the full forward pass.
        _, compressed_lengths = model.encode_audio_cached(
            batch["encoder_hidden_states"], batch["ctc_predicted_ids"], batch["encoder_lengths"]
        )
        out = run_forward(model, batch, cached=True)

    pre_frames  = batch["encoder_lengths"].float()
    post_frames = compressed_lengths.float()
    ratios = (post_frames / pre_frames).tolist()
    log.info(f"[4] CTCCompressor compression ratio (post/pre frames) per sample: "
             f"{[f'{r:.3f}' for r in ratios]}")

    loss = out["loss"].item()
    ce_loss = out["ce_loss"].item()
    logits = out["logits"]
    log.info(f"[5] forward_cached() OK: loss={loss:.4f}, ce_loss={ce_loss:.4f}, "
             f"logits shape={tuple(logits.shape)}")

    assert torch.isfinite(out["loss"]), f"loss is not finite: {loss}"
    assert out["ctc_loss"].item() == 0.0, "ctc_loss should be exactly 0.0 in cached mode"
    for r in ratios:
        assert 0.05 < r <= 1.0, f"suspicious compression ratio {r} (expected roughly 0.3-0.9)"

    log.info("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
