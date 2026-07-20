"""
One-shot smoke test for the live/unfrozen omniASR_CTC_1B encoder training path.

Builds SpeechAura with a live, trainable OmniASREncoder from
a stage4 live-omniASR config, pulls one real batch from
RawAudioDataset/RawAudioCollator, runs one forward + backward pass, and
confirms gradients actually reach the encoder's own parameters — not just
that the pipeline runs without crashing.

MUST run via the isolated omniasr_extract env (torch==2.8.0+cu128), NOT
Aura_base/env (torch==2.6.0+cu124) — this instantiates the live fairseq2
encoder. A ready-made one lives at
/ocean/projects/cis250145p/tanghang/iwslt2026/.envs/omniasr_extract.

Usage (from repo root, on a GPU node):
    <omniasr_extract>/bin/python scripts/smoke_test_omniasr_live.py \
        --config configs/experiment/stage4/smoke/stage4_omniasr_live_smoke.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "/ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1")
sys.path.insert(0, "/ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/src")

import torch
from torch.utils.data import DataLoader

from st.data import RawAudioDataset, RawAudioCollator, DurationBucketSampler
from st.training.train_st import build_model, run_forward
from st.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device: {device}")

    cfg = load_config(args.config)
    assert cfg.get("encoder", {}).get("type") == "omniasr_live", \
        "config must have encoder.type: omniasr_live"
    assert cfg["training"].get("ctc_weight", 0.0) == 0.0, \
        "config must have training.ctc_weight: 0.0"

    log.info("Building model (SpeechAura with live OmniASREncoder)...")
    model = build_model(cfg).to(device)
    model.train()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log.info(f"[1] Model built OK. {n_total:,} total params, {n_trainable:,} trainable")
    log.info(f"    encoder.ctc_head trainable: "
             f"{any(p.requires_grad for p in model.encoder.ctc_head.parameters())} "
             f"(expected False — freeze_ctc_head)")
    log.info(f"    encoder._model.encoder.training: {model.encoder._model.encoder.training} "
             f"(expected True — model.train() must propagate through the manual "
             f"submodule-call forward)")

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    ds = RawAudioDataset(
        index_path=data_cfg["train_index"],
        split=data_cfg.get("train_split", "train"),
        languages=data_cfg.get("languages"),
        max_duration=data_cfg.get("max_duration", 20.0),
        lowercase=data_cfg.get("lowercase", False),
    )
    log.info(f"[2] RawAudioDataset built OK: {len(ds)} entries")

    sample0 = ds[0]
    log.info(
        f"    sample0: waveform={tuple(sample0['waveform'].shape)} "
        f"dtype={sample0['waveform'].dtype}, waveform_len={sample0['waveform_len']}, "
        f"lang={sample0['language']}"
    )

    collator = RawAudioCollator(
        tokenizer=model.aura.tokenizer,
        max_target_tokens=train_cfg.get("max_target_tokens", 256),
    )

    sampler = DurationBucketSampler(
        dataset=ds,
        target_duration=train_cfg.get("max_batch_duration", 40.0),
        max_batch_size=args.batch_size,
        shuffle=True,
        shuffle_buckets=True,
    )
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0, collate_fn=collator)

    batch = next(iter(loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    B, T = batch["audio_features"].shape
    log.info(
        f"[3] Batch loaded OK: B={B}, T_samples={T}, "
        f"audio_lengths={batch['audio_lengths'].tolist()}, "
        f"target_ids={tuple(batch['target_ids'].shape)}, "
        f"languages={batch['language']}"
    )

    # --- Pass 1: forward + backward from the true zero-init state ---
    # NOTE: TransformerProjector.output_proj is deliberately zero-initialized
    # (src/st/models/projector.py) for training stability. For y = x @ W^T + b
    # with W=0, dL/dx = dL/dy @ W = 0 — meaning on this very first backward
    # pass, EVERYTHING upstream of output_proj (its own preceding transformer
    # layers, CTCCompressor, and the encoder) gets exactly zero gradient,
    # while output_proj's OWN gradient (dL/dW = x^T @ dL/dy) is nonzero. This
    # is expected zero-init-gating behavior (same idea as ControlNet), not a
    # bug — verified below, then a real optimizer step moves output_proj away
    # from zero so pass 2 can confirm gradient genuinely reaches the encoder.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5,
    )

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        out = run_forward(model, batch, cached=False)
    loss = out["loss"]
    log.info(f"[4] Pass 1 forward() OK: loss={loss.item():.4f}, "
             f"ce_loss={out['ce_loss'].item():.4f}, logits shape={tuple(out['logits'].shape)}")
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"

    loss.backward()

    out_proj_grad = model.projector.output_proj.weight.grad
    assert out_proj_grad is not None and out_proj_grad.abs().sum().item() > 0, (
        "output_proj's OWN gradient is zero/None on pass 1 — expected nonzero even "
        "at zero-init (dL/dW = x^T @ dL/dy doesn't depend on W's own value)"
    )
    log.info(f"[5] Pass 1: output_proj.weight.grad.abs().sum()="
             f"{out_proj_grad.abs().sum().item():.6f} (nonzero, as expected at zero-init)")

    last_layer = model.encoder._model.encoder.layers[-1]
    pass1_encoder_grads = [p.grad.abs().sum().item() for p in last_layer.parameters()
                           if p.requires_grad and p.grad is not None]
    log.info(f"[6] Pass 1: encoder last-layer grads all zero: "
             f"{all(g == 0 for g in pass1_encoder_grads)} (expected True — zero-init gating)")

    optimizer.step()
    optimizer.zero_grad()

    # --- Pass 2: output_proj has moved off zero, gradient should now reach the encoder ---
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        out2 = run_forward(model, batch, cached=False)
    loss2 = out2["loss"]
    log.info(f"[7] Pass 2 forward() OK: loss={loss2.item():.4f}")
    assert torch.isfinite(loss2), f"pass 2 loss is not finite: {loss2.item()}"

    loss2.backward()

    log.info("[8] Pass 2: last-layer param grad status:")
    any_nonzero_last = False
    for name, p in last_layer.named_parameters():
        if not p.requires_grad:
            continue
        gsum = 0.0 if p.grad is None else p.grad.abs().sum().item()
        log.info(f"    {name}: grad={'None' if p.grad is None else f'{gsum:.8f}'}")
        if gsum > 0:
            any_nonzero_last = True

    assert any_nonzero_last, (
        "encoder's last-layer gradients are STILL all-zero on pass 2, after output_proj "
        "moved off zero-init — this is a real problem (gradients not reaching the "
        "encoder), not the expected zero-init-gating behavior from pass 1"
    )
    log.info("[9] Pass 2: encoder last-layer gradients are nonzero — "
             "gradients genuinely reach the encoder once training progresses past step 1")

    ctc_head_grads = [p.grad for p in model.encoder.ctc_head.parameters()]
    assert all(g is None for g in ctc_head_grads), \
        "ctc_head has a gradient despite freeze_ctc_head — regression in the freeze logic"
    log.info("[10] ctc_head params: grad is None (correctly frozen)")

    llm_grads = [p.grad for p in model.aura.model.parameters()]
    assert all(g is None for g in llm_grads), "frozen LLM has a gradient — freeze_llm is broken"
    log.info("[11] Frozen LLM params: grad is None (correctly frozen)")

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    log.info(f"[12] Peak GPU memory: {peak_mem_gb:.2f} GB")

    log.info("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
