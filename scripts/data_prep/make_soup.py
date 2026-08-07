#!/usr/bin/env python3
"""Linearly interpolate the fine-tuned Stage1 v2 encoder with the base
omniASR_CTC_1B checkpoint ("model soup"), to see whether it recovers some of
the FLEURS-domain generalization lost to full-parameter fine-tuning while
keeping most of the WAXAL/NCHLT/Bemba gains.

Both checkpoints load into the SAME raw fairseq2 Wav2Vec2Asr key space once
the fine-tuned checkpoint's "_model." prefix is stripped (see
aura-asr-v1/src/st/models/omniasr_encoder.py's _build_and_load, which does
exactly this before calling model.load_state_dict). We reuse that same key
transform here so the interpolated result loads back through the unmodified
adapter, saved in the same bare {"model": state_dict} format the base
checkpoint uses.
"""

import sys
import torch

BASE_CKPT = "/ocean/projects/cis250145p/tanghang/iwslt2026/checkpoints/omniasr_ctc_1b/omniASR-CTC-1B.pt"
FT_CKPT = "/ocean/projects/cis250145p/tanghang/iwslt2026/runs/stage1_v2_ddp2/encoder_step50000.pt"
OUT_DIR = "/ocean/projects/cis250145p/tanghang/iwslt2026/runs/stage1_v2_ddp2"

ALPHAS = [0.5, 0.7]  # fraction of fine-tuned weights kept


def load_base_state():
    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    return ckpt["model"]


def load_ft_state():
    ckpt = torch.load(FT_CKPT, map_location="cpu", weights_only=False)
    msd = ckpt["model_state_dict"]
    del ckpt  # drop optimizer/scheduler state (~2/3 of the 11.7GB file) ASAP
    return {
        k.removeprefix("_model."): v
        for k, v in msd.items()
        if k.startswith("_model.")
    }


def main():
    print("Loading base checkpoint...", flush=True)
    base = load_base_state()
    print(f"  {len(base)} tensors", flush=True)

    print("Loading fine-tuned checkpoint...", flush=True)
    ft = load_ft_state()
    print(f"  {len(ft)} tensors", flush=True)

    if base.keys() != ft.keys():
        missing_in_ft = set(base) - set(ft)
        missing_in_base = set(ft) - set(base)
        print("KEY MISMATCH", file=sys.stderr)
        print("in base, not ft:", missing_in_ft, file=sys.stderr)
        print("in ft, not base:", missing_in_base, file=sys.stderr)
        sys.exit(1)

    for alpha in ALPHAS:
        print(f"Building soup alpha={alpha} ({alpha*100:.0f}% fine-tuned)...", flush=True)
        soup = {}
        for k in base:
            b, f = base[k], ft[k]
            if torch.is_floating_point(b):
                soup[k] = (alpha * f.float() + (1 - alpha) * b.float()).to(b.dtype)
            else:
                soup[k] = f  # non-float buffers: just keep fine-tuned's value
        out_path = f"{OUT_DIR}/soup_alpha{int(alpha*100)}.pt"
        torch.save({"model": soup}, out_path)
        print(f"  -> {out_path}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
