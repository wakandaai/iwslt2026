"""
Cascade baseline for AST: SpeechAura ASR → NLLB text MT.

    audio --[SpeechAura ASR]--> hypothesis --[NLLB]--> translation --> BLEU/chrF

This is the number an end-to-end speech→text translation system has to beat.
`--topline` skips the ASR stage and feeds the ground-truth transcript to NLLB
instead; the gap between topline and cascade is the cost of ASR error.

Usage:
    python scripts/cascade_baseline.py \
        --config     exports/speech_aura_transcribe/config.yaml \
        --checkpoint exports/speech_aura_transcribe \
        --index      /ocean/projects/cis250145p/shared/datasets/AST_INDEX.csv \
        --out        runs/cascade_dev.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.utils.config import load_config
from st.data.dataset import SpeechDataset
from st.data.nllb_lang import to_flores, verify_lang_codes
from st.inference.generate import build_model_for_inference
from st.utils.metrics import compute_bleu, compute_chrf, compute_wer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cascade")

NLLB_DEFAULT = "/ocean/projects/cis250145p/shared/checkpoints/nllb-200-distilled-600M"


def translate(
    model, tokenizer, texts: list[str], src_lang: str, tgt_lang: str,
    device: torch.device, batch_size: int = 32, num_beams: int = 4,
    max_new_tokens: int = 256,
) -> list[str]:
    """Translate texts with NLLB. src_lang/tgt_lang are FLORES-200 codes."""
    tokenizer.src_lang = src_lang
    bos = tokenizer.convert_tokens_to_ids(tgt_lang)
    if bos == tokenizer.unk_token_id:
        raise ValueError(f"target code {tgt_lang!r} is <unk> in the NLLB tokenizer")

    out: list[str] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        inputs = tokenizer(
            chunk, return_tensors="pt", padding=True, truncation=True, max_length=512,
        ).to(device)
        with torch.inference_mode():
            ids = model.generate(
                **inputs,
                forced_bos_token_id=bos,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
            )
        out.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="ASR → NLLB cascade baseline")
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--index",      required=True)
    p.add_argument("--split",      default="dev")
    p.add_argument("--nllb",       default=NLLB_DEFAULT)
    p.add_argument("--out",        default="cascade.jsonl")
    p.add_argument("--limit",      type=int, default=0, help="0 = all")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_beams",  type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--topline",    action="store_true",
                   help="Feed ground-truth transcripts to NLLB (skip ASR)")
    p.add_argument("--asr_cache",  default="runs/cascade/asr_hyps.jsonl",
                   help="Reuse/append ASR hypotheses here. The ASR stage takes ~1h "
                        "and the MT stage ~1min, so caching makes MT-side changes "
                        "(beam size, NLLB checkpoint) cheap to re-run.")
    p.add_argument("--device",     default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = SpeechDataset(
        index_path=args.index, split=args.split, task="st", max_duration=30.0,
    )
    n = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    log.info(f"{args.split}: {len(ds)} utterances, evaluating {n}")

    # ---- NLLB ----------------------------------------------------------
    from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast

    tokenizer = NllbTokenizerFast.from_pretrained(args.nllb)
    verify_lang_codes(tokenizer)
    nllb = AutoModelForSeq2SeqLM.from_pretrained(
        args.nllb, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()
    log.info(f"NLLB loaded ← {args.nllb}")

    # ---- ASR stage -----------------------------------------------------
    # Cache by audio_id: the eng→X directions reuse the same English audio
    # across every target language, so ASR would otherwise run ~9x per clip.
    asr_cache: dict[str, str] = {}
    cache_f = None
    asr_model = None

    if not args.topline:
        # Resume from a previous run: ASR is ~1h, MT is ~1min.
        cache_path = Path(args.asr_cache)
        if cache_path.exists():
            with cache_path.open() as f:
                for line in f:
                    d = json.loads(line)
                    asr_cache[d["audio_id"]] = d["asr_hyp"]
            log.info(f"Resuming: {len(asr_cache)} cached ASR hypotheses ← {cache_path}")

        cfg = load_config(args.config)
        asr_model = build_model_for_inference(cfg, args.checkpoint, device)
        log.info(f"ASR model loaded ← {args.checkpoint}")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_f = cache_path.open("a")

    rows: list[dict] = []
    t0 = time.time()

    for i in range(n):
        s = ds[i]
        if args.topline:
            source_text = s["transcript"]
        else:
            aid = s["audio_id"]
            if aid not in asr_cache:
                mel = s["mel"].unsqueeze(0).to(device)
                mel_len = torch.tensor([s["mel_len"]], device=device)
                with torch.inference_mode():
                    hyp = asr_model.generate(
                        audio_features=mel, audio_lengths=mel_len,
                        src_lang=s["src_language"], task="asr",
                        max_new_tokens=args.max_new_tokens,
                    )
                asr_cache[aid] = asr_model._strip_special_tokens(hyp).strip()
                cache_f.write(json.dumps(
                    {"audio_id": aid, "asr_hyp": asr_cache[aid]}, ensure_ascii=False,
                ) + "\n")
                cache_f.flush()
            source_text = asr_cache[aid]

        rows.append({
            "audio_id":     s["audio_id"],
            "src_language": s["src_language"],
            "tgt_language": s["tgt_language"],
            "transcript":   s["transcript"],
            "asr_hyp":      None if args.topline else source_text,
            "source_text":  source_text,
            "reference":    s["translation"],
        })

        if (i + 1) % 200 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (n - i - 1) / rate / 60
            log.info(f"  ASR {i + 1}/{n}  ({rate:.1f} utt/s, "
                     f"{len(asr_cache)} unique, ETA {eta:.0f} min)")
            sys.stderr.flush()

    if cache_f is not None:
        cache_f.close()
    if asr_model is not None:
        del asr_model
        torch.cuda.empty_cache()

    # ---- MT stage, batched per direction --------------------------------
    by_dir: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for j, r in enumerate(rows):
        by_dir[(r["src_language"], r["tgt_language"])].append(j)

    for (src, tgt), idxs in sorted(by_dir.items()):
        hyps = translate(
            nllb, tokenizer,
            [rows[j]["source_text"] for j in idxs],
            to_flores(src), to_flores(tgt), device,
            batch_size=args.batch_size, num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
        )
        for j, h in zip(idxs, hyps):
            rows[j]["mt_hyp"] = h
        log.info(f"  MT {src}->{tgt}: {len(idxs)} done")

    # ---- Score ----------------------------------------------------------
    mode = "TOPLINE (gold transcripts)" if args.topline else "CASCADE (ASR → NLLB)"
    print(f"\n=== {mode} — {args.split} ===")
    print(f"{'direction':28s} {'n':>5s} {'BLEU':>7s} {'chrF++':>7s} {'ASR WER':>8s}")
    print("-" * 60)

    for (src, tgt), idxs in sorted(by_dir.items(), key=lambda kv: -len(kv[1])):
        preds = [rows[j]["mt_hyp"] for j in idxs]
        refs  = [rows[j]["reference"] for j in idxs]
        bleu  = compute_bleu(preds, refs)["bleu"]
        chrf  = compute_chrf(preds, refs)["chrf"]
        if args.topline:
            wer_s = "     —"
        else:
            wer = compute_wer(
                [rows[j]["asr_hyp"] for j in idxs],
                [rows[j]["transcript"] for j in idxs],
            )
            wer_s = f"{wer * 100:7.1f}" if wer <= 1.5 else f"{wer:7.2f}"
        print(f"{src + '->' + tgt:28s} {len(idxs):5d} {bleu:7.2f} {chrf:7.2f} {wer_s}")

    preds = [r["mt_hyp"] for r in rows]
    refs  = [r["reference"] for r in rows]
    print("-" * 60)
    print(f"{'CORPUS':28s} {len(rows):5d} "
          f"{compute_bleu(preds, refs)['bleu']:7.2f} "
          f"{compute_chrf(preds, refs)['chrf']:7.2f}")
    print("\nNote: corpus average is dominated by the largest direction — "
          "read the per-direction rows.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
