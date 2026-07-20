"""
One-shot comparison: internal encoder-CTC-only decode vs full
encoder->CTCCompressor->projector->LLM pipeline, on the same dev utterances
used in the stage2_omniasr_cached validation run.

CPU-only — decodes the already-cached predicted_ids (argmax CTC output)
directly via the omniASR SentencePiece tokenizer, no GPU/model load needed.
Cross-references against runs/stage2_omniasr_cached/val_preds_step250.csv
(the full-pipeline hypotheses already saved by the training job's eval loop).
"""

import csv
import sys

sys.path.insert(0, "/ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1")
sys.path.insert(0, "/ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/src")

import torch
import sentencepiece as spm

from st.data.dataset import load_index_csv

# CACHE_DIR/INDEX/VAL_PREDS_CSV point at the historical stage2_omniasr_cached
# run's artifacts, which only exist under the original iwslt2026 checkout.
CACHE_DIR = "/ocean/projects/cis250145p/tanghang/iwslt2026/cache/omniasr_ctc_1b"
INDEX = "/ocean/projects/cis250145p/shared/datasets/ASR_INDEX.csv"
TOKENIZER = "/ocean/projects/cis250145p/tanghang/iwslt2026/checkpoints/omniasr_ctc_1b/omniASR_tokenizer.model"
VAL_PREDS_CSV = "/ocean/projects/cis250145p/tanghang/iwslt2026/runs/stage2_omniasr_cached/val_preds_step250.csv"


def cache_path(audio_id: str) -> str:
    return f"{CACHE_DIR}/{audio_id[:2]}/{audio_id}.pt"


def ctc_greedy_decode(ids: list[int], blank_id: int = 0) -> list[int]:
    out, prev = [], -1
    for i in ids:
        if i != blank_id and i != prev:
            out.append(i)
        prev = i
    return out


def build_val_generate_indices(entries, samples_per_lang=20):
    from collections import defaultdict
    lang_indices = defaultdict(list)
    for idx, entry in enumerate(entries):
        lang = entry.get("language") or entry.get("src_language") or "?"
        if len(lang_indices[lang]) < samples_per_lang:
            lang_indices[lang].append(idx)
    indices = []
    for lang in sorted(lang_indices):
        indices.extend(lang_indices[lang])
    return indices


def main():
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER)

    # Restricted to igbo+yoruba only, matching exactly what the training job's
    # dev cache contained when it ran its step-250 eval — the concurrent
    # extract_4lang_balanced job has since added hausa/bemba dev entries to
    # the same shared cache_dir, which would otherwise silently misalign this
    # script's index-based zip against val_preds_step250.csv (computed before
    # those languages existed in the cache).
    entries = load_index_csv(
        INDEX, "dev", ["igbo", "yoruba"], None, 0.1, 20.0
    )
    # Same cache-coverage filter CachedFeatureDataset applies
    import pathlib
    cached_ids = {p.stem for p in pathlib.Path(CACHE_DIR).glob("*/*.pt")}
    entries = [e for e in entries if e.get("audio_id", "") in cached_ids]

    val_indices = build_val_generate_indices(entries, samples_per_lang=20)

    # Load the full-pipeline hypotheses (in the same idx order as val_indices)
    with open(VAL_PREDS_CSV, newline="", encoding="utf-8") as f:
        pipeline_rows = list(csv.DictReader(f))

    print(f"{len(val_indices)} val-generate samples, {len(pipeline_rows)} pipeline rows\n")

    for i, idx in enumerate(val_indices):
        entry = entries[idx]
        audio_id = entry.get("audio_id", "")
        lang = entry.get("language", "")
        ref = entry.get("transcript", "")

        cached = torch.load(cache_path(audio_id), map_location="cpu", weights_only=True)
        pred_ids = cached["predicted_ids"].long().tolist()
        decoded_ids = ctc_greedy_decode(pred_ids)
        ctc_only_hyp = sp.decode(decoded_ids)

        pipeline_hyp = pipeline_rows[i]["hypothesis"] if i < len(pipeline_rows) else "?"

        print(f"[{i}] lang={lang} audio_id={audio_id}")
        print(f"  REF          : {ref[:100]}")
        print(f"  CTC-ONLY     : {ctc_only_hyp[:100]}")
        print(f"  FULL-PIPELINE: {pipeline_hyp[:100]}")
        print()


if __name__ == "__main__":
    main()
