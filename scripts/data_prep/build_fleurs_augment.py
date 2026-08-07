#!/usr/bin/env python3
"""Build new FLEURS train-split rows for languages the current
ASR_INDEX_V4_16k.csv never included, and write a new versioned index
(ASR_INDEX_V5_16k.csv = V4 + these rows appended).

Why: the Stage 1 fine-tuned CTC encoder regressed on FLEURS vs the base
checkpoint (see stt_bench eval), because FLEURS made up ~0.1% of the
fine-tuning mix and 9/11 affected languages had ZERO FLEURS training rows at
all -- full-parameter fine-tuning drifted the encoder's acoustic weights
toward the (very different-domain) sources it was actually shown. FLEURS
train-split audio already exists on disk for 11 of our 22 languages; this
just wires it into the index so the next Stage 1 continuation can include it.

Schema (matches ASR_INDEX_V4_16k.csv exactly):
    audio_id, path, transcript, language, split, source, speaker_id,
    sample_rate, duration

Row format matches the existing fleurs_hausa/_igbo/_yoruba rows already in
the index (reverse-engineered from those): audio_id = "fleurs_{lang}_{id}",
transcript = the tsv's raw_transcription column (capitalized, punctuated,
NOT the lowercased one), speaker_id left blank, duration = num_samples/16000.

Kinyarwanda is deliberately excluded: the mbazaNLP_fleurs-kinyarwanda source
used for eval has no train split (dev/test only) -- there is nothing to add.
"""

import csv
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

FLEURS_ROOT = Path("/ocean/projects/cis250145p/shared/datasets/FLEURS/original_data")
FLEURS_16K_MIRROR = Path("/ocean/projects/cis250145p/shared/datasets_16k/FLEURS/original_data")

V4_INDEX = Path("/ocean/projects/cis250145p/shared/ASR_INDEX_V4_16k.csv")
V5_INDEX = Path("/ocean/projects/cis250145p/shared/ASR_INDEX_V5_16k.csv")

# fleurs code -> (language name matching index convention, audio root to use)
# Languages already fully indexed (ha_ng/ig_ng/yo_ng) are intentionally
# excluded here -- adding them again would create duplicate rows.
LANGS = {
    "am_et": "amharic",
    "en_us": "english",
    "lg_ug": "luganda",
    "ln_cd": "lingala",
    "sn_zw": "shona",
    "sw_ke": "swahili",
    "xh_za": "xhosa",
    "zu_za": "zulu",
    "fr_fr": "french",
    "pt_br": "portuguese",
    "ar_eg": "arabic",
}

TSV_COLUMNS = [
    "id", "file_name", "raw_transcription", "transcription",
    "char_transcription", "num_samples", "gender",
]


def build_rows_for_lang(code: str, lang_name: str) -> list[dict]:
    tsv_path = FLEURS_ROOT / code / "train.tsv"
    # Files are split across the two locations (not duplicated) -- the 16k
    # mirror only has ~half of each language's train set. Check both, mirror
    # first (matches the existing hausa/igbo/yoruba index rows' provenance).
    mirror_dir = FLEURS_16K_MIRROR / code / "audio" / "train"
    orig_dir = FLEURS_ROOT / code / "audio" / "train"
    rows = []
    skipped_missing = 0
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for parts in reader:
            if len(parts) != len(TSV_COLUMNS):
                continue
            rec = dict(zip(TSV_COLUMNS, parts))
            file_id = Path(rec["file_name"]).stem
            audio_path = mirror_dir / rec["file_name"]
            if not audio_path.is_file():
                audio_path = orig_dir / rec["file_name"]
            if not audio_path.is_file():
                skipped_missing += 1
                continue
            try:
                num_samples = int(rec["num_samples"])
            except ValueError:
                skipped_missing += 1
                continue
            rows.append({
                "audio_id": f"fleurs_{lang_name}_{file_id}",
                "path": str(audio_path),
                "transcript": rec["raw_transcription"],
                "language": lang_name,
                "split": "train",
                "source": "fleurs",
                "speaker_id": "",
                "sample_rate": "16000",
                "duration": f"{num_samples / 16000:.2f}",
            })
    print(f"  {code:8s} -> {lang_name:12s} {len(rows):5d} rows "
          f"({skipped_missing} skipped: missing audio/bad row)", flush=True)
    return rows


def main():
    print("Building new FLEURS train rows...", flush=True)
    all_new_rows = []
    for code, lang_name in LANGS.items():
        all_new_rows.extend(build_rows_for_lang(code, lang_name))
    print(f"Total new rows: {len(all_new_rows)}", flush=True)

    print(f"Copying {V4_INDEX} -> {V5_INDEX} and appending...", flush=True)
    with open(V4_INDEX, newline="", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        with open(V5_INDEX, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(header)
            n_old = 0
            for row in reader:
                writer.writerow(row)
                n_old += 1
            for r in all_new_rows:
                writer.writerow([r[col] for col in header])

    print(f"Done. V4 had {n_old} rows; V5 has {n_old + len(all_new_rows)} rows "
          f"({len(all_new_rows)} new FLEURS rows).", flush=True)


if __name__ == "__main__":
    main()
