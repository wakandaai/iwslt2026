#!/usr/bin/env python3
"""Build new NCHLT train-split rows for the 4 languages we actually evaluate
NCHLT on (Zulu, Xhosa, Sesotho, Setswana), and write a new versioned index
(ASR_INDEX_V6_16k.csv = V5 + these rows appended).

Why: the only NCHLT-sourced training rows currently indexed (source=nchlt,
source=nchlt_aux) are 100% Afrikaans (161,731 rows) -- Afrikaans is not even
one of the 4 languages stt_bench evaluates NCHLT on. The real NCHLT corpus
for zul/xho/sot/tsn sits unindexed on disk at
/ocean/projects/cis250145p/shared/datasets/NCHLT/nchlt_{lang}/, already
16kHz mono PCM (confirmed via soundfile.info -- no resampling needed), with
an official train/test split baked into the corpus itself
(transcriptions/nchlt_{lang}.trn.xml vs .tst.xml). stt_bench's own NCHLT eval
(stt_benchmark/datasets/nchlt.py) reads the SAME root directory and reads
ONLY the .tst.xml manifest -- so using .trn.xml here is guaranteed disjoint
from the eval set by the corpus's own official split, not a split we invent.

Schema (matches ASR_INDEX_V5_16k.csv exactly):
    audio_id, path, transcript, language, split, source, speaker_id,
    sample_rate, duration

audio_id convention matches the existing nchlt/afrikaans rows exactly:
"nchlt_{language}_{original_filename_stem}".
"""

import csv
import re
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

NCHLT_ROOT = Path("/ocean/projects/cis250145p/shared/datasets/NCHLT")

V5_INDEX = Path("/ocean/projects/cis250145p/shared/ASR_INDEX_V5_16k.csv")
V6_INDEX = Path("/ocean/projects/cis250145p/shared/ASR_INDEX_V6_16k.csv")

# nchlt corpus code -> language name matching index convention
LANGS = {
    "zul": "zulu",
    "xho": "xhosa",
    "sot": "sotho",
    "tsn": "tswana",
}

RECORDING_RE = re.compile(r'<recording audio="([^"]+)"[^>]*duration="([0-9.]+)"[^>]*>')
SPEAKER_RE = re.compile(r'<speaker id="([^"]+)"')
ORTH_RE = re.compile(r"<orth>(.*?)</orth>", re.DOTALL)


def build_rows_for_lang(code: str, lang_name: str) -> list[dict]:
    """Parse nchlt_{code}.trn.xml by hand (regex, not a full XML parser --
    the file is simple enough and this matches build_fleurs_augment.py's
    style of not pulling in extra deps for a one-off script)."""
    xml_path = NCHLT_ROOT / f"nchlt_{code}" / "transcriptions" / f"nchlt_{code}.trn.xml"
    audio_root = NCHLT_ROOT / f"nchlt_{code}"

    rows = []
    skipped_missing = 0
    cur_speaker = None
    with open(xml_path, encoding="utf-8") as f:
        content = f.read()

    # Walk speaker blocks so each recording gets the right speaker id.
    speaker_blocks = re.split(r'(?=<speaker id=")', content)
    for block in speaker_blocks:
        sp_match = SPEAKER_RE.search(block)
        if not sp_match:
            continue
        cur_speaker = sp_match.group(1)
        for rec_match in re.finditer(
            r'<recording audio="([^"]+)"[^>]*duration="([0-9.]+)"[^>]*>(.*?)</recording>',
            block, re.DOTALL,
        ):
            audio_rel, duration, rec_body = rec_match.groups()
            orth_match = ORTH_RE.search(rec_body)
            if not orth_match:
                skipped_missing += 1
                continue
            transcript = orth_match.group(1).strip()
            audio_path = NCHLT_ROOT / audio_rel
            if not audio_path.is_file():
                skipped_missing += 1
                continue
            file_stem = Path(audio_rel).stem
            rows.append({
                "audio_id": f"nchlt_{lang_name}_{file_stem}",
                "path": str(audio_path),
                "transcript": transcript,
                "language": lang_name,
                "split": "train",
                "source": "nchlt",
                "speaker_id": cur_speaker,
                "sample_rate": "16000",
                "duration": duration,
            })
    print(f"  nchlt_{code:4s} -> {lang_name:8s} {len(rows):6d} rows "
          f"({skipped_missing} skipped: missing audio/orth)", flush=True)
    return rows


def main():
    print("Building new NCHLT train rows (zul/xho/sot/tsn)...", flush=True)
    all_new_rows = []
    for code, lang_name in LANGS.items():
        all_new_rows.extend(build_rows_for_lang(code, lang_name))
    print(f"Total new rows: {len(all_new_rows)}", flush=True)
    total_hours = sum(float(r["duration"]) for r in all_new_rows) / 3600
    print(f"Total new hours: {total_hours:.1f}", flush=True)

    print(f"Copying {V5_INDEX} -> {V6_INDEX} and appending...", flush=True)
    with open(V5_INDEX, newline="", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        with open(V6_INDEX, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(header)
            n_old = 0
            for row in reader:
                writer.writerow(row)
                n_old += 1
            for r in all_new_rows:
                writer.writerow([r[col] for col in header])

    print(f"Done. V5 had {n_old} rows; V6 has {n_old + len(all_new_rows)} rows "
          f"({len(all_new_rows)} new NCHLT rows).", flush=True)


if __name__ == "__main__":
    main()
