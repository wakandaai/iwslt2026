#!/usr/bin/env python3
"""
Compute per-language corpus statistics from ASR_INDEX_V4_16k.csv.

Header (confirmed via head -2):
  audio_id, path, transcript, language, split, source, speaker_id, sample_rate, duration
"""

import sys
import pandas as pd

CSV_PATH = "/ocean/projects/cis250145p/shared/ASR_INDEX_V4_16k.csv"

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    print(f"Reading {path} ...")

    df = pd.read_csv(
        path,
        quotechar='"',
        on_bad_lines="warn",
        dtype={"sample_rate": "Int32", "duration": "float32"},
    )

    df["duration_s"] = pd.to_numeric(df["duration"], errors="coerce")
    df = df.dropna(subset=["duration_s", "language"])

    stats = (
        df.groupby("language")
        .agg(
            Utterances=("audio_id", "count"),
            total_s=("duration_s", "sum"),
            Avg_Dur_s=("duration_s", "mean"),
        )
        .reset_index()
    )

    stats["Hours"] = stats["total_s"] / 3600.0
    total_hours = stats["Hours"].sum()
    stats["pct_Hours"] = stats["Hours"] / total_hours * 100
    stats["Language"] = stats["language"].str.capitalize()

    stats = stats.sort_values("Hours", ascending=False).reset_index(drop=True)

    # ── Print table ────────────────────────────────────────────────────
    hdr = (
        f"{'Language':<15} {'Utterances':>12} {'Hours':>10} "
        f"{'% Hours':>9} {'Avg Dur (s)':>12}"
    )
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)
    for _, row in stats.iterrows():
        print(
            f"{row['Language']:<15} {int(row['Utterances']):>12,} "
            f"{row['Hours']:>10.1f} {row['pct_Hours']:>8.1f} "
            f"{row['Avg_Dur_s']:>12.1f}"
        )
    print(sep)
    total_utt = int(stats["Utterances"].sum())
    total_h   = stats["Hours"].sum()
    total_pct = stats["pct_Hours"].sum()
    print(
        f"{'Total':<15} {total_utt:>12,} {total_h:>10.1f} "
        f"{total_pct:>8.1f}"
    )
    print(sep)

    # ── Optional: also write a CSV ─────────────────────────────────────
    out_cols = ["Language", "Utterances", "Hours", "pct_Hours", "Avg_Dur_s"]
    out = stats[out_cols].rename(columns={
        "pct_Hours": "pct_Hours",
        "Avg_Dur_s": "Avg_Dur_s",
    })
    out.to_csv("corpus_stats.csv", index=False, float_format="%.1f")
    print("\nSaved → corpus_stats.csv")

if __name__ == "__main__":
    main()