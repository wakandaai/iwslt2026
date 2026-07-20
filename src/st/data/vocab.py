"""CTC character vocabulary utilities."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def build_vocab_from_index(
    index_path: str | Path,
    text_column: str = "transcript",
    split: str | None = "train",
    languages: list[str] | None = None,
    lowercase: bool = True,
) -> dict[str, int]:
    """Build character-level vocab from an index CSV.

    Index 0 is reserved for CTC blank. Characters are sorted for
    deterministic ordering across runs.
    """
    chars: set[str] = set()
    lang_set = set(languages) if languages else None

    with open(index_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split is not None and row.get("split", "") != split:
                continue
            lang = row.get("language") or row.get("src_language") or ""
            if lang_set is not None and lang not in lang_set:
                continue
            text = row.get(text_column, "")
            if lowercase:
                text = text.lower()
            chars.update(text)

    vocab: dict[str, int] = {"<blank>": 0}
    for i, c in enumerate(sorted(chars)):
        vocab[c] = i + 1

    log.info(f"Built vocab: {len(vocab)} tokens (incl. blank) from {index_path}")
    return vocab


def save_vocab(vocab: dict[str, int], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    log.info(f"Saved vocab ({len(vocab)} tokens) → {path}")


def load_vocab(path: str | Path) -> dict[str, int]:
    with open(path, encoding="utf-8") as f:
        vocab = json.load(f)
    log.info(f"Loaded vocab: {len(vocab)} tokens ← {path}")
    return vocab
