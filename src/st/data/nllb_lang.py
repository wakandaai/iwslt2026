"""
Dataset language name → NLLB-200 FLORES-200 code.

NLLB selects the output language via a single token (`forced_bos_token_id`), so a
code that is not in the tokenizer vocabulary silently becomes <unk> and the model
translates into *some other* language rather than failing. verify_lang_codes()
guards against that and should be called once at startup.

Covers every language appearing in AST_INDEX.csv (src_language / tgt_language).
"""

from __future__ import annotations

# Keys are the dataset's own language names, as they appear in AST_INDEX.csv.
NLLB_LANG_CODE: dict[str, str] = {
    "afrikaans":   "afr_Latn",
    "amharic":     "amh_Ethi",
    "arabic":      "arb_Arab",   # Modern Standard Arabic
    "bemba":       "bem_Latn",
    "english":     "eng_Latn",
    "french":      "fra_Latn",
    "hausa":       "hau_Latn",
    "igbo":        "ibo_Latn",
    "kinyarwanda": "kin_Latn",
    "lingala":     "lin_Latn",
    "luganda":     "lug_Latn",   # NLLB calls this "Ganda"
    "malagasy":    "plt_Latn",   # Plateau Malagasy
    "portuguese":  "por_Latn",
    "shona":       "sna_Latn",
    "sotho":       "sot_Latn",   # Southern Sotho — the corpus is za_african_next_voices,
                                 # not Northern Sotho (nso_Latn)
    "swahili":     "swh_Latn",
    "tigrinya":    "tir_Ethi",
    "tsonga":      "tso_Latn",
    "tswana":      "tsn_Latn",
    "xhosa":       "xho_Latn",
    "yoruba":      "yor_Latn",
    "zulu":        "zul_Latn",
}


def to_flores(name: str) -> str:
    """Map a dataset language name to its FLORES-200 code."""
    key = name.strip().lower()
    if key in NLLB_LANG_CODE:
        return NLLB_LANG_CODE[key]
    if key in set(NLLB_LANG_CODE.values()):  # already a FLORES code
        return key
    raise KeyError(
        f"No NLLB code for language {name!r}. "
        f"Known: {sorted(NLLB_LANG_CODE)}"
    )


def verify_lang_codes(tokenizer) -> None:
    """Raise if any code is missing from the tokenizer vocabulary."""
    unk = tokenizer.unk_token_id
    bad = [
        f"{name} ({code})"
        for name, code in NLLB_LANG_CODE.items()
        if tokenizer.convert_tokens_to_ids(code) in (None, unk)
    ]
    if bad:
        raise ValueError(
            "NLLB tokenizer does not recognise these language codes, so they would "
            f"silently decode into the wrong language: {', '.join(bad)}"
        )
