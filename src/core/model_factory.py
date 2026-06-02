"""
model_factory.py — model config builder.

Builder for ModelArgs that decouples architectural shape from tokenizer
choice and context length. API:
    build_model_config(model_type, size, vocab_size, max_seq_len) -> ModelArgs
"""

from __future__ import annotations

from st.models.llama3 import ModelArgs


# ----------------------------------------------------------------------------
#  Architectural presets — shape only. No vocab_size, no max_seq_len.
# ----------------------------------------------------------------------------

ARCH_PRESETS: dict[str, dict[str, dict]] = {
    "llama-iwslt": {
        "124m": dict(n_layers=12, n_heads=12, dim=768,  intermediate_size=4 * 768,  n_kv_heads=12),
        "500m": dict(n_layers=32, n_heads=16, dim=1024, intermediate_size=3072,     n_kv_heads=4),
        "978m": dict(n_layers=36, n_heads=20, dim=1280, intermediate_size=5120,     n_kv_heads=4),
        "1b":   dict(n_layers=36, n_heads=20, dim=1280, intermediate_size=5120,     n_kv_heads=4),
        "2b":   dict(n_layers=36, n_heads=32, dim=2048, intermediate_size=5120,     n_kv_heads=8),
    },
}


def build_model_config(
    model_type: str,
    size: str,
    vocab_size: int,
    max_seq_len: int,
) -> ModelArgs:
    """Build a ModelArgs combining architectural preset + runtime vocab/seq_len."""
    if model_type not in ARCH_PRESETS:
        raise ValueError(f"unknown model_type {model_type!r}; "
                         f"expected one of {list(ARCH_PRESETS)}")
    if size not in ARCH_PRESETS[model_type]:
        raise ValueError(f"unknown size {size!r} for {model_type}; "
                         f"expected one of {list(ARCH_PRESETS[model_type])}")
    arch = ARCH_PRESETS[model_type][size]
    return ModelArgs(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        **arch,
    )