"""
Aura-1B LLM wrapper.

Isolates all llama3.py / model_factory.py imports so nothing else in the
codebase needs to touch them. Handles:
  - Loading pretrained weights
  - Freezing / unfreezing
  - LoRA adapter injection (manual, no peft dependency)
  - Exposing embed layer and a forward pass that accepts inputs_embeds

The wrapper is intentionally thin — it does not know about audio.
SpeechAura (speech_aura.py) owns the audio → LLM fusion.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# Language token IDs — the checkpoint's real tokenizer.json added_tokens
# (<|bem_Latn|>, <|eng_Latn|>, etc.), not an arbitrary reserved-slot scheme.
# Confirmed against the checkpoint's own tokenizer.json, not guessed.
#
# Covers 21 of our 22 trained languages (see ASR_INDEX_V5_16k.csv) — every one
# except Tsonga, which has no dedicated tag in this tokenizer (the closest
# entry, tsn_Latn, is Tswana — a different language, not a valid substitute).
# Previously only 5 of 22 were mapped here; the other 17 (now 16) silently
# fell back to the English tag in both training and inference, meaning most
# languages were trained/generated under the wrong language conditioning.
# Tsonga still falls back to English below — a known, isolated gap, not a
# silent one.
LANG_MAP: dict[str, int] = {
    "afrikaans": 4,     "afr": 4,
    "amharic": 5,       "amh": 5,
    "arabic": 6,        "arb": 6,
    "bemba": 7,         "bem": 7,
    "english": 8,       "eng": 8,
    "french": 10,       "fra": 10,
    "hausa": 11,        "hau": 11,
    "igbo": 12,         "ibo": 12,
    "kinyarwanda": 13,  "kin": 13,
    "lingala": 14,      "lin": 14,
    "luganda": 15,      "lug": 15,
    "malagasy": 17,     "plt": 17,
    "portuguese": 18,   "por": 18,
    "shona": 19,        "sna": 19,
    "sotho": 21,        "sot": 21,
    "swahili": 22,      "swh": 22,
    "tigrinya": 23,     "tir": 23,
    "tswana": 24,       "tsn": 24,
    "xhosa": 26,        "xho": 26,
    "yoruba": 27,       "yor": 27,
    "zulu": 28,         "zul": 28,
}

# Task marker tokens — also real special_tokens.json entries, not reserved
# slots. Selects the boundary token between audio embeddings and target text
# based on training.task ("transcribe" | "translate", see experiment configs).
TRANSCRIBE_TOKEN_ID = 29   # <|transcribe|>
TRANSLATE_TOKEN_ID  = 30   # <|translate|>


class AuraLLM(nn.Module):
    """Thin wrapper around the Aura-1B LlamaTransformer.

    Args:
        ckpt_path:       Path to Aura model.pt checkpoint.
        tokenizer_path:  Path to Aura tokenizer.json.
        size:            Model size key for model_presets (default "1b").
        freeze:          Freeze all LLM weights on init.
        lora_rank:       LoRA rank. 0 = no LoRA.
        lora_alpha:      LoRA scaling alpha.
        lora_targets:    Module name substrings to apply LoRA to.
        task:            "transcribe" or "translate" — selects the real
                        <|transcribe|>/<|translate|> boundary token used
                        between audio embeddings and target text.
    """

    def __init__(
        self,
        ckpt_path: str,
        tokenizer_path: str,
        size: str = "1b",
        freeze: bool = True,
        lora_rank: int = 0,
        lora_alpha: int = 32,
        lora_targets: list[str] | None = None,
        task: str = "transcribe",
    ):
        super().__init__()
        from transformers import PreTrainedTokenizerFast

        # --- Tokenizer ---
        # Load via from_pretrained on the checkpoint directory (not
        # PreTrainedTokenizerFast(tokenizer_file=...) on tokenizer.json alone) —
        # the raw tokenizer.json has no bos/eos/pad attributes set, but the
        # sibling tokenizer_config.json defines them ("<s>"=1, "</s>"=2, "<pad>"=0,
        # already inside the checkpoint's 64000-row embedding table). Loading
        # tokenizer.json directly left bos/eos/pad as None, which triggered
        # add_special_tokens to invent *new* tokens at ids 64000/64001 — past
        # the embedding table's bounds, causing an out-of-range CUDA assert.
        import os
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(os.path.dirname(tokenizer_path))

        self.bos_id  = self.tokenizer.bos_token_id
        self.eos_id  = self.tokenizer.eos_token_id

        if task not in ("transcribe", "translate"):
            raise ValueError(f"task must be 'transcribe' or 'translate', got {task!r}")
        self.task_token_id = TRANSCRIBE_TOKEN_ID if task == "transcribe" else TRANSLATE_TOKEN_ID

        # --- LLM ---
        # llama3 and model_factory live in st/models/ alongside this file
        from st.models.llama3 import LlamaTransformer
        from st.models.model_factory import model_presets

        cfg = model_presets["llama-iwslt"][size]
        self.model = LlamaTransformer(cfg)
        self.hidden_size: int = cfg.dim
        self.vocab_size: int  = cfg.vocab_size
        self.n_layers: int    = cfg.n_layers

        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(ckpt_path, device="cpu")
        else:
            # The checkpoint was saved with llama3/model_factory as top-level modules.
            # Remap them to their new location (st.models.*) so torch.load's
            # internal unpickling can resolve the classes.
            from st.models import llama3 as _llama3_mod
            from st.models import model_factory as _mf_mod
            import sys as _sys
            _sys.modules.setdefault("llama3", _llama3_mod)
            _sys.modules.setdefault("model_factory", _mf_mod)

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt)

        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state)
        log.info(f"Loaded Aura-{size} from {ckpt_path}")
        del ckpt, state

        import gc; gc.collect()

        if freeze:
            self.freeze()

        # LoRA (applied after freeze so only adapters are trainable)
        self._lora_layers: nn.ModuleDict | None = None
        if lora_rank > 0 and freeze:
            self._apply_lora(lora_rank, lora_alpha, lora_targets or ["q_proj", "v_proj"])

        self._log_trainable()

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def _apply_lora(self, rank: int, alpha: int, targets: list[str]) -> None:
        self._lora_layers = nn.ModuleDict()
        scale = alpha / rank

        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(t in name for t in targets):
                continue

            safe = name.replace(".", "_")
            lora_a = nn.Linear(module.in_features, rank, bias=False)
            lora_b = nn.Linear(rank, module.out_features, bias=False)
            nn.init.kaiming_uniform_(lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(lora_b.weight)

            self._lora_layers[f"{safe}_a"] = lora_a
            self._lora_layers[f"{safe}_b"] = lora_b

            orig_fwd = module.forward

            def _make(orig, la, lb, s):
                def _fwd(x):
                    return orig(x) + lb(la(x.to(la.weight.dtype))) * s
                return _fwd

            module.forward = _make(orig_fwd, lora_a, lora_b, scale)

        log.info(f"  LoRA (rank={rank}, alpha={alpha}) applied to: {targets}")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_embed_layer(self) -> nn.Embedding:
        return self.model.model.embed_tokens

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run transformer layers on pre-built embeddings.

        Args:
            inputs_embeds: (B, S, hidden_size)
            position_ids:  (B, S) — auto-generated if None

        Returns:
            logits: (B, S, vocab_size) float32
        """
        # Call LlamaModel directly so we can pass custom position_ids
        # (needed for variable-length packed sequences). LlamaModel applies
        # gradient checkpointing per-layer and final norm before returning.
        h = self.model.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )
        logits = self.model.lm_head(h).float()
        return logits

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_adapter(self, path: str) -> None:
        """Save LoRA adapter weights (projector saved separately by SpeechAura)."""
        if self._lora_layers is None:
            log.warning("No LoRA layers to save.")
            return
        torch.save(self._lora_layers.state_dict(), path)
        log.info(f"Saved LoRA adapter → {path}")

    def load_adapter(self, path: str) -> None:
        if self._lora_layers is None:
            raise RuntimeError("No LoRA layers initialized — call with lora_rank > 0.")
        state = torch.load(path, map_location="cpu", weights_only=True)
        self._lora_layers.load_state_dict(state)
        log.info(f"Loaded LoRA adapter ← {path}")

    def save_full(self, path: str) -> None:
        """Save full LLM weights (for full fine-tune mode)."""
        torch.save(self.model.state_dict(), path)
        log.info(f"Saved full LLM → {path}")

    def load_full(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        log.info(f"Loaded full LLM ← {path}")

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _log_trainable(self) -> None:
        total  = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"  AuraLLM: {total:,} params total, {trainable:,} trainable")