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

# Language token IDs (Aura reserved slots).
# NOTE: These were assigned for the old 49157-vocab tokenizer. The new Aura
# tokenizer (64k) uses a different special-token layout — see
# train_tokenizer.py in the Aura repo: <pad>=0, <s>=1, </s>=2, <unk>=3,
# then 4..28 are 25 language tokens in sorted-alphabetical order:
#   4: <|afr_Latn|>   5: <|amh_Ethi|>   6: <|arb_Arab|>  7: <|bem_Latn|>
#   8: <|eng_Latn|>   9: <|fon_Latn|>  10: <|fra_Latn|> 11: <|hau_Latn|>
#  12: <|ibo_Latn|>  13: <|kin_Latn|>  14: <|lin_Latn|> 15: <|lug_Latn|>
#  16: <|nya_Latn|>  17: <|plt_Latn|>  18: <|por_Latn|> 19: <|sna_Latn|>
#  20: <|som_Latn|>  21: <|sot_Latn|>  22: <|swh_Latn|> 23: <|tir_Ethi|>
#  24: <|tsn_Latn|>  25: <|wol_Latn|>  26: <|xho_Latn|> 27: <|yor_Latn|>
#  28: <|zul_Latn|>
# Then 29: <|transcribe|>, 30: <|translate|>, 31: <|synthesize|>, 32: <|prev|>,
# 33: <|im_start|>, 34: <|im_end|>, 35..37: chat roles, 38: <|eot|>,
# 39..54: <|reserved_0|>..<|reserved_15|>.
LANG_MAP: dict[str, int] = {
    "afr": 4,  "afr_Latn": 4,
    "amh": 5,  "amh_Ethi": 5,
    "arb": 6,  "arb_Arab": 6,
    "bemba": 7, "bem": 7, "bem_Latn": 7,
    "english": 8, "eng": 8, "eng_Latn": 8,
    "fon": 9,  "fon_Latn": 9,
    "french": 10, "fra": 10, "fra_Latn": 10,
    "hausa": 11, "hau": 11, "hau_Latn": 11,
    "igbo": 12,  "ibo": 12, "ibo_Latn": 12,
    "kinyarwanda": 13, "kin": 13, "kin_Latn": 13,
    "lingala": 14, "lin": 14, "lin_Latn": 14,
    "luganda": 15, "lug": 15, "lug_Latn": 15,
    "chichewa": 16, "nya": 16, "nya_Latn": 16,
    "malagasy": 17, "plt": 17, "plt_Latn": 17, "mlg": 17,
    "portuguese": 18, "por": 18, "por_Latn": 18,
    "shona": 19, "sna": 19, "sna_Latn": 19,
    "somali": 20, "som": 20, "som_Latn": 20,
    "sotho": 21, "sot": 21, "sot_Latn": 21, "sesotho": 21, "nso": 21,
    "swahili": 22, "swh": 22, "swh_Latn": 22, "sw": 22,
    "tigrinya": 23, "tir": 23, "tir_Ethi": 23, "tigrigna": 23,
    "tswana": 24, "tsn": 24, "tsn_Latn": 24,
    "wolof": 25, "wol": 25, "wol_Latn": 25,
    "xhosa": 26, "xho": 26, "xho_Latn": 26,
    "yoruba": 27, "yor": 27, "yor_Latn": 27,
    "zulu": 28, "zul": 28, "zul_Latn": 28,
}

# Task / structure tokens (new tokenizer layout, 64k vocab).
# The new Aura tokenizer ships these named tokens:
#   <|transcribe|>=29, <|translate|>=30, <|synthesize|>=31, <|prev|>=32,
#   <|im_start|>=33, <|im_end|>=34, <|system|>=35, <|user|>=36,
#   <|assistant|>=37, <|eot|>=38, <|reserved_0|>..<|reserved_15|>=39..54.
#
# We map our speech-pipeline roles onto this layout. TASK_ASR uses
# <|transcribe|> (semantic fit). TRANSCRIPT_START / AUDIO / TASK_COT have no
# semantic fit so they go into reserved slots. We keep TASK_ASR and
# TRANSCRIPT_START on DIFFERENT IDs — they appear at different structural
# positions in _build_inputs() (task token then audio frames then TS), and
# collapsing them would force the model to disambiguate by position alone.
AUDIO_PLACEHOLDER_ID = 39   # <|reserved_0|>
TRANSCRIPT_START_ID  = 40   # <|reserved_1|>
TASK_ASR_ID          = 29   # <|transcribe|>
TASK_COT_ID          = 41   # <|reserved_2|>
TRANSLATE_START_ID   = 30   # <|translate|>


def verify_special_token_ids(tokenizer) -> None:
    """Verify reserved-token assignments still point at the expected tokens.

    With the new 64k Aura tokenizer the special-token names are concrete
    (<|transcribe|>, <|translate|>, etc.) so we check those instead of just
    'starts with <| ends with |>'.
    """
    expected = {
        AUDIO_PLACEHOLDER_ID: ("audio_placeholder", "<|reserved_0|>"),
        TRANSCRIPT_START_ID:  ("transcript_start",  "<|reserved_1|>"),
        TASK_ASR_ID:          ("task_asr",          "<|transcribe|>"),
        TASK_COT_ID:          ("task_cot",          "<|reserved_2|>"),
        TRANSLATE_START_ID:   ("translate_start",   "<|translate|>"),
    }
    for tok_id, (role, want) in expected.items():
        decoded = tokenizer.decode([tok_id], skip_special_tokens=False)
        # Strip whitespace that some tokenizers prepend.
        decoded = decoded.strip()
        if decoded != want:
            log.warning(
                f"Token id {tok_id} (role={role}) decodes to {decoded!r}, "
                f"expected {want!r}. Either the tokenizer changed or "
                f"the special-token IDs in aura.py need updating."
            )


class AuraLLM(nn.Module):
    """Thin wrapper around the Aura LlamaTransformer.

    Args:
        ckpt_path:       Path to Aura model.pt checkpoint.
        tokenizer_path:  Path to Aura tokenizer.json.
        size:            Model size key for ARCH_PRESETS (default "1b").
        freeze:          Freeze all LLM weights on init.
        lora_rank:       LoRA rank. 0 = no LoRA.
        lora_alpha:      LoRA scaling alpha.
        lora_targets:    Module name substrings to apply LoRA to.
        vocab_size:      Override vocab size. If None, read from tokenizer.
        max_seq_len:     Max position embeddings for RoPE. Default 1024.
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
        vocab_size: int | None = None,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        from transformers import PreTrainedTokenizerFast

        # --- Tokenizer ---
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        special = {}
        if self.tokenizer.eos_token is None:
            special["eos_token"] = "</s>"
        if self.tokenizer.pad_token is None:
            special["pad_token"] = "<pad>"
        if self.tokenizer.bos_token is None:
            special["bos_token"] = "<s>"
        if special:
            self.tokenizer.add_special_tokens(special)

        verify_special_token_ids(self.tokenizer)

        self.bos_id  = self.tokenizer.bos_token_id or 1
        self.eos_id  = self.tokenizer.eos_token_id or 2
        self.audio_token_id      = AUDIO_PLACEHOLDER_ID
        self.transcript_start_id = TRANSCRIPT_START_ID
        self.task_asr_id         = TASK_ASR_ID
        self.task_cot_id         = TASK_COT_ID
        self.translate_start_id  = TRANSLATE_START_ID

        # --- LLM ---
        # llama3 and model_factory live in st/models/ alongside this file.
        from st.models.llama3 import LlamaTransformer
        from st.models.model_factory import build_model_config

        # vocab_size: use override, else read from tokenizer. The new Aura
        # uses 64k; pad up to a multiple of 64 for efficient matmuls if not
        # already, matching what training did.
        if vocab_size is None:
            vocab_size = self.tokenizer.vocab_size

        cfg = build_model_config(
            model_type="llama-iwslt", size=size,
            vocab_size=vocab_size, max_seq_len=max_seq_len,
        )
        self.model = LlamaTransformer(cfg)
        self.hidden_size: int = cfg.dim
        self.vocab_size: int  = cfg.vocab_size
        self.n_layers: int    = cfg.n_layers

        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(ckpt_path, device="cpu")
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt)
            del ckpt

        # Strip torch.compile and DDP prefixes.
        state = {
            k.replace("_orig_mod.", "").replace("module.", ""): v
            for k, v in state.items()
        }
        self.model.load_state_dict(state)
        log.info(f"Loaded Aura-{size} from {ckpt_path} (vocab={vocab_size})")
        del state

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
        B, S, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = torch.arange(S, device=inputs_embeds.device).unsqueeze(0).expand(B, -1)

        h = inputs_embeds
        for layer in self.model.model.layers:
            h = layer(h, position_ids=position_ids)
        h = self.model.model.norm(h)
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