"""
llama3.py — Llama-style transformer for the Aura multilingual base LLM.

Vendored into the speech repo from the Aura training codebase. Architecture:
  - RMSNorm
  - Rotary positional embeddings (RoPE, theta=500_000)
  - Grouped-query attention (GQA) with configurable n_kv_heads
  - SwiGLU MLP
  - Pre-norm residual blocks

Flash Attention is used when importable; otherwise SDPA falls back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from st.models.kvcache import KVcache

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


# ============================================================================
#  Model config
# ============================================================================

@dataclass
class ModelArgs:
    dim: int = 2048
    intermediate_size: int = 5120
    vocab_size: int = 64000
    n_layers: int = 36
    n_heads: int = 32
    n_kv_heads: int = 8
    rope_theta: int = 500_000
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_seq_len: int = 2048


# ============================================================================
#  RMSNorm
# ============================================================================

class LLamaRMSNorm(nn.Module):
    """Root-mean-square layer normalization. https://arxiv.org/abs/1910.07467"""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        # Variance in fp32 for stability, then cast back.
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)


# ============================================================================
#  Rotary positional embeddings
# ============================================================================

class LlamaRotaryEmbedding(nn.Module):
    """RoPE with configurable base frequency.
    Llama-3 uses theta=500_000 to support longer contexts.
    https://arxiv.org/abs/2104.09864"""

    def __init__(self, dim: int, max_position_embeddings: int = 2048,
                 base: int = 500_000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor):
        """Return (cos, sin) for the given positions, in x's dtype."""
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        # Force fp32 for the trig functions; bf16 loses precision on long contexts.
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = torch.einsum("i, b j -> b j i", self.inv_freq, position_ids.float())
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor,
                             cos: torch.Tensor, sin: torch.Tensor,
                             unsqueeze_dim: int = 1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads from n_kv_heads to n_kv_heads * n_rep, matching n_heads.
    Equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep)."""
    batch, n_kv_heads, seqlen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, n_kv_heads, n_rep, seqlen, head_dim)
    return hidden_states.reshape(batch, n_kv_heads * n_rep, seqlen, head_dim)


# ============================================================================
#  Attention (GQA + Flash Attention)
# ============================================================================

class LlamaAttention(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.dim
        self.num_heads = config.n_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.n_kv_heads
        self.n_kv_groups = self.num_heads // self.num_kv_heads
        self.max_position_embeddings = config.max_seq_len
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads "
                f"(got hidden_size={self.hidden_size}, num_heads={self.num_heads})")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rope = LlamaRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        cache: KVcache = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.shape

        # Project to Q, K, V.
        query_states = self.q_proj(hidden_states).view(bsz, seqlen, self.num_heads, self.head_dim)
        key_states   = self.k_proj(hidden_states).view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        # RoPE expects [bs, heads, seq, dim].
        cos, sin = self.rope(value_states, position_ids)
        query_states = query_states.transpose(1, 2)
        key_states   = key_states.transpose(1, 2)
        query_states, key_states = self.rope.apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        # KV cache handling (inference only).
        if use_cache and cache is not None:
            value_states = value_states.transpose(1, 2)
            if len(cache.key_list[self.layer_idx]) > 0:
                k_old, v_old = cache.get_key_values(self.layer_idx)
                key_states = torch.cat([k_old, key_states], dim=2)
                value_states = torch.cat([v_old, value_states], dim=2)
            cache.update(self.layer_idx, key_states, value_states)
        else:
            value_states = value_states.transpose(1, 2)

        # GQA: expand KV heads to match Q heads.
        key_states = repeat_kv(key_states, self.n_kv_groups)
        value_states = repeat_kv(value_states, self.n_kv_groups)

        # Causal mask only needed during prefill (seqlen > 1); decode is single-token.
        is_causal = seqlen > 1

        if FLASH_ATTN_AVAILABLE and query_states.is_cuda:
            # Flash Attention 2 layout: [batch, seq, heads, dim].
            q = query_states.transpose(1, 2)
            k = key_states.transpose(1, 2)
            v = value_states.transpose(1, 2)
            attn_output = flash_attn_func(q, k, v, dropout_p=0.0, causal=is_causal)
            attn_output = attn_output.reshape(bsz, seqlen, self.hidden_size)
        else:
            # SDPA fallback. Layout: [bs, heads, seq, dim].
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states, is_causal=is_causal)
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_size)

        return self.o_proj(attn_output)


# ============================================================================
#  MLP (SwiGLU)
# ============================================================================

class LLamaMLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hidden_size = config.dim
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(gate_proj(x)) * up_proj(x), then down_proj.
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
#  Transformer block (pre-norm)
# ============================================================================

class Block(nn.Module):
    def __init__(self, layer_id: int, config: ModelArgs):
        super().__init__()
        self.attn = LlamaAttention(config, layer_idx=layer_id)
        self.mlp = LLamaMLP(config)
        self.input_layernorm = LLamaRMSNorm(config.dim, eps=config.norm_eps)
        self.post_attention_layernorm = LLamaRMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        cache: KVcache = None,
    ) -> torch.Tensor:
        # Self-attention with residual.
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            use_cache=use_cache,
            cache=cache,
        )
        hidden_states = residual + hidden_states

        # MLP with residual.
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


# ============================================================================
#  Backbone (stack of blocks + embedding + final norm)
# ============================================================================

class LlamaModel(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.config = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.embed_tokens = nn.Embedding(params.vocab_size, params.dim)
        self.layers = nn.ModuleList(
            Block(layer_id, params) for layer_id in range(params.n_layers))
        self.norm = LLamaRMSNorm(params.dim, eps=params.norm_eps)
        # Gradient checkpointing toggle. When True (and training, no KV cache),
        # layer forwards are wrapped in torch.utils.checkpoint, recomputing
        # activations during backward instead of storing them. ~30-50%
        # activation-memory savings at ~30% throughput cost. Off by default.
        # Disabled automatically during eval / inference / no_grad contexts.
        self.gradient_checkpointing = False

    def forward(
        self,
        tokens: torch.Tensor | None = None,
        position_ids: torch.LongTensor = None,
        use_cache: bool = False,
        cache: KVcache = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.embed_tokens(tokens)
        # Only checkpoint during training (no KV cache) and when in a
        # grad-tracking context. Eval/inference always takes the direct path.
        do_checkpoint = (self.gradient_checkpointing and self.training
                         and not use_cache and torch.is_grad_enabled())
        for layer in self.layers:
            if do_checkpoint:
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, position_ids, use_cache, cache,
                    use_reentrant=False,
                )
            else:
                h = layer(h, position_ids, use_cache=use_cache, cache=cache)
        return self.norm(h)


# ============================================================================
#  Top-level transformer (adds embedding tying + lm_head + training/eval glue)
# ============================================================================

class LlamaTransformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def gradient_checkpointing_enable(self):
        """Trade ~30% throughput for ~30-50% lower activation memory."""
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        targets: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        cache: Optional[KVcache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        # Resolve batch/seq from whichever input was provided.
        if inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
            device = inputs_embeds.device
        else:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device

        # Position ids: contiguous; if decoding from a non-empty cache, start
        # from the cache's current length so RoPE indexes the correct positions.
        if use_cache and cache is not None and len(cache.key_list[0]) > 0:
            start_pos = cache.key_list[0].shape[2]
            position_ids = torch.arange(
                start_pos, start_pos + seq_len,
                dtype=torch.long, device=device,
            ).unsqueeze(0).expand(batch_size, seq_len)
        else:
            position_ids = torch.arange(
                seq_len, device=device,
            ).unsqueeze(0).expand(batch_size, seq_len)

        hidden_states = self.model(
            tokens=input_ids,
            position_ids=position_ids,
            use_cache=use_cache,
            cache=cache,
            inputs_embeds=inputs_embeds,
        )
        # Cast logits to fp32 for numerical stability in cross_entropy.
        logits = self.lm_head(hidden_states).float()
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """Greedy generation with KV-cache. Sampling is handled outside; this
        method exists for quick smoke tests. SpeechAura.generate() has its own
        decode loop and does NOT call this."""
        from st.models.kvcache import KVcache as _KV
        batch_size = input_ids.shape[0]
        device = input_ids.device
        cache = _KV(self.config.n_layers)

        # Prefill.
        logits = self.forward(input_ids, use_cache=True, cache=cache)
        next_logits = logits[:, -1, :]
        if temperature != 1.0 and temperature > 0:
            next_logits = next_logits / temperature
        next_token = torch.argmax(next_logits, dim=-1).unsqueeze(1)
        generated = [next_token]
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Decode loop.
        for _ in range(max_new_tokens - 1):
            logits = self.forward(next_token, use_cache=True, cache=cache)
            next_logits = logits[:, -1, :]
            if temperature != 1.0 and temperature > 0:
                next_logits = next_logits / temperature
            next_token = torch.argmax(next_logits, dim=-1).unsqueeze(1)
            if eos_token_id is not None:
                is_eos = (next_token.squeeze(1) == eos_token_id)
                next_token[~unfinished] = eos_token_id
                unfinished = unfinished & ~is_eos
            generated.append(next_token)
            if eos_token_id is not None and not unfinished.any():
                break

        return torch.cat([input_ids] + generated, dim=1)