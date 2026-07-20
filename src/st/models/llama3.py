import inspect
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from st.models.kvcache import KVcache
from transformers import AutoModelForCausalLM
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

block_size = 2048              # embedding dimension (hidden size)
intermediate_size = 5120        # feedforward intermediate size
vocab_size = 49152              # vocabulary size (overridden per model_presets entry)
n_layers = 36                   # number of transformer layers
n_heads = 32                    # number of attention heads
n_kv_heads = 8                  # number of key/value heads (GQA)
rope_theta = 500000             # RoPE base frequency
max_seq_len = 2048               # maximum supported sequence length
multiple_of = 256                # make SwiGLU hidden size a multiple of this
norm_eps = 1e-5                  # layernorm epsilon


# ------------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------------

@dataclass
class ModelArgs:
    dim: int = block_size
    intermediate_size: int = intermediate_size
    vocab_size: int = vocab_size
    n_layers: int = n_layers
    n_heads: int = n_heads
    n_kv_heads: int = n_kv_heads
    rope_theta: int = rope_theta
    multiple_of: int = multiple_of
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = norm_eps
    max_seq_len: int = max_seq_len


# ------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------

class LLamaRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (https://arxiv.org/abs/1910.07467).

    Normalizes only the variance (no re-centering) — the LLaMA paper found
    re-scaling invariance, not re-centering invariance, to be what matters
    for LayerNorm's stabilizing effect.
    """

    def __init__(self, hidden_size, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states.to(input_dtype)
        return self.weight * hidden_states


# ------------------------------------------------------------------
# Rotary position embedding (RoPE)
# ------------------------------------------------------------------

class LlamaRotaryEmbedding(nn.Module):
    """Rotary Position Embedding (https://arxiv.org/abs/2104.09864).

    LLaMA 3 raised the RoPE base frequency to 500,000 (from the original
    10,000) to better support longer contexts (Xiong et al. 2023).
    """
    def __init__(self, dim, max_position_embeddings=2048, base=500000, device=None, scaling_factor=1.0):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.dim = dim
        self.scaling_factor = scaling_factor
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # Force float32 since bfloat16 loses precision on long contexts
        # (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = torch.einsum("i, b j -> b j i", self.inv_freq, position_ids.float())
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def rotate_half(self, x):
        """
        Rotary Embedding helper function that rotates half the hidden dims
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids = None, unqueeze_dim = 1):
        """
        Apply Rotary Embedding to query and key tensors.
        Args:

        q[torch.Tensor]: query tensor of shape [bs, num_attention_heads, seq_len, head_size]
        k[torch.Tensor]: key tensor of shape [bs, num_attention_heads, seq_len, head_size]
        cos[torch.Tensor]: precomputed cosines of shape [bs, seq_len, head_size]
        sin[torch.Tensor]: precomputed sines of shape [bs, seq_len, head_size]
        position_ids[torch.Tensor]: position ids of shape [bs, seq_len]
            position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
            unsqueeze_dim[int]: dimension to unsqueeze cos and sin tensors

        Returns:
        tuple(torch.Tensor): comprising the query and key tensors after applying rotary embeddings.
   
        """
        cos = cos.unsqueeze(unqueeze_dim)
        sin = sin.unsqueeze(unqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed


# ------------------------------------------------------------------
# Attention (Grouped Query Attention)
# ------------------------------------------------------------------

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads for multi-head attention when num_kv_heads < num_heads.
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)

    """

    batch, n_kv_heads, seqlen, head_dim = hidden_states.shape

    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seqlen, head_dim)
    return hidden_states.reshape(batch, n_kv_heads * n_rep, seqlen, head_dim)


class LlamaAttention(nn.Module):
    """Grouped Query Attention (GQA, https://arxiv.org/abs/2305.13245) —
    num_kv_heads < num_heads, with key/value heads repeated (repeat_kv) to
    match the query head count before the attention computation.
    """
    def __init__(self, config: ModelArgs, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.dim
        self.num_heads = config.n_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.n_kv_heads
        self.n_kv_groups = self.num_heads // self.num_kv_heads
        self.max_postion_embeddings = config.max_seq_len
        self.rope_theta = config.rope_theta
        self.is_causal = True


        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                    f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                    f" and `num_heads`: {self.num_heads})."
                )


        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        self.rope = LlamaRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.max_postion_embeddings,
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
        
        # Project
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to [bsz, seqlen, num_heads, head_dim] 
        # Note: We keep seqlen at dim 1 for RoPE and Flash Attn compatibility
        query_states = query_states.view(bsz, seqlen, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        # 1. Generate RoPE frequencies
        # Adjust rope input: it usually expects [bsz, num_heads, seqlen, head_dim] or handled inside apply_rotary
        cos, sin = self.rope(value_states, position_ids)

        # 2. Apply RoPE to Query and the CURRENT Key shard
        # Transpose to [bs, heads, seq, dim] for the apply_rotary_pos_emb function
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        
        query_states, key_states = self.rope.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # 3. Handle KV Cache (Concatenate AFTER RoPE for the new token)
        if use_cache and cache is not None:
            value_states = value_states.transpose(1, 2) # [bs, kv_heads, seq, dim]
            
            if len(cache.key_list[self.layer_idx]) > 0:
                key_states_old, value_states_old = cache.get_key_values(self.layer_idx)
                key_states = torch.cat([key_states_old, key_states], dim=2)
                value_states = torch.cat([value_states_old, value_states], dim=2)
            
            cache.update(self.layer_idx, key_states, value_states)
        else:
            value_states = value_states.transpose(1, 2)

        # 4. Grouped Query Attention (Repeat KV heads)
        key_states = repeat_kv(key_states, self.n_kv_groups)
        value_states = repeat_kv(value_states, self.n_kv_groups)

        # 5. Attention Computation
        # If prefilling (seqlen > 1), we MUST use causal masking so prompt tokens don't attend to future prompt tokens.
        # If decoding (seqlen == 1), causal masking is irrelevant but SDPA/Flash might expect specific inputs.
        is_causal = seqlen > 1

        if FLASH_ATTN_AVAILABLE and query_states.is_cuda:
            # Flash Attention 2 requires [batch, seq, heads, dim]
            q = query_states.transpose(1, 2)
            k = key_states.transpose(1, 2)
            v = value_states.transpose(1, 2)
            
            attn_output = flash_attn_func(q, k, v, dropout_p=0.0, causal=is_causal)
            attn_output = attn_output.reshape(bsz, seqlen, self.hidden_size)
        else:
            # standard SDPA expects [bs, heads, seq, dim]
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, is_causal=is_causal
            )
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_size)

        return self.o_proj(attn_output)


# ------------------------------------------------------------------
# Feedforward (SwiGLU)
# ------------------------------------------------------------------

class LLamaMLP(nn.Module):
    """LLaMA feedforward network with SwiGLU activation."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hidden_size = config.dim
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.activation = nn.SiLU()

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = self.activation(gate) * up
        return self.down_proj(hidden)


# ------------------------------------------------------------------
# Transformer block (attn + MLP, pre-norm residual)
# ------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, layer_id: int, config: ModelArgs):
        super().__init__()
        self.hidden_state = config.dim
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
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self-Attention
        hidden_states = self.attn(
            hidden_states =hidden_states,
            position_ids=position_ids,
            use_cache=use_cache,
            cache=cache
        )

        hidden_states = residual + hidden_states

        # Feedforward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = hidden_states
        return outputs


# ------------------------------------------------------------------
# Full transformer stack (embeddings + layers + final norm)
# ------------------------------------------------------------------

class LlamaModel(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.config = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.gradient_checkpointing = False

        self.embed_tokens = nn.Embedding(params.vocab_size, params.dim)
        self.layers = nn.ModuleList(
            Block(layer_id, params) for layer_id in range(params.n_layers)
        )
        self.norm = LLamaRMSNorm(params.dim, eps=params.norm_eps)

    def forward(self, tokens: torch.Tensor | None = None, position_ids: torch.LongTensor | None = None, use_cache: bool = False, cache: KVcache = None, inputs_embeds: torch.Tensor | None = None):
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.embed_tokens(tokens)

        def create_custom_forward(layer):
            def custom_forward(*args):
                return layer(*args)
            return custom_forward

        for layer in self.layers:
            if self.gradient_checkpointing and not use_cache:
                h = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    h,
                    position_ids,
                    use_cache,
                    cache,
                    use_reentrant=False
                )
            else:
                h = layer(
                    h,
                    position_ids,
                    use_cache=use_cache,
                    cache=cache
                )
        h = self.norm(h)
        return h


# ------------------------------------------------------------------
# LM wrapper (lm_head, forward w/ CE loss, KV-cache generation, checkpoint I/O)
# ------------------------------------------------------------------

class LlamaTransformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for the model."""
        self.gradient_checkpointing = True
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the model."""
        self.gradient_checkpointing = False
        self.model.gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        targets: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        cache: Optional[KVcache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        if inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
            device = inputs_embeds.device
        else:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device

        if use_cache and len(cache.key_list[0]) > 0:
            start_pos = cache.key_list[0].shape[2]
            position_ids = torch.arange(
                start_pos,
                start_pos + seq_len,
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0).expand(batch_size, seq_len)
        else:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        
        outputs = self.model(
            tokens=input_ids,
            position_ids = position_ids,
            use_cache=use_cache,
            cache=cache,
            inputs_embeds=inputs_embeds,
        )  # [bs, seq_len, hidden_size]

        logits = self.lm_head(outputs)  # [bs, seq_len, vocab_size]
        logits = logits.float()  # convert to float32 for numerical stability
        if targets is None:
            return logits

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ):
        """Greedy (or temperature-scaled) token generation using KV caching."""
        from st.models.kvcache import KVcache
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize Cache
        cache = KVcache(self.config.n_layers)
        
        # 1. Prefill step
        logits = self.forward(input_ids, use_cache=True, cache=cache)
        
        # 2. Extract out the last token logits
        next_token_logits = logits[:, -1, :]
        
        # Temperature & Greedy decoding for baseline
        if temperature != 1.0 and temperature > 0:
            next_token_logits = next_token_logits / temperature
        
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
        generated_tokens = [next_token]
        
        # Create an early stopping mask
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # 3. Decoding Loop
        for _ in range(max_new_tokens - 1):
            logits = self.forward(next_token, use_cache=True, cache=cache)
            next_token_logits = logits[:, -1, :]
            
            if temperature != 1.0 and temperature > 0:
                next_token_logits = next_token_logits / temperature
                
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
            if eos_token_id is not None:
                # If a sequence hits eos_token_id, we replace subsequent tokens with eos_token_id (or pad)
                # and if all sequences are finished, we can break early
                is_eos = (next_token.squeeze(1) == eos_token_id)
                next_token[~unfinished_sequences] = eos_token_id
                unfinished_sequences = unfinished_sequences & ~is_eos
                
            generated_tokens.append(next_token)
            
            if eos_token_id is not None and not unfinished_sequences.any():
                break
                
        # Concatenate generated tokens and attach to input context
        generated_tensor = torch.cat(generated_tokens, dim=1)
        return torch.cat([input_ids, generated_tensor], dim=1)



    @classmethod
    def from_custom_pretrained(cls,ckp_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ckp = torch.load(ckp_path, map_location=device)
        config = ckp['config']
        model = LlamaTransformer(config)
        model.load_state_dict(ckp['model'])
        return model

    @classmethod
    def from_pretrained(cls,model_type):
        print("loading weights from pretrained model... ", model_type )
        model_hf = AutoModelForCausalLM.from_pretrained(model_type)
        
        dim = model_hf.config.hidden_size
        intermediate_size = model_hf.config.intermmediate_size
        n_layers = model_hf.config.num_hidden_layers
        n_heads = model_hf.config.num_attention_heads
        n_kv_heads = model_hf.config.num_key_value_heads 
        vocab_size = model_hf.config.vocab_size
        rope_theta = model_hf.config.rope_theta
        max_seq_len = model_hf.config.max_position_embeddings



        config = ModelArgs(
            dim=dim,
            intermediate_size=intermediate_size,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            vocab_size=vocab_size,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
        )
        model = LlamaTransformer(config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if k.endswith('bias')] # discard bias terms
        print("total parameters to load: ", len(sd_keys))

        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type,master_process):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
