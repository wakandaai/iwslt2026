"""
depth_transformer.py — RQ-Transformer depth decoder + codec embedding/head
bundle for Aura-TTS.

Decomposition (Moshi / RQ-Transformer, Lee et al. 2022):
    p(frame_t | past) = p(c_t^0 | h_t) · ∏_{k=1}^{K-1} p(c_t^k | h_t, c_t^{<k})

  - The TEMPORAL transformer (Aura, via AuraLLM.forward_hidden) produces one
    hidden state h_t per frame. It carries all long-range dependency. Cheap on
    sequence length: one position per frame (~50 Hz at DAC-16kHz).
  - The DEPTH transformer (this module, small) autoregresses over the K=12
    codebooks WITHIN a frame, conditioned on h_t and the already-emitted finer
    codes of that same frame. Its sequence length is K, not T — the ×K blowup
    never touches Aura's context window.

Two pieces here:
  CodecEmbeddings  K embedding tables (one per codebook), summed per frame to
                   form the temporal-transformer frame input; plus per-codebook
                   input embeddings for the depth transformer's AR steps.
  DepthTransformer small causal transformer over the K depth positions, with K
                   output heads (one Linear per codebook -> 1024 logits).

Shapes throughout: B=batch (frames flattened for training), K=n_codebooks,
card=cardinality (1024), d=depth model dim, D=Aura hidden dim.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================================
#  Codec embeddings (input side)
# ============================================================================

class CodecEmbeddings(nn.Module):
    """K per-codebook embedding tables.

    Used in two places:
      1. Temporal frame input: sum the K codebook embeddings of a frame and
         project to Aura's hidden dim D. This is the standard codec-LM frame
         representation (VALL-E / MusicGen / Moshi all sum codebook embeddings).
      2. Depth-transformer input: the embedding of codebook k feeds the depth
         step that predicts codebook k+1. Same tables reused (the depth model
         consumes finer codes it just predicted).

    A learned per-codebook positional bias is added inside the depth model, not
    here, so these tables are pure content embeddings.
    """

    def __init__(self, n_codebooks: int, cardinality: int, depth_dim: int,
                 aura_dim: int):
        super().__init__()
        self.K = n_codebooks
        self.card = cardinality
        self.depth_dim = depth_dim
        self.aura_dim = aura_dim

        # One table per codebook. Separate tables (not one shared 12*1024) so a
        # given id means different things in different codebooks (RVQ levels are
        # not interchangeable).
        self.tables = nn.ModuleList(
            nn.Embedding(cardinality, depth_dim) for _ in range(n_codebooks)
        )
        # Project the summed frame embedding (depth_dim) up to Aura's dim for
        # the temporal transformer input.
        self.to_aura = nn.Linear(depth_dim, aura_dim)

    def frame_embedding(self, codes_BK: torch.Tensor) -> torch.Tensor:
        """Sum the K codebook embeddings of each frame, project to Aura dim.

        Args:
            codes_BK: (B, K) long — the K codebook ids for B frames.
        Returns:
            (B, aura_dim) — temporal-transformer input embedding per frame.
        """
        B, K = codes_BK.shape
        assert K == self.K, f"expected {self.K} codebooks, got {K}"
        acc = self.tables[0](codes_BK[:, 0])
        for k in range(1, self.K):
            acc = acc + self.tables[k](codes_BK[:, k])
        return self.to_aura(acc)

    def codebook_embedding(self, codes_B: torch.Tensor, k: int) -> torch.Tensor:
        """Embed codebook-k ids for the depth transformer. (B,) -> (B, depth_dim)."""
        return self.tables[k](codes_B)


# ============================================================================
#  Depth transformer (output side)
# ============================================================================

class DepthTransformer(nn.Module):
    """Small causal transformer over the K codebook positions of one frame.

    At depth step k (0-indexed), the input sequence is:
        [ proj(h_t),  emb_0(c^0),  emb_1(c^1),  ...,  emb_{k-1}(c^{k-1}) ]
    and the model predicts c^k from the hidden state at the last position.
    Step 0 sees only proj(h_t) and predicts c^0 (the coarsest code).

    Training runs all K steps in parallel with a causal mask (teacher forcing):
    feed [proj(h_t), emb_0(c^0), ..., emb_{K-2}(c^{K-2})] (length K) and predict
    [c^0, c^1, ..., c^{K-1}] at the K positions via the K heads.

    Args:
        n_codebooks:  K (12 for DAC-16kHz)
        cardinality:  per-codebook vocab (1024)
        aura_dim:     D, dim of the temporal hidden state h_t
        depth_dim:    d, the (small) depth model width
        n_layers:     depth transformer depth (Moshi used ~6; small)
        n_heads:      attention heads in the depth transformer
        codec_emb:    shared CodecEmbeddings (input tables reused here)
    """

    def __init__(
        self,
        n_codebooks: int,
        cardinality: int,
        aura_dim: int,
        depth_dim: int,
        n_layers: int,
        n_heads: int,
        codec_emb: CodecEmbeddings,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.K = n_codebooks
        self.card = cardinality
        self.depth_dim = depth_dim
        self.codec_emb = codec_emb

        # Condition: project Aura's hidden state down to the depth width. This
        # is the "step-0 input" / global frame conditioning (Moshi's design).
        self.cond_proj = nn.Linear(aura_dim, depth_dim)

        # Learned positional embedding over the K depth positions (which RVQ
        # level we're at). Tiny — K positions.
        self.depth_pos = nn.Parameter(torch.zeros(n_codebooks, depth_dim))
        nn.init.normal_(self.depth_pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=depth_dim, nhead=n_heads,
            dim_feedforward=depth_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False,
        )

        # K output heads: head k maps depth position k's hidden state -> card logits.
        self.heads = nn.ModuleList(
            nn.Linear(depth_dim, cardinality) for _ in range(n_codebooks)
        )

        # Causal mask over K positions (depth step k attends to <= k).
        mask = torch.triu(torch.full((n_codebooks, n_codebooks), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def _depth_inputs(self, h_t: torch.Tensor, codes_BK: torch.Tensor) -> torch.Tensor:
        """Assemble the (B, K, depth_dim) teacher-forcing input sequence.

        Position 0: conditioning proj(h_t).
        Position k (1..K-1): embedding of codebook (k-1)'s true id.
        Plus depth positional embedding at every position.
        """
        B = h_t.size(0)
        cond = self.cond_proj(h_t)                       # (B, depth_dim)
        seq = [cond.unsqueeze(1)]                        # pos 0
        for k in range(self.K - 1):                      # pos 1..K-1
            seq.append(self.codec_emb.codebook_embedding(codes_BK[:, k], k).unsqueeze(1))
        x = torch.cat(seq, dim=1)                         # (B, K, depth_dim)
        return x + self.depth_pos.unsqueeze(0)           # add depth PE

    def forward(self, h_t: torch.Tensor, codes_BK: torch.Tensor) -> torch.Tensor:
        """Training forward: predict all K codebooks of a frame in parallel.

        Args:
            h_t:      (B, aura_dim) temporal hidden states (one per frame; B is
                      the flattened count of supervised frames in the batch).
            codes_BK: (B, K) long — the frame's true codebook ids (teacher input
                      + targets are derived from these by the loss).
        Returns:
            logits: (B, K, cardinality) — logits[:, k] predicts codebook k.
        """
        x = self._depth_inputs(h_t, codes_BK)            # (B, K, depth_dim)
        x = self.transformer(x, mask=self.causal_mask)   # causal over depth
        # Per-position head: position k -> codebook-k logits.
        out = torch.stack(
            [self.heads[k](x[:, k]) for k in range(self.K)], dim=1
        )                                                 # (B, K, card)
        return out

    @torch.inference_mode()
    def generate_frame(self, h_t: torch.Tensor, greedy: bool = True,
                       temperature: float = 1.0,
                       top_k: int | None = None,
                       top_p: float | None = None,
                       eos_id: int | None = None) -> torch.Tensor:
        """Autoregressive decode of one frame's K codes from a single h_t.

        Args:
            h_t: (B, aura_dim) — one temporal hidden state per item (B usually 1
                 at TTS inference, but batched decode is supported).
            greedy: take the argmax. Note: greedy decoding of RVQ codec LMs
                 reliably collapses codebook 0 to the lowest-energy (silence)
                 token and then never escapes — prefer sampling with top_k/top_p.
            top_k: keep only the top_k logits before sampling (None = no cap).
            top_p: nucleus sampling — keep the smallest set of codes whose
                 cumulative prob >= top_p (None = no cap). Applied after top_k.
            eos_id: if given, the EOS code is allowed only in codebook 0 (the
                 stop signal); its logit is masked out for k >= 1 so a real frame
                 never contains EOS in a finer codebook (which DAC can't decode).
        Returns:
            codes_BK: (B, K) long — sampled codebook ids for the frame.
        """
        B = h_t.size(0)
        device = h_t.device
        cond = self.cond_proj(h_t)                        # (B, depth_dim)
        emitted: list[torch.Tensor] = []                  # each (B,)

        # Build the depth sequence incrementally. We re-run the small depth
        # transformer over the growing prefix each step (K is tiny, so this is
        # cheap and avoids a second KV cache).
        for k in range(self.K):
            seq = [cond.unsqueeze(1)]
            for j in range(k):
                seq.append(
                    self.codec_emb.codebook_embedding(emitted[j], j).unsqueeze(1)
                )
            x = torch.cat(seq, dim=1)                     # (B, k+1, depth_dim)
            x = x + self.depth_pos[: k + 1].unsqueeze(0)
            mask = self.causal_mask[: k + 1, : k + 1]
            x = self.transformer(x, mask=mask)
            logits = self.heads[k](x[:, -1])              # (B, card)
            if eos_id is not None and k > 0:
                logits[:, eos_id] = float("-inf")         # EOS only in codebook 0
            if greedy:
                nxt = logits.argmax(dim=-1)               # (B,)
            else:
                logits = logits / max(temperature, 1e-5)
                logits = _truncate_logits(logits, top_k, top_p)
                probs = logits.softmax(dim=-1)
                nxt = torch.multinomial(probs, 1).squeeze(-1)
            emitted.append(nxt)
        return torch.stack(emitted, dim=1)                # (B, K)


def _truncate_logits(logits: torch.Tensor, top_k: int | None,
                     top_p: float | None) -> torch.Tensor:
    """Mask logits outside the top-k / top-p (nucleus) set with -inf. (B, V)."""
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = logits.topk(k, dim=-1).values[..., -1, None]   # (B, 1)
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Keep tokens up to and including the one that crosses top_p.
        remove = cum - sorted_logits.softmax(dim=-1) >= top_p
        remove_orig = remove.scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(remove_orig, float("-inf"))
    return logits