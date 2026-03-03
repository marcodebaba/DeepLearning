"""
GPT model — implemented from scratch.

Architecture (follows GPT-2 design):
  Token Embedding
  + Positional Embedding
  -> N x TransformerBlock
       -> LayerNorm
       -> CausalSelfAttention   (masked multi-head attention)
       -> residual add
       -> LayerNorm
       -> FeedForward           (MLP with GELU)
       -> residual add
  -> LayerNorm
  -> Linear head (project to vocab_size logits)

Key design choices:
  - Pre-LayerNorm (norm before attention/FFN, not after) — more stable training
  - Causal mask: each position can only attend to itself and earlier positions
  - Weight tying: token embedding and output head share the same weight matrix
"""

import math
import torch
import torch.nn as nn
from torch import Tensor

from config import GPTConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with a causal (autoregressive) mask.

    For a sequence of length T, position i can only attend to positions 0..i.
    This is enforced by adding -inf to all "future" positions before softmax.

    Shape walkthrough (B = batch, T = seq_len, C = d_model):
      input x : (B, T, C)
      Q, K, V : (B, T, C)  -- after projection
      split to n_heads: (B, n_heads, T, head_dim)  where head_dim = C // n_heads
      attn scores: (B, n_heads, T, T)
      output  : (B, T, C)
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = math.sqrt(self.head_dim)  # attention score scaling factor

        # Single fused projection for Q, K, V (3x faster than 3 separate Linear layers)
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.out_dropout = nn.Dropout(cfg.dropout)

        # Causal mask: upper-triangle is masked out (future positions)
        # Registered as buffer so it moves with .to(device) automatically
        mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length))
        self.register_buffer("mask", mask.view(1, 1, cfg.context_length, cfg.context_length))

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape

        # Project to Q, K, V then split into 3 tensors
        qkv = self.qkv_proj(x)                         # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)                 # each (B, T, C)

        # Reshape into (B, n_heads, T, head_dim)
        def split_heads(t: Tensor) -> Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention scores: (B, n_heads, T, T)
        scores = (q @ k.transpose(-2, -1)) / self.scale

        # Apply causal mask: mask out future positions with -inf
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # Softmax over last dim → attention weights
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # Weighted sum of values: (B, n_heads, T, head_dim)
        out = weights @ v

        # Merge heads: (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network (applied independently to each token).

    Two linear layers with GELU activation:
      d_model -> 4 * d_model -> d_model

    The 4x expansion is a standard GPT design choice — it gives the model
    more capacity to transform each token's representation.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One Transformer block = Pre-LN Attention + Pre-LN FFN, both with residual connections.

    Pre-LayerNorm (GPT-2 style):
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    vs. original Transformer (Post-LayerNorm):
        x = LayerNorm(x + Attention(x))
        x = LayerNorm(x + FFN(x))

    Pre-LN makes gradients flow more cleanly through the residual path.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))   # attention sub-layer
        x = x + self.ffn(self.ln2(x))    # feed-forward sub-layer
        return x


# ---------------------------------------------------------------------------
# Full GPT model
# ---------------------------------------------------------------------------

class GPTModel(nn.Module):
    """
    Complete GPT language model.

    Input:  token indices  (B, T)  — integers in [0, vocab_size)
    Output: logits         (B, T, vocab_size)  — raw scores for next-token prediction

    During autoregressive generation, we feed the last `context_length` tokens,
    get the logits for the *last* position, sample the next token, append it,
    and repeat.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        self.ln_final = nn.LayerNorm(cfg.d_model)
        # Project back to vocabulary — no bias, weight-tied with token embedding
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: share the embedding matrix with the output projection.
        # Saves parameters and often improves performance.
        self.head.weight = self.token_emb.weight

        # Initialize weights following GPT-2 paper
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: Tensor) -> Tensor:
        """
        idx: (B, T) token indices
        returns: (B, T, vocab_size) logits
        """
        B, T = idx.shape
        assert T <= self.cfg.context_length, (
            f"Sequence length {T} exceeds context_length {self.cfg.context_length}"
        )

        device = idx.device
        positions = torch.arange(T, device=device)          # (T,)

        # Sum token and positional embeddings
        x = self.token_emb(idx) + self.pos_emb(positions)  # (B, T, d_model)
        x = self.emb_dropout(x)

        x = self.blocks(x)       # N Transformer blocks
        x = self.ln_final(x)     # final layer norm
        logits = self.head(x)    # (B, T, vocab_size)

        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Loss helper
# ---------------------------------------------------------------------------

def compute_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Cross-entropy loss for next-token prediction.

    logits : (B, T, vocab_size)
    targets: (B, T)  — each value is the index of the correct next token
    """
    B, T, V = logits.shape
    # Flatten to (B*T, V) and (B*T,) for F.cross_entropy
    return nn.functional.cross_entropy(logits.view(B * T, V), targets.view(B * T))
