"""GPT baseline. Decoder-only transformer: RoPE, grouped-query attention, pre-norm RMSNorm, GELU MLP."""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 1024
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 8        # == n_heads -> MHA; < n_heads -> GQA
    max_seq_len: int = 256
    rope_base: float = 10000.0


def rope_cache(seq_len, d_head, base=10000.0, device=None):
    """Returns (cos, sin) of shape (seq_len, d_head // 2) for angles m * base^(-2i / d_head)."""
    theta = base ** (-torch.arange(0, d_head, 2) / d_head)
    position = torch.arange(0, seq_len)
    angles = torch.outer(position, theta)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """Rotates interleaved pairs (x[..., 2i], x[..., 2i+1]) of x: (B, H, T, d_head) by position."""
    B, H, T, d_head = x.shape
    # e^{i m theta} as a complex tensor, broadcast over batch and heads
    freqs = torch.view_as_complex(torch.stack((cos, sin), dim=-1))
    freqs = freqs.reshape(1, 1, freqs.shape[0], freqs.shape[-1])[:, :, :T, :]
    x = torch.view_as_complex(x.reshape(B, H, T, d_head // 2, 2))
    return torch.view_as_real(x * freqs).reshape(B, H, T, d_head)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, -1) + self.eps)
        return torch.divide(x, rms[:, :, None]) * self.weight


class Attention(nn.Module):
    """Causal self-attention with RoPE on q and k. Query head h reads kv head h // (n_heads // n_kv_heads)."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.heads = cfg.n_heads
        self.kv_heads = cfg.n_kv_heads
        self.d_model = cfg.d_model
        self.d_head = self.d_model // self.heads
        kv_dim = self.d_head * self.kv_heads
        self.wq = nn.Linear(self.d_model, self.d_model, bias=False)
        self.wk = nn.Linear(self.d_model, kv_dim, bias=False)
        self.wv = nn.Linear(self.d_model, kv_dim, bias=False)
        self.wo = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, x, cos, sin):
        # x: (B, T, d_model)
        Q, K, V = self.wq(x), self.wk(x), self.wv(x)
        q = Q.reshape(*Q.shape[:-1], self.heads, self.d_head).transpose(1, 2)      # (B, H, T, d_head)
        k = K.reshape(*K.shape[:-1], self.kv_heads, self.d_head).transpose(1, 2)   # (B, H_kv, T, d_head)
        v = V.reshape(*V.shape[:-1], self.kv_heads, self.d_head).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA: share each kv head across its group of query heads
        k = k.repeat_interleave(self.heads // self.kv_heads, 1)
        v = v.repeat_interleave(self.heads // self.kv_heads, 1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)  # (B, T, H, d_head)
        out = out.reshape(*out.shape[:-2], self.heads * self.d_head)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.activation1 = nn.GELU()
        self.fc2 = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        return self.fc2(self.activation1(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    """
    Untied embedding and LM head. GPT-2 init: N(0, 0.02) for all matmul and embedding
    weights, N(0, 0.02 / sqrt(2 * n_layers)) for the residual projections (attn.wo, mlp.fc2).
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(num_embeddings=cfg.vocab_size, embedding_dim=cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        cos, sin = rope_cache(cfg.max_seq_len, cfg.d_model // cfg.n_heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.mlp.fc2.weight, mean=0, std=std)
            nn.init.normal_(block.attn.wo.weight, mean=0, std=std)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, idx):
        """idx: (B, T) int64 -> logits: (B, T, vocab_size)"""
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        return self.lm_head(self.norm_f(x))


def param_count(cfg: GPTConfig):
    """Exact parameter count from the config: L * (10d^2 + 2 d d_head n_kv + 2d) + 2Vd + d."""
    d, L, V = cfg.d_model, cfg.n_layers, cfg.vocab_size
    d_head = d // cfg.n_heads
    return L * (10 * d ** 2 + 2 * d * d_head * cfg.n_kv_heads + 2 * d) + 2 * V * d + d
