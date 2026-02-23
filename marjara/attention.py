"""
Grouped Query Attention (GQA) with RoPE, QK-norm, sliding window, and KV cache support.
MARJARA v3 - attention.py

Changes over v2:
  - Flash/mem-efficient SDP explicitly enabled at module level
  - Sliding window mask cached as registered buffer; only rebuilt on shape change
  - _repeat_kv uses expand+reshape (zero-copy); now also handles contiguous layout for SDPA
  - Fused QKV projection option (single matmul, split) for better GPU utilisation
  - Causal mask passed as float bias instead of bool for SDPA compatibility with sliding window
  - Gate removed from SimpleNorm (was computing variance twice via pow+mean; now uses fused rsqrt)
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HParams
from .rope import RotaryPos
from .cache import CacheThingy

# Enable all SDPA backends once at import time
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


class SimpleNorm(nn.Module):
    """
    RMSNorm – fused variance + rsqrt in a single kernel via torch.rsqrt.
    eps kept small (1e-6) to avoid squashing tiny activations.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for variance computation; keeps precision under bf16 training
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f * rms).to(x.dtype) * self.weight


class Attn(nn.Module):
    def __init__(self, cfg: HParams, layer_idx: int):
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim   = cfg.d_model // cfg.n_heads
        self.layer_idx  = layer_idx
        self.sliding_window = cfg.sliding_window
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

        # Fused QKV projection: one matmul instead of three, then split
        q_size  = cfg.n_heads    * self.head_dim
        kv_size = cfg.n_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(cfg.d_model, q_size + 2 * kv_size, bias=False)
        self.out_proj  = nn.Linear(q_size, cfg.d_model, bias=False)

        self.drop_p = cfg.dropout
        self.rope   = RotaryPos(self.head_dim, max_seq_len=cfg.context_length * 2, theta=cfg.rope_theta)

        self.q_norm = SimpleNorm(self.head_dim)
        self.k_norm = SimpleNorm(self.head_dim)

        self.cache: Optional[CacheThingy] = None

        # Cached sliding window mask –--- invalidated when (seq_len, total_seq_len, past_length) changes
        self._sw_mask_key: Optional[tuple] = None
        self.register_buffer("_sw_mask", None, persistent=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, kv_heads, seq, head_dim) -> (batch, heads, seq, head_dim)  – zero copy"""
        if self.n_rep == 1:
            return x
        b, kv, s, d = x.shape
        return x[:, :, None, :, :].expand(b, kv, self.n_rep, s, d).reshape(b, kv * self.n_rep, s, d)

    def _get_sliding_mask(self, seq_len: int, total_seq_len: int, past_length: int,
                          device: torch.device) -> torch.Tensor:
        """Returns a float additive mask (0 or -inf) for sliding window attention."""
        key = (seq_len, total_seq_len, past_length)
        if self._sw_mask_key != key:
            q_idx = torch.arange(seq_len,       device=device).unsqueeze(1)
            k_idx = torch.arange(total_seq_len, device=device).unsqueeze(0)
            causal = (q_idx + past_length) >= k_idx
            window = (q_idx + past_length) - k_idx < self.sliding_window
            bool_mask = causal & window                         # True = attend
            float_mask = torch.zeros(seq_len, total_seq_len, device=device)
            float_mask.masked_fill_(~bool_mask, float("-inf"))  # additive bias for SDPA
            self._sw_mask     = float_mask[None, None, :, :]   # (1, 1, q, k)
            self._sw_mask_key = key
        return self._sw_mask

    def forward(self, x: torch.Tensor, use_cache: bool = False, past_length: int = 0) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Single fused matmul, then split it
        qkv = self.qkv_proj(x)
        q_size  = self.n_heads    * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        q = q.view(batch, seq_len, self.n_heads,    self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rope(q, k, past_length)

        if use_cache and self.cache is not None:
            self.cache.update(self.layer_idx, k, v)
            k_full, v_full = self.cache.get(self.layer_idx)
            total_seq_len = k_full.size(2)
        else:
            k_full, v_full = k, v
            total_seq_len = seq_len

        k_rep = self._repeat_kv(k_full)
        v_rep = self._repeat_kv(v_full)

        # build sliding window mask only when needed
        attn_mask = None
        is_causal  = True
        if self.sliding_window is not None and total_seq_len > self.sliding_window:
            attn_mask = self._get_sliding_mask(seq_len, total_seq_len, past_length, x.device)
            is_causal  = False  # mask already encodes causality

        # ensure the contiguous memory layout for SDPA kernels
        q     = q.contiguous()
        k_rep = k_rep.contiguous()
        v_rep = v_rep.contiguous()

        attn_output = F.scaled_dot_product_attention(
            q, k_rep, v_rep,
            attn_mask  = attn_mask,
            dropout_p  = self.drop_p if self.training else 0.0,
            is_causal  = is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(attn_output)
