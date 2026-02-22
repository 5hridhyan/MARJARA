"""
KV Cache – unified, per-batch position tracking.

Assumes all batch items have identical sequence lengths (true for
training & greedy generation).
"""
import torch


class CacheThingy:
    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        shape = (n_layers, batch_size, n_kv_heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.seen_tokens = torch.zeros(batch_size, dtype=torch.long, device=device)

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Store new k/v for a layer at the current positions."""
        seq_len = k.shape[-2]
        start = self.seen_tokens[0].item()
        end = start + seq_len
        self.k_cache[layer_idx, :, :, start:end, :] = k.transpose(1, 2)
        self.v_cache[layer_idx, :, :, start:end, :] = v.transpose(1, 2)

    def get(self, layer_idx: int):
        """Return all cached k/v for a layer up to the maximum seen tokens."""
        max_seen = self.seen_tokens.max().item()
        return (
            self.k_cache[layer_idx, :, :, :max_seen, :],
            self.v_cache[layer_idx, :, :, :max_seen, :],
        )

    def increment(self, seq_len: int):
        """Advance seen_tokens for all batch items by seq_len."""
        self.seen_tokens += seq_len

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seen_tokens.zero_()
