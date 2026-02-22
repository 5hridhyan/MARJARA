"""
MARJARA v2 – HParams configuration dataclass
"""
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class HParams:
    vocab_size: int = 50304
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 2
    n_layers: int = 6
    d_ff: Optional[int] = None
    dropout: float = 0.1
    context_length: int = 1024
    rope_theta: float = 10000.0
    sliding_window: Optional[int] = None

    # MoE
    use_moe: bool = False
    n_experts: int = 8
    n_active_experts: int = 2
    moe_aux_loss_coef: float = 0.01
    moe_z_loss_coef: float = 0.0001
    moe_capacity_factor: Optional[float] = None
    moe_noise_std: float = 0.0

    # Regularization & init
    weight_tying: bool = True
    init_std: float = 0.02
    scale_init_by_depth: bool = True

    # Efficiency
    gradient_checkpointing: bool = False

    # Generation
    max_generation_length: int = 512

    @classmethod
    def tiny(cls):
        return cls(d_model=128, n_heads=4, n_kv_heads=1, n_layers=4)

    @classmethod
    def small(cls):
        return cls(d_model=256, n_heads=8, n_kv_heads=2, n_layers=8)

    @classmethod
    def medium(cls):
        return cls(d_model=512, n_heads=8, n_kv_heads=2, n_layers=12)

    @classmethod
    def large(cls):
        return cls(d_model=768, n_heads=12, n_kv_heads=4, n_layers=24)

    def __post_init__(self):
        if self.d_ff is None:
            # LLaMA-style: (8/3)*d_model rounded to multiple of 256
            self.d_ff = int(((8 / 3 * self.d_model) + 255) // 256 * 256)
        assert self.d_model % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
