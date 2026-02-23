"""
Feed-Forward Network modules: SwiGLU and MoELayer.
MARJARA v3 - ffn.py

Changes over v2:
  - Expert dispatch vectorized with scatter/gather (no Python loop over unique experts)
  - z-loss computed on clean logits BEFORE noise (DeepSeek had this correct; kept)
  - Expert dropout applied as weight zeroing AFTER renorm (not before), preventing NaN
  - set_noise_std() kept; noise_std now stored as plain float (not buffer) to avoid
    spurious state_dict entries
  - Capacity masking uses cumsum trick instead of Python loop (no CPU sync)
  - aux_loss uses detached load to prevent routing collapse from gradient feedback
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HParams


class SwiGLU(nn.Module):
    def __init__(self, cfg: HParams):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up   = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class MoELayer(nn.Module):
    """
    Mixture of Experts with:
      - Top-k routing with renormalized weights
      - Vectorized expert dispatch (scatter/gather, no Python loop)
      - Load-balancing aux loss with detached load estimate
      - Z-loss on clean logits
      - Capacity factor with cumsum-based token dropping (no CPU sync)
      - Routing noise with external decay via set_noise_std()
      - Expert dropout
    """

    def __init__(self, cfg: HParams):
        super().__init__()
        self.n_experts        = cfg.n_experts
        self.n_active         = cfg.n_active_experts
        self.aux_loss_coef    = cfg.moe_aux_loss_coef
        self.z_loss_coef      = cfg.moe_z_loss_coef
        self.capacity_factor  = cfg.moe_capacity_factor
        self.expert_dropout   = cfg.dropout
        self.noise_std        = cfg.moe_noise_std   # plain float, decayed externally

        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)

        # Stacked expert weights – shape: (n_experts, d_model, d_ff) etc.
        self.w1 = nn.Parameter(torch.empty(cfg.n_experts, cfg.d_model, cfg.d_ff))
        self.w2 = nn.Parameter(torch.empty(cfg.n_experts, cfg.d_ff,    cfg.d_model))
        self.w3 = nn.Parameter(torch.empty(cfg.n_experts, cfg.d_model, cfg.d_ff))

        std_in  = cfg.init_std
        std_out = cfg.init_std / math.sqrt(2 * cfg.n_layers) if cfg.scale_init_by_depth else cfg.init_std
        nn.init.normal_(self.w1, std=std_in)
        nn.init.normal_(self.w2, std=std_out)
        nn.init.normal_(self.w3, std=std_in)

    def set_noise_std(self, value: float):
        self.noise_std = value

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, S, D = x.shape
        x_flat   = x.view(-1, D)          # (T, D)  T = B*S
        T        = x_flat.shape[0]

        # Router -------------------------------------------------------
        logits = self.router(x_flat)       # (T, E)

        # Z-loss on CLEAN logits (before noise)
        z_loss = (logits.float() ** 2).mean() * self.z_loss_coef if self.z_loss_coef > 0 else 0.0

        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        probs       = F.softmax(logits, dim=-1, dtype=torch.float32)    # (T, E)
        topk_w, topk_idx = torch.topk(probs, self.n_active, dim=-1)    # (T, k)
        topk_w      = topk_w / topk_w.sum(dim=-1, keepdim=True)        # renorm

        # Expert dropout: zero some weights, renorm survivors
        if self.training and self.expert_dropout > 0:
            keep = torch.bernoulli(torch.full_like(topk_w, 1.0 - self.expert_dropout))
            topk_w = topk_w * keep
            denom  = topk_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            topk_w = topk_w / denom

        # Capacity masking (vectorized, no CPU sync)
        # token_expert: (T*k,) flat expert assignment
        token_expert = topk_idx.reshape(-1)                            # (T*k,)
        token_ids    = torch.arange(T, device=x.device).repeat_interleave(self.n_active)
        flat_weights = topk_w.reshape(-1)

        if self.capacity_factor is not None and self.training:
            capacity = math.ceil((T / self.n_experts) * self.capacity_factor)
            # For each (expert, position) pair compute arrival order
            # One-hot encode expert assignment, cumsum gives per-expert position
            one_hot = F.one_hot(token_expert, self.n_experts).float()  # (T*k, E)
            arrival = (one_hot.cumsum(0) * one_hot).sum(-1) - 1        # 0-indexed arrival
            keep_mask = arrival < capacity
            token_expert = token_expert[keep_mask]
            token_ids    = token_ids[keep_mask]
            flat_weights = flat_weights[keep_mask]

        # Vectorized expert dispatch
        # Group tokens by expert assignment using one-hot + matmul.
        # For each expert e: gather tokens assigned to e, run FFN, scatter-add back.
        # avoiding a Python loop by using batched matmul over all experts at once.
        #
        # Build dispatch matrix D: (n_experts, T)
        # D[e, t] = weight of token t assigned to expert e (0 if not assigned)

        n_assigned = token_expert.shape[0]
        if n_assigned == 0:
            outputs = torch.zeros(T, D, device=x.device, dtype=x.dtype)
        else:
            # dispatch: (n_experts, T)  – weighted assignment matrix
            dispatch = torch.zeros(self.n_experts, T, device=x.device, dtype=x.float().dtype)
            dispatch.index_put_(
                (token_expert, token_ids),
                flat_weights.float(),
                accumulate=True,
            )
            # combine: (n_experts, T, D)  = dispatch[:, :, None] * x_flat[None]
            # expert_in[e] = weighted sum of tokens assigned to expert e
            # Shape: (E, T, D)
            expert_in  = torch.einsum("et,td->etd", dispatch, x_flat.float())

            # Expert FFN (batched over E):  w1,w3: (E,D,F)  w2: (E,F,D)
            gate_out   = F.silu(torch.bmm(expert_in, self.w1))         # (E, T, F)
            up_out     = torch.bmm(expert_in, self.w3)                  # (E, T, F)
            expert_out = torch.bmm(gate_out * up_out, self.w2)          # (E, T, D)

            # combine back: sum over experts
            outputs = expert_out.sum(0).to(x.dtype)                     # (T, D)

        # Load-balancing aux loss 
        # Use detached load so routing doesn't collapse from its own gradient
        importance = probs.mean(0)                                       # (E,)
        load = torch.zeros(self.n_experts, device=x.device, dtype=torch.float32)
        load.scatter_add_(0, topk_idx[:, 0],
                          torch.ones(T, device=x.device, dtype=torch.float32))
        load = (load / T).detach()
        aux_loss = self.n_experts * (importance * load).sum()

        losses = {
            "aux": aux_loss * self.aux_loss_coef,
            "z":   z_loss,
        }
        return outputs.view(B, S, D), losses
