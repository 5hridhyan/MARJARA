"""
Feed-Forward Network modules:
  - SwiGLU (dense layers)
  - MoELayer (Mixture of Experts with capacity factor and routing noise)
"""
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HParams


class SwiGLU(nn.Module):
    def __init__(self, cfg: HParams):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class MoELayer(nn.Module):
    """
    Mixture of Experts with:
      - Top-k expert routing
      - Load-balancing auxiliary loss (Switch Transformer style)
      - Optional z-loss
      - Capacity factor (token dropping)
      - Routing noise for exploration during training
    
    Known limitation: expert dispatch is still a Python-level loop
    (CPU branch heavy) — future work: replace with scatter/gather kernels.
    """

    def __init__(self, cfg: HParams):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_active_experts = cfg.n_active_experts
        self.aux_loss_coef = cfg.moe_aux_loss_coef
        self.z_loss_coef = cfg.moe_z_loss_coef
        self.capacity_factor = cfg.moe_capacity_factor
        self.noise_std = cfg.moe_noise_std

        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)

        # Stacked expert weights (w1/w3 = gate/up, w2 = down)
        self.w1 = nn.Parameter(torch.randn(cfg.n_experts, cfg.d_model, cfg.d_ff))
        self.w2 = nn.Parameter(torch.randn(cfg.n_experts, cfg.d_ff, cfg.d_model))
        self.w3 = nn.Parameter(torch.randn(cfg.n_experts, cfg.d_model, cfg.d_ff))

        std_out = cfg.init_std / math.sqrt(2 * cfg.n_layers) if cfg.scale_init_by_depth else cfg.init_std
        nn.init.normal_(self.w1, std=cfg.init_std)
        nn.init.normal_(self.w2, std=std_out)
        nn.init.normal_(self.w3, std=cfg.init_std)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        n_tokens = x_flat.shape[0]

        router_logits = self.router(x_flat)

        if self.training and self.noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_std

        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(routing_weights, self.n_active_experts, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        expert_indices = topk_indices.view(-1)
        token_indices = torch.arange(n_tokens, device=x.device).repeat_interleave(self.n_active_experts)
        expert_weights = topk_weights.view(-1)

        capacity = (
            math.ceil((n_tokens / self.n_experts) * self.capacity_factor)
            if self.capacity_factor is not None and self.training
            else None
        )

        sorted_expert_indices, sort_idx = torch.sort(expert_indices)
        sorted_token_indices = token_indices[sort_idx]
        sorted_weights = expert_weights[sort_idx]

        if capacity is not None:
            _, counts = torch.unique_consecutive(sorted_expert_indices, return_counts=True)
            keep_mask = torch.zeros_like(sorted_expert_indices, dtype=torch.bool)
            pos = 0
            for count in counts:
                keep_count = min(count.item(), capacity)
                keep_mask[pos : pos + keep_count] = True
                pos += count.item()
            sorted_expert_indices = sorted_expert_indices[keep_mask]
            sorted_token_indices = sorted_token_indices[keep_mask]
            sorted_weights = sorted_weights[keep_mask]

        outputs = torch.zeros(n_tokens, d_model, device=x.device, dtype=x.dtype)

        if len(sorted_expert_indices) > 0:
            unique_experts, counts = torch.unique_consecutive(sorted_expert_indices, return_counts=True)
            start = 0
            for expert_idx, count in zip(unique_experts, counts):
                end = start + count.item()
                idx = sorted_token_indices[start:end]
                w = sorted_weights[start:end, None]

                expert_tokens = x_flat[idx]
                gate_out = F.silu(torch.mm(expert_tokens, self.w1[expert_idx]))
                up_out = torch.mm(expert_tokens, self.w3[expert_idx])
                expert_out = torch.mm(gate_out * up_out, self.w2[expert_idx])
                outputs.index_add_(0, idx, w * expert_out)
                start = end

        # Load-balancing auxiliary loss
        probs = F.softmax(router_logits, dim=-1)
        importance = probs.mean(dim=0)
        top1_expert = topk_indices[:, 0]
        load = torch.zeros(self.n_experts, device=x.device, dtype=torch.float32)
        load.scatter_add_(0, top1_expert, torch.ones_like(top1_expert, dtype=torch.float32))
        load = load / n_tokens
        aux_loss = self.n_experts * (importance * load).sum()

        z_loss = (router_logits ** 2).mean() * self.z_loss_coef if self.z_loss_coef > 0 else 0.0

        losses = {"aux": aux_loss * self.aux_loss_coef, "z": z_loss}
        return outputs.view(batch, seq_len, -1), losses
