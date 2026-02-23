#!/usr/bin/env python3
"""
MARJARA v3 – training entry point.

Single-GPU:
    python train.py --data data/train.bin --model_size small --mixed_precision

Multi-GPU (via torchrun):
    torchrun --nproc_per_node=4 train.py --data data/train.bin --distributed --mixed_precision

Changes over v2:
  - Added --wandb_project for W&B logging (new in trainer v3)
  - Added --early_stop_patience and --early_stop_min_delta (new in trainer v3)
  - Added --local_rank auto-detection from env var (torchrun sets LOCAL_RANK)
  - Flash/mem-efficient SDP backends explicitly enabled
  - Deterministic CUDA mode flag (--deterministic) for reproducibility debugging
  - torch.compile mode configurable (--compile_mode)
  - Removed version tag in description (now v3)
"""
import argparse
import os
import random

import torch

from marjara import Learner


def parse_args():
    p = argparse.ArgumentParser(description="MARJARA v3 – LLM training")

    # ── Data ────────────────────────────────────────────────────────────────
    p.add_argument("--data",           type=str, required=True,
                   help="Path to .txt or .bin tokenised file")
    p.add_argument("--val_data",       type=str, default=None,
                   help="Optional separate validation .bin file")
    p.add_argument("--tokenizer",      type=str, default="cl100k_base",
                   help="tiktoken encoding name")
    p.add_argument("--output_dir",     type=str, default="checkpoints")

    # ── Model ───────────────────────────────────────────────────────────────
    p.add_argument("--model_size",     type=str, default="small",
                   choices=["tiny", "small", "medium", "large"])
    p.add_argument("--context_length", type=int, default=1024)
    p.add_argument("--use_moe",        action="store_true",
                   help="Enable Mixture-of-Experts FFN")
    p.add_argument("--sliding_window", type=int, default=None)

    # ── Training ────────────────────────────────────────────────────────────
    p.add_argument("--epochs",              type=int,   default=10)
    p.add_argument("--batch_size",          type=int,   default=32)
    p.add_argument("--lr",                  type=float, default=3e-4)
    p.add_argument("--warmup_steps",        type=int,   default=100)
    p.add_argument("--accumulation_steps",  type=int,   default=1)
    p.add_argument("--grad_clip",           type=float, default=1.0)

    # ── Early stopping (new in v3) ───────────────────────────────────────────
    p.add_argument("--early_stop_patience",  type=int,   default=5,
                   help="Stop after N epochs with no val_ce improvement")
    p.add_argument("--early_stop_min_delta", type=float, default=0.0,
                   help="Minimum improvement in val_ce to count as progress")

    # ── Optimisation ────────────────────────────────────────────────────────
    p.add_argument("--mixed_precision",        action="store_true")
    p.add_argument("--compile",                action="store_true",
                   help="torch.compile the model")
    p.add_argument("--compile_mode",           type=str, default="default",
                   choices=["default", "reduce-overhead", "max-autotune"],
                   help="torch.compile mode (reduce-overhead good for fixed shapes)")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--use_ema",                action="store_true")
    p.add_argument("--ema_decay",              type=float, default=0.9999)

    # ── Distributed ─────────────────────────────────────────────────────────
    p.add_argument("--distributed", action="store_true")
    # torchrun sets LOCAL_RANK env var; fall back to arg for manual launches
    p.add_argument("--local_rank",  type=int,
                   default=int(os.environ.get("LOCAL_RANK", 0)),
                   help="Set automatically by torchrun via LOCAL_RANK env var")

    # ── Checkpointing ───────────────────────────────────────────────────────
    p.add_argument("--resume",     type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--save_every", type=int, default=5,
                   help="Save epoch checkpoint every N epochs")

    # ── Logging (new in v3) ─────────────────────────────────────────────────
    p.add_argument("--wandb_project", type=str, default=None,
                   help="Weights & Biases project name (omit to disable W&B)")

    # ── Reproducibility ─────────────────────────────────────────────────────
    p.add_argument("--seed",          type=int,  default=42)
    p.add_argument("--deterministic", action="store_true",
                   help="Enable fully deterministic CUDA ops (slower, for debugging)")

    return p.parse_args()


def setup_torch(args):
    """Seed everything and configure global torch backends."""
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        return

    torch.cuda.manual_seed_all(args.seed)

    # TF32 – free ~10% speedup on Ampere+ with negligible precision loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision("high")

    # Flash / memory-efficient attention (picked automatically by SDPA)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    if args.deterministic:
        # Forces deterministic algorithms; some ops will raise if no
        # deterministic kernel exists. Use only for debugging.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        torch.use_deterministic_algorithms(True)
        print("Warning: deterministic mode enabled – training will be slower.")
    else:
        # cuDNN auto-tuner: finds fastest conv algorithms for fixed input sizes
        torch.backends.cudnn.benchmark = True


def main():
    args = parse_args()
    setup_torch(args)
    trainer = Learner(args)
    trainer.train()


if __name__ == "__main__":
    main()
