# MARJARA v2

A transformer trained from scratch — production-ready LLM training stack with GQA, RoPE, MoE, KV cache, and distributed training.

## Project Structure

```
marjara/
├── train.py                 # Entry point
├── requirements.txt
├── marjara/
│   ├── __init__.py
│   ├── config.py            # HParams dataclass (model configuration)
│   ├── rope.py              # Rotary Positional Embeddings
│   ├── cache.py             # KV Cache with per-batch position tracking
│   ├── attention.py         # GQA Attention + RMSNorm + QK-norm
│   ├── ffn.py               # SwiGLU (dense) and MoE (sparse) FFN
│   ├── model.py             # Block + MarjaraModel (forward + generate)
│   ├── datasets.py          # TextDataset + MMapSliceDataset
│   └── trainer.py           # Learner: full training/eval/checkpoint loop
```

## Quick Start

```bash
pip install -r requirements.txt

# Single GPU
python train.py --data data/train.txt --model_size small --mixed_precision

# Multi-GPU (torchrun)
torchrun --nproc_per_node=4 train.py \
    --data data/train.bin \
    --model_size medium \
    --distributed \
    --mixed_precision \
    --compile
```

## Model Presets

| Size   | d_model | n_layers | n_heads |
|--------|---------|----------|---------|
| tiny   | 128     | 4        | 4       |
| small  | 256     | 8        | 8       |
| medium | 512     | 12       | 8       |
| large  | 768     | 24       | 12      |

## Key Features

- **GQA** – Grouped Query Attention for efficient inference
- **RoPE** – Rotary positional embeddings (LLaMA-style)
- **QK-norm** – Improved training stability at scale
- **SwiGLU / MoE** – Dense or sparse FFN (enable with `--use_moe`)
- **KV Cache** – Efficient autoregressive generation
- **Mixed precision** – bf16/fp16 via `torch.amp`
- **DDP** – Multi-GPU via `torchrun`
- **EMA** – Exponential Moving Average weights
- **Gradient checkpointing** – Memory saving for deep models

## Known Limitations / Future Work

- MoE expert dispatch is Python-loop based (CPU branch heavy) → future: scatter/gather CUDA kernels
- KV cache shape may fragment with variable-length batches
- lm-evaluation-harness integration is a stub (`evaluate_benchmarks` in trainer)

## Data Format

MARJARA accepts two formats:

- **Plain text** (`.txt`): tokenised with tiktoken on the fly, auto-split 90/10
- **Binary** (`.bin`): `uint16` token array (e.g. from `numpy.memmap`), for large corpora

## Resuming Training

```bash
python train.py --data data/train.bin --resume checkpoints/checkpoint_latest.pt
```
