"""
MARJARA v2 – Learner (Trainer)

Handles:
  - Distributed training (DDP via torchrun)
  - Mixed precision (bf16/fp16 via torch.amp)
  - Gradient accumulation & clipping
  - Cosine LR schedule with linear warmup
  - EMA weights
  - Checkpoint save / resume
  - lm-evaluation-harness stub
"""
import logging
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import tiktoken

from .config import HParams
from .model import MarjaraModel
from .datasets import TextDataset, MMapSliceDataset


class Learner:
    def __init__(self, args):
        self.args = args
        self._setup_logging()
        self._setup_distributed()
        self._setup_device()
        self._setup_tokenizer()
        self._load_data()
        self._build_model()
        self._setup_ema()
        self._maybe_compile()
        self._maybe_ddp()
        self._setup_optimizer_scheduler()
        self._setup_mixed_precision()
        self._setup_checkpointing()

        self.current_step = 0
        self.best_val_ce = float("inf")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _setup_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO if self.args.local_rank == 0 else logging.WARNING,
        )
        self.logger = logging.getLogger(__name__)

    def _setup_distributed(self):
        self.is_distributed = self.args.distributed
        self.local_rank = self.args.local_rank
        self.world_size = 1
        self.rank = 0
        if self.is_distributed:
            if not torch.cuda.is_available():
                raise RuntimeError("Distributed training requires CUDA")
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            torch.cuda.set_device(self.local_rank)

    def _setup_device(self):
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")
            if self.is_distributed:
                self.logger.warning("Distributed training on CPU unsupported; disabling.")
                self.is_distributed = False

    def _setup_tokenizer(self):
        try:
            self.tokenizer = tiktoken.get_encoding(self.args.tokenizer)
        except Exception:
            self.logger.warning(f"Tokenizer '{self.args.tokenizer}' not found, falling back to cl100k_base")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _load_data(self):
        if self.args.data.endswith(".bin"):
            self.logger.info(f"Loading memory-mapped dataset from {self.args.data}")
            data_mmap = np.memmap(self.args.data, dtype=np.uint16, mode="r")
            total_len = len(data_mmap)
            self.args.vocab_size = max(50304, int(data_mmap.max()) + 1)
            split = int(0.9 * total_len)
            if self.args.val_data is not None:
                val_mmap = np.memmap(self.args.val_data, dtype=np.uint16, mode="r")
                val_len = len(val_mmap)
                self.train_dataset = MMapSliceDataset(self.args.data, 0, total_len, self.args.context_length)
                self.val_dataset = MMapSliceDataset(self.args.val_data, 0, val_len, self.args.context_length)
            else:
                self.train_dataset = MMapSliceDataset(self.args.data, 0, split, self.args.context_length)
                self.val_dataset = MMapSliceDataset(self.args.data, split, total_len, self.args.context_length)
        else:
            self.logger.info(f"Loading text file from {self.args.data}")
            with open(self.args.data, "r", encoding="utf-8") as f:
                text = f.read()
            tokens = self.tokenizer.encode(text)
            self.args.vocab_size = self.tokenizer.n_vocab
            data = torch.tensor(tokens, dtype=torch.long)
            split = int(0.9 * len(data))
            self.train_dataset = TextDataset(data[:split], self.args.context_length)
            self.val_dataset = TextDataset(data[split:], self.args.context_length)

        if self.is_distributed:
            self.train_sampler = DistributedSampler(
                self.train_dataset, num_replicas=self.world_size, rank=self.rank,
                shuffle=True, drop_last=True,
            )
            shuffle = False
        else:
            self.train_sampler = None
            shuffle = True

        num_workers = min(4, os.cpu_count() or 1)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.args.batch_size,
            shuffle=shuffle, sampler=self.train_sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.args.batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True,
        ) if self.val_dataset else None

        self.logger.info(f"Train samples: {len(self.train_dataset)}")
        if self.val_dataset:
            self.logger.info(f"Val samples: {len(self.val_dataset)}")

    def _build_model(self):
        preset_map = {"tiny": HParams.tiny, "small": HParams.small,
                      "medium": HParams.medium, "large": HParams.large}
        cfg = preset_map[self.args.model_size]() if self.args.model_size in preset_map else HParams()
        cfg.vocab_size = self.args.vocab_size
        cfg.context_length = self.args.context_length
        cfg.use_moe = self.args.use_moe
        cfg.sliding_window = self.args.sliding_window
        cfg.gradient_checkpointing = self.args.gradient_checkpointing
        self.cfg = cfg
        self.model = MarjaraModel(cfg).to(self.device)

    def _setup_ema(self):
        self.ema_model = None
        if self.args.use_ema:
            self.ema_model = MarjaraModel(self.cfg).to(self.device)
            self.ema_model.load_state_dict(self.model.state_dict())
            for p in self.ema_model.parameters():
                p.requires_grad = False
            self.ema_decay = self.args.ema_decay
            self.logger.info(f"EMA enabled with decay {self.ema_decay}")

    def _maybe_compile(self):
        if self.args.compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
            self.logger.info("Model compiled with torch.compile")

    def _maybe_ddp(self):
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])

    def _setup_optimizer_scheduler(self):
        decay_params, no_decay_params, seen_ids = [], [], set()
        for name, param in self.model.named_parameters():
            if id(param) in seen_ids:
                continue
            seen_ids.add(id(param))
            if param.ndim < 2 or any(x in name.lower() for x in ["norm", "embed"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        try:
            self.optimizer = AdamW(param_groups, lr=self.args.lr, betas=(0.9, 0.95), fused=True)
            self.logger.info("Using fused AdamW")
        except TypeError:
            self.optimizer = AdamW(param_groups, lr=self.args.lr, betas=(0.9, 0.95))
            self.logger.info("Using regular AdamW (fused not available)")

        steps_per_epoch = len(self.train_loader)
        eff_steps = math.ceil(steps_per_epoch / self.args.accumulation_steps)
        total_steps = self.args.epochs * eff_steps

        warmup = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.args.warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=max(1, total_steps - self.args.warmup_steps))
        self.scheduler = SequentialLR(self.optimizer, [warmup, cosine], milestones=[self.args.warmup_steps])

    def _setup_mixed_precision(self):
        self.use_amp = self.args.mixed_precision and self.device.type == "cuda"
        if self.use_amp:
            if torch.cuda.is_bf16_supported():
                self.precision_dtype = torch.bfloat16
                self.logger.info("Using bfloat16 mixed precision")
            else:
                self.precision_dtype = torch.float16
                self.logger.info("Using float16 mixed precision")
        else:
            self.precision_dtype = torch.float32
        try:
            from torch.amp import GradScaler
            self.scaler = GradScaler("cuda") if self.use_amp else None
        except ImportError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _setup_checkpointing(self):
        self.checkpoint_dir = Path(self.args.output_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------
    def _update_ema(self):
        if self.ema_model is None:
            return
        with torch.no_grad():
            model_state = self.model.state_dict()
            ema_state = self.ema_model.state_dict()
            for name, param in model_state.items():
                ema_name = name.replace("module.", "")
                if ema_name in ema_state:
                    ema_state[ema_name].mul_(self.ema_decay).add_(param, alpha=1 - self.ema_decay)

    def _compute_loss(self, logits, targets, aux_losses=None) -> Tuple[torch.Tensor, torch.Tensor]:
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss = ce_loss
        if aux_losses:
            for v in aux_losses.values():
                total_loss = total_loss + v
        return total_loss, ce_loss

    def _optimizer_step(self):
        if self.use_amp and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        if self.rank == 0 and self.current_step % 100 == 0:
            self.logger.info(f"Step {self.current_step} grad norm: {grad_norm:.4f}")
        if self.use_amp and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()
        self._update_ema()

    def train_epoch(self, epoch: int):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        self.model.train()
        total_loss = total_ce = 0.0
        total_aux = {"aux": 0.0, "z": 0.0} if self.cfg.use_moe else {}
        num_batches = 0
        self.optimizer.zero_grad()

        log_every = max(1, len(self.train_loader) // 100)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", disable=self.rank != 0)

        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.precision_dtype, enabled=self.use_amp):
                out = self.model(x)
                logits, aux_losses = out if isinstance(out, tuple) else (out, None)
                loss, ce_loss = self._compute_loss(logits, y, aux_losses)

            scaled_loss = loss / self.args.accumulation_steps
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (batch_idx + 1) % self.args.accumulation_steps == 0:
                self._optimizer_step()
                self.scheduler.step()
                self.current_step += 1

            total_loss += loss.item()
            total_ce += ce_loss.item()
            if aux_losses:
                for k in total_aux:
                    total_aux[k] += aux_losses.get(k, 0.0)
            num_batches += 1

            if self.rank == 0 and (batch_idx + 1) % log_every == 0:
                postfix = {"loss": f"{loss.item():.4f}", "ce": f"{ce_loss.item():.4f}"}
                if aux_losses:
                    postfix.update({k: f"{v.item():.4f}" for k, v in aux_losses.items()})
                pbar.set_postfix(postfix)

        # Handle leftover accumulation steps
        if len(self.train_loader) % self.args.accumulation_steps != 0:
            self._optimizer_step()
            self.scheduler.step()
            self.current_step += 1

        avg_loss = total_loss / num_batches
        avg_ce = total_ce / num_batches
        avg_aux = {k: v / num_batches for k, v in total_aux.items()} if self.cfg.use_moe else {}
        return avg_loss, avg_ce, avg_aux

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        if self.val_loader is None:
            return 0.0, 0.0
        self.model.eval()
        total_loss = total_ce = 0.0
        num_batches = 0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.precision_dtype, enabled=self.use_amp):
                out = self.model(x)
                logits, aux_losses = out if isinstance(out, tuple) else (out, None)
                loss, ce_loss = self._compute_loss(logits, y, aux_losses)
            total_loss += loss.item()
            total_ce += ce_loss.item()
            num_batches += 1

        avg_ce = total_ce / num_batches
        perplexity = math.exp(min(avg_ce, 20.0))
        if self.rank == 0:
            self.logger.info(f"Validation | CE: {avg_ce:.4f} | PPL: {perplexity:.2f}")
        return total_loss / num_batches, avg_ce

    # ------------------------------------------------------------------
    # Extras
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_benchmarks(self):
        """Stub for lm-evaluation-harness. Implement wrapper to use."""
        try:
            import lm_eval  # noqa: F401
            self.logger.info("lm-eval integration stub – implement wrapper to use.")
        except ImportError:
            self.logger.warning("lm-eval not installed, skipping benchmark evals")

    @torch.no_grad()
    def generate_sample(self, prompt: str, max_tokens: int = 100) -> str:
        model = self.ema_model if self.ema_model is not None else self.model
        if self.is_distributed:
            model = model.module
        model.eval()
        tokens = self.tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([tokens], device=self.device)
        generated = model.generate(
            prompt_tensor, max_new_tokens=max_tokens,
            temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.1,
        )
        return self.tokenizer.decode(generated[0].tolist())

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch: int, val_loss: float, val_ce: float, is_best: bool = False):
        if self.rank != 0:
            return
        model_state = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "val_loss": val_loss,
            "val_ce": val_ce,
            "config": asdict(self.cfg),
            "tokenizer_name": self.args.tokenizer,
            "args": vars(self.args),
        }
        if self.ema_model is not None:
            ckpt["ema_state_dict"] = self.ema_model.state_dict()
        torch.save(ckpt, self.checkpoint_dir / "checkpoint_latest.pt")
        if is_best:
            torch.save(ckpt, self.checkpoint_dir / "checkpoint_best.pt")
        if epoch % self.args.save_every == 0:
            torch.save(ckpt, self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
        self.logger.info(f"Saved checkpoint at epoch {epoch}")

    def load_checkpoint(self, path: Union[str, Path]):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        target = self.model.module if self.is_distributed else self.model
        target.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict") and self.scaler:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if self.ema_model and "ema_state_dict" in ckpt:
            self.ema_model.load_state_dict(ckpt["ema_state_dict"])
        return ckpt["epoch"], ckpt.get("val_ce", float("inf"))

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self):
        start_epoch = 0
        if self.args.resume:
            start_epoch, _ = self.load_checkpoint(self.args.resume)
            self.logger.info(f"Resumed from epoch {start_epoch}")

        for epoch in range(start_epoch, self.args.epochs):
            train_loss, train_ce, aux = self.train_epoch(epoch)
            val_loss, val_ce = self.validate()
            is_best = val_ce < self.best_val_ce
            if is_best:
                self.best_val_ce = val_ce

            lr = self.scheduler.get_last_lr()[0]
            msg = (f"Epoch {epoch+1}/{self.args.epochs} | "
                   f"Train loss: {train_loss:.4f} | Train CE: {train_ce:.4f} | "
                   f"Val CE: {val_ce:.4f} | LR: {lr:.2e}{' | BEST' if is_best else ''}")
            if aux:
                msg += f" | Aux: { {k: f'{v:.4f}' for k, v in aux.items()} }"
            self.logger.info(msg)

            if (epoch + 1) % 5 == 0 and self.rank == 0:
                self.logger.info(f"Sample:\n{self.generate_sample('The future of AI is')}")

            self.save_checkpoint(epoch + 1, val_loss, val_ce, is_best)

        if self.is_distributed:
            dist.destroy_process_group()
