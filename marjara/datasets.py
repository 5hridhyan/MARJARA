"""
Dataset classes for MARJARA v2.

  - TextDataset: in-memory dataset for plain .txt files
  - MMapSliceDataset: memory-mapped uint16 binary dataset with train/val views
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """In-memory dataset for .txt files (small corpora)."""

    def __init__(self, data: torch.Tensor, context_length: int):
        self.data = data
        self.context_length = context_length

    def __len__(self):
        return max(0, len(self.data) - self.context_length)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.context_length]
        y = self.data[idx + 1 : idx + self.context_length + 1]
        return x, y


class MMapSliceDataset(Dataset):
    """
    Slice of a memory-mapped uint16 array.
    Suitable for TB-scale tokenised corpora.

    Args:
        bin_path: path to a .bin file tokenised as uint16
        start: first token index (inclusive)
        end: last token index (exclusive)
        context_length: number of tokens per sample
    """

    def __init__(self, bin_path: str, start: int, end: int, context_length: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.start = start
        self.end = end
        self.context_length = context_length
        self.length = max(0, (end - start) - context_length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        global_idx = self.start + idx
        chunk = self.data[global_idx : global_idx + self.context_length + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y
