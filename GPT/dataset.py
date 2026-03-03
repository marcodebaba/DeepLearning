"""
Text dataset for GPT training.

Input/target pairs are created with a sliding window:
  Input  : tokens[i : i + context_length]
  Target : tokens[i+1 : i + context_length + 1]

This teaches the model to predict the *next* token at every position,
which is the core objective of a language model.

Example with context_length=4:
  Corpus: "hello"
  tokens:  [7, 3, 11, 11, 14]
  i=0 -> input=[7,3,11,11], target=[3,11,11,14]
  i=1 -> input=[3,11,11,14], target=[11,11,14, ...]
"""

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split

from tokenizer import CharTokenizer
from config import GPTConfig


class TextDataset(Dataset):
    def __init__(self, token_ids: list[int], context_length: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self) -> int:
        # Each sample needs context_length+1 tokens (input + one target)
        return len(self.data) - self.context_length

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        chunk = self.data[idx : idx + self.context_length + 1]
        x = chunk[:-1]   # input tokens
        y = chunk[1:]    # target tokens (shifted by 1)
        return x, y


def build_dataloaders(
    tokenizer: CharTokenizer,
    text: str,
    cfg: GPTConfig,
) -> tuple[DataLoader, DataLoader]:
    """
    Tokenize the full text, split into train/val, return DataLoaders.
    """
    token_ids = tokenizer.encode(text)
    dataset = TextDataset(token_ids, cfg.context_length)

    # Split into train / validation
    train_size = int(len(dataset) * cfg.train_split)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, drop_last=True)

    print(f"Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,}")
    return train_loader, val_loader
