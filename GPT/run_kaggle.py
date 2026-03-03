"""
Kaggle entry point for GPT training.

Steps:
  1. Download TinyShakespeare corpus (~1 MB) from GitHub
  2. Train a medium-scale GPT-2-like model on Kaggle GPU
  3. Save the best checkpoint to /kaggle/working/gpt_best.pth

Hardware target: Kaggle T4 (16 GB VRAM)
Estimated training time: ~20 min for 50 epochs
"""

import os
import sys
import urllib.request

# ── Ensure sibling modules are importable ──────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── Download training corpus ───────────────────────────────────────────────
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
DATA_PATH = "/kaggle/working/tinyshakespeare.txt"

print("Downloading TinyShakespeare corpus...")
urllib.request.urlretrieve(DATA_URL, DATA_PATH)
print(f"Saved {os.path.getsize(DATA_PATH):,} bytes → {DATA_PATH}\n")

# ── Training config ────────────────────────────────────────────────────────
from config import GPTConfig
from train import train

# Medium config — tuned for Kaggle T4 (16 GB VRAM)
#   Parameters ≈ 25 M
#   Memory usage ≈ 4–6 GB with batch_size=64
cfg = GPTConfig(
    vocab_size=256,         # updated from actual corpus vocab at runtime
    context_length=256,     # tokens the model sees at once
    d_model=512,            # embedding dimension
    n_heads=8,              # attention heads  (512 / 8 = 64 head_dim)
    n_layers=8,             # Transformer blocks
    dropout=0.1,
    batch_size=64,
    learning_rate=3e-4,
    max_epochs=50,
    eval_interval=5,
    train_split=0.9,
)

# ── Train ──────────────────────────────────────────────────────────────────
train(cfg, DATA_PATH)
