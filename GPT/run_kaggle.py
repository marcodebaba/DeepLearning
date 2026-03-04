"""
Kaggle entry point for GPT training.

Module files (config.py, model.py, etc.) are uploaded as a Kaggle Dataset
(marcopan111/gpt-modules) and mounted at /kaggle/input/gpt-modules/.

Steps:
  1. Add dataset path to sys.path so imports work
  2. Download TinyShakespeare corpus
  3. Train GPT-2-like model on Kaggle GPU
  4. Save checkpoint to /kaggle/working/gpt_best.pth
"""

import glob
import os
import sys
import urllib.request

# Find config.py anywhere under /kaggle/input/ to handle different dataset structures
_found = glob.glob("/kaggle/input/**/config.py", recursive=True)
if _found:
    sys.path.insert(0, os.path.dirname(_found[0]))
else:
    sys.path.insert(0, "/kaggle/input/gpt-modules/")

from config import GPTConfig  # noqa: E402
from train import train  # noqa: E402

# ---------------------------------------------------------------------------
# Download training corpus
# ---------------------------------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
DATA_PATH = "/kaggle/working/tinyshakespeare.txt"

print("Downloading TinyShakespeare corpus...")
urllib.request.urlretrieve(DATA_URL, DATA_PATH)
print(f"Saved {os.path.getsize(DATA_PATH):,} bytes\n")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

cfg = GPTConfig(
    vocab_size=256,        # updated from actual corpus at runtime
    context_length=256,
    d_model=512,
    n_heads=8,
    n_layers=8,
    dropout=0.1,
    batch_size=64,
    learning_rate=3e-4,
    max_epochs=20,
    eval_interval=2,
    train_split=0.9,
)

train(cfg, DATA_PATH, checkpoint_path="/kaggle/working/gpt_best.pth")
