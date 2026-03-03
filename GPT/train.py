"""
GPT training script.

Usage:
    python train.py

Saves the best checkpoint to gpt_best.pth.
Prints train/val loss every eval_interval epochs.
"""

import os
import sys
import torch
import torch.optim as optim

# Make imports work from this directory
sys.path.insert(0, os.path.dirname(__file__))

from config import GPTConfig, SmallGPTConfig
from tokenizer import build_tokenizer
from dataset import build_dataloaders
from model import GPTModel, compute_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")   # Apple Silicon GPU
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: GPTModel, loader, device: torch.device) -> float:
    """Compute average loss over the entire loader."""
    model.eval()
    total_loss, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = compute_loss(logits, y)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: GPTConfig, text_path: str) -> GPTModel:
    device = get_device()
    print(f"Device: {device}")

    # --- Data ---
    tokenizer, text = build_tokenizer(text_path)

    # Update vocab_size to match actual corpus vocabulary
    cfg.vocab_size = tokenizer.vocab_size

    train_loader, val_loader = build_dataloaders(tokenizer, text, cfg)

    # --- Model ---
    model = GPTModel(cfg).to(device)
    print(f"Parameters: {model.num_params():,}")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.1)

    # Cosine annealing: gradually reduce LR to 0 over all epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)

    best_val_loss = float("inf")
    checkpoint_path = os.path.join(os.path.dirname(__file__), "gpt_best.pth")

    print("\n===== Training =====")
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        epoch_loss, steps = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = compute_loss(logits, y)
            loss.backward()

            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()
            steps += 1

        scheduler.step()

        # --- Periodic evaluation ---
        if epoch % cfg.eval_interval == 0 or epoch == 1:
            train_loss = epoch_loss / max(steps, 1)
            val_loss = evaluate(model, val_loader, device)

            marker = " *" if val_loss < best_val_loss else ""
            print(
                f"Epoch {epoch:>4} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}{marker}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "cfg": cfg,
                        "vocab": {
                            "char_to_idx": tokenizer.char_to_idx,
                            "idx_to_char": tokenizer.idx_to_char,
                        },
                    },
                    checkpoint_path,
                )

    print(f"\nBest val loss: {best_val_loss:.4f} — saved to {checkpoint_path}")
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Text file to train on — defaults to the-verdict.txt in parent directory
    script_dir = os.path.dirname(__file__)
    default_text = os.path.join(script_dir, "..", "the-verdict.txt")

    text_path = sys.argv[1] if len(sys.argv) > 1 else default_text

    if not os.path.exists(text_path):
        print(f"Text file not found: {text_path}")
        print("Usage: python train.py [path/to/text.txt]")
        sys.exit(1)

    # Use SmallGPTConfig for quick local runs; switch to GPTConfig for full training
    cfg = SmallGPTConfig
    train(cfg, text_path)
