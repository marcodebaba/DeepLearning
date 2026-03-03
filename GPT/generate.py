"""
Text generation with a trained GPT model.

Usage:
    python generate.py                          # use default prompt
    python generate.py "It was a dark night"    # use custom prompt

Sampling strategies:
  - Temperature scaling : divide logits by T before softmax
      T < 1  → more confident / repetitive
      T = 1  → standard sampling
      T > 1  → more creative / random
  - Top-k filtering     : zero out all but the k highest-probability tokens
"""

import os
import sys
import torch
from torch import Tensor

sys.path.insert(0, os.path.dirname(__file__))

from config import GPTConfig
from tokenizer import CharTokenizer
from model import GPTModel


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """
    Keep only the top-k logits; set the rest to -inf so softmax gives them 0 prob.

    logits: (vocab_size,)  — raw scores for one position
    """
    if k <= 0:
        return logits
    # Find the k-th largest value
    threshold = torch.topk(logits, k).values[-1]
    return logits.masked_fill(logits < threshold, float("-inf"))


@torch.no_grad()
def generate(
    model: GPTModel,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 1.0,
    top_k: int = 50,
    device: torch.device | None = None,
) -> str:
    """
    Autoregressively generate text given a prompt.

    At each step:
      1. Feed the last `context_length` tokens into the model
      2. Get logits for the *last* position
      3. Apply temperature + top-k filtering
      4. Sample the next token from the resulting distribution
      5. Append it and repeat
    """
    model.eval()
    device = device or next(model.parameters()).device

    # Encode prompt to token IDs
    context_ids = tokenizer.encode(prompt)
    context = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    context_length = model.cfg.context_length
    generated_ids: list[int] = []

    for _ in range(max_new_tokens):
        # Crop context to the last `context_length` tokens
        ctx = context[:, -context_length:]

        # Forward pass — only need logits at the last position
        logits = model(ctx)          # (1, T, vocab_size)
        next_logits = logits[0, -1]  # (vocab_size,)

        # Temperature scaling
        next_logits = next_logits / temperature

        # Top-k filtering
        next_logits = top_k_filter(next_logits, top_k)

        # Convert to probabilities and sample
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1,)

        # Append to context
        context = torch.cat([context, next_token.unsqueeze(0)], dim=1)  # (1, T+1)
        generated_ids.append(next_token.item())

    return prompt + tokenizer.decode(generated_ids)


# ---------------------------------------------------------------------------
# Load checkpoint and generate
# ---------------------------------------------------------------------------

def load_and_generate(
    checkpoint_path: str,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 1.0,
    top_k: int = 50,
) -> None:
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: GPTConfig = checkpoint["cfg"]
    vocab = checkpoint["vocab"]

    # Rebuild tokenizer from saved vocabulary
    tokenizer = CharTokenizer.__new__(CharTokenizer)
    tokenizer.char_to_idx = vocab["char_to_idx"]
    tokenizer.idx_to_char = vocab["idx_to_char"]
    tokenizer.vocab_size = len(tokenizer.char_to_idx)

    # Rebuild model
    model = GPTModel(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (device: {device})")
    print(f"Parameters: {model.num_params():,}\n")
    print("=" * 60)

    result = generate(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )

    print(result)
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    checkpoint_path = os.path.join(script_dir, "gpt_best.pth")

    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        print("Train the model first: python train.py")
        sys.exit(1)

    prompt = sys.argv[1] if len(sys.argv) > 1 else "I had always thought"

    load_and_generate(
        checkpoint_path=checkpoint_path,
        prompt=prompt,
        max_new_tokens=300,
        temperature=1.0,
        top_k=50,
    )
