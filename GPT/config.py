from dataclasses import dataclass


@dataclass
class GPTConfig:
    # --- Model architecture ---
    vocab_size: int = 256        # character-level vocab (full ASCII range)
    context_length: int = 256    # max tokens the model sees at once
    d_model: int = 256           # embedding dimension
    n_heads: int = 8             # number of attention heads (d_model must be divisible)
    n_layers: int = 6            # number of Transformer blocks
    dropout: float = 0.1         # dropout rate

    # --- Training ---
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_epochs: int = 100
    eval_interval: int = 10      # evaluate every N epochs
    train_split: float = 0.9     # fraction of data used for training

    # --- Generation ---
    max_new_tokens: int = 300    # tokens to generate
    temperature: float = 1.0     # >1 = more random, <1 = more greedy
    top_k: int = 50              # keep only top-k candidates (0 = disabled)


# Small config for quick local testing
SmallGPTConfig = GPTConfig(
    vocab_size=256,
    context_length=128,
    d_model=128,
    n_heads=4,
    n_layers=4,
    dropout=0.1,
    batch_size=16,
    learning_rate=3e-4,
    max_epochs=200,
    eval_interval=20,
)
