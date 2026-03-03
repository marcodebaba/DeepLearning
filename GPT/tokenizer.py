"""
Character-level tokenizer.

Maps every unique character in the training corpus to an integer index.
This is the simplest tokenizer — no subword merging, just raw characters.

Vocabulary example:
  'a' -> 10,  'b' -> 11,  ' ' -> 0,  '\n' -> 1, ...
"""


class CharTokenizer:
    def __init__(self, text: str):
        # Build vocabulary from all unique characters in the corpus
        chars = sorted(set(text))
        self.vocab_size = len(chars)

        # char -> index
        self.char_to_idx: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        # index -> char
        self.idx_to_char: dict[int, str] = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of token IDs."""
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Convert a list of token IDs back to a string."""
        return "".join(self.idx_to_char[i] for i in ids)

    def __len__(self) -> int:
        return self.vocab_size


def build_tokenizer(path: str) -> tuple["CharTokenizer", str]:
    """Load text file and return (tokenizer, raw_text)."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    print(f"Corpus length: {len(text):,} chars | Vocab size: {tokenizer.vocab_size}")
    return tokenizer, text
