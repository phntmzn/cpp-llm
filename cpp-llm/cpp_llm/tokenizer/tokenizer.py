"""Tokenizer utilities for the cpp-llm project."""

from pathlib import Path

import tiktoken


class CppTokenizer:
    """Encode and decode C++ source code using a pretrained tokenizer."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        """Initialize the tokenizer.

        Args:
            encoding_name: Name of the tiktoken encoding to use.
                Defaults to cl100k_base.
        """
        self.encoding_name = encoding_name
        self.encoding = tiktoken.get_encoding(encoding_name)

        self.vocab_size = self.encoding.n_vocab
        self.eos_token_id = self.encoding.eot_token

    def encode(self, text: str) -> list[int]:
        """Convert source code or text into token IDs."""
        if not isinstance(text, str):
            raise TypeError("Input must be a string.")

        return self.encoding.encode(
            text,
            allowed_special={"<|endoftext|>"},
        )

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back into source code or text."""
        if not token_ids:
            return ""

        return self.encoding.decode(token_ids)

    def encode_file(self, path: str | Path) -> list[int]:
        """Read a source file and encode its contents."""
        file_path = Path(path)

        if not file_path.is_file():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        text = file_path.read_text(encoding="utf-8")
        return self.encode(text)

    def decode_to_file(
        self,
        token_ids: list[int],
        path: str | Path,
    ) -> None:
        """Decode token IDs and write the result to a file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        text = self.decode(token_ids)
        output_path.write_text(text, encoding="utf-8")

    def save_tokens(
        self,
        token_ids: list[int],
        path: str | Path,
    ) -> None:
        """Save token IDs as a binary uint32 array."""
        import numpy as np

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tokens = np.asarray(token_ids, dtype=np.uint32)
        tokens.tofile(output_path)

    def load_tokens(self, path: str | Path) -> list[int]:
        """Load token IDs from a binary uint32 array."""
        import numpy as np

        input_path = Path(path)

        if not input_path.is_file():
            raise FileNotFoundError(f"Token file not found: {input_path}")

        if input_path.stat().st_size == 0:
            return []

        tokens = np.fromfile(input_path, dtype=np.uint32)
        return tokens.astype(np.int64).tolist()

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        return len(self.encode(text))


def main() -> None:
    """Run a basic tokenizer demonstration."""
    tokenizer = CppTokenizer()

    sample_code = """
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
"""

    token_ids = tokenizer.encode(sample_code)
    decoded_text = tokenizer.decode(token_ids)

    print("Original source:")
    print(sample_code)

    print(f"Token count: {len(token_ids)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Token IDs: {token_ids}")

    print("\nDecoded source:")
    print(decoded_text)


if __name__ == "__main__":
    main()
