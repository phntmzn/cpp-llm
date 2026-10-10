"""Tests for the C++ language model tokenizer."""

from pathlib import Path

import numpy as np
import pytest

from cpp_llm.tokenizer.tokenizer import CppTokenizer


@pytest.fixture
def tokenizer() -> CppTokenizer:
    """Create a tokenizer instance."""
    return CppTokenizer()


@pytest.fixture
def sample_cpp() -> str:
    """Provide a sample C++ program."""
    return """#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
"""


def test_tokenizer_initialization(tokenizer: CppTokenizer) -> None:
    """Test that the tokenizer initializes correctly."""
    assert tokenizer is not None
    assert tokenizer.vocab_size > 0


def test_tokenizer_has_eos_token(tokenizer: CppTokenizer) -> None:
    """Test that the tokenizer exposes an end-of-text token."""
    assert isinstance(tokenizer.eos_token_id, int)
    assert tokenizer.eos_token_id >= 0


def test_encode_returns_list(tokenizer: CppTokenizer) -> None:
    """Test that encoding returns a list of token IDs."""
    tokens = tokenizer.encode("int main() { return 0; }")

    assert isinstance(tokens, list)
    assert all(isinstance(token, int) for token in tokens)


def test_encode_nonempty_text(tokenizer: CppTokenizer) -> None:
    """Test encoding a nonempty string."""
    tokens = tokenizer.encode("int main() {}")

    assert len(tokens) > 0


def test_decode_returns_string(tokenizer: CppTokenizer) -> None:
    """Test that decoding returns a string."""
    tokens = tokenizer.encode("int value = 42;")
    decoded = tokenizer.decode(tokens)

    assert isinstance(decoded, str)


def test_encode_decode_round_trip(
    tokenizer: CppTokenizer,
    sample_cpp: str,
) -> None:
    """Test that encoding and decoding preserve source text."""
    tokens = tokenizer.encode(sample_cpp)
    decoded = tokenizer.decode(tokens)

    assert decoded == sample_cpp


def test_encode_empty_string(tokenizer: CppTokenizer) -> None:
    """Test encoding an empty string."""
    tokens = tokenizer.encode("")

    assert tokens == []


def test_decode_empty_tokens(tokenizer: CppTokenizer) -> None:
    """Test decoding an empty token sequence."""
    decoded = tokenizer.decode([])

    assert decoded == ""


def test_encode_cpp_keywords(tokenizer: CppTokenizer) -> None:
    """Test encoding common C++ keywords."""
    code = "class struct public private virtual constexpr"
    tokens = tokenizer.encode(code)

    assert tokens
    assert tokenizer.decode(tokens) == code


def test_encode_cpp_operators(tokenizer: CppTokenizer) -> None:
    """Test encoding common C++ operators."""
    code = "a += b; x == y; ptr->value; result && valid;"
    tokens = tokenizer.encode(code)

    assert tokens
    assert tokenizer.decode(tokens) == code


def test_encode_cpp_comments(tokenizer: CppTokenizer) -> None:
    """Test encoding C++ comments."""
    code = """// Single-line comment
/* Multi-line
   comment */
int value = 10;
"""
    tokens = tokenizer.encode(code)

    assert tokens
    assert tokenizer.decode(tokens) == code


def test_encode_unicode(tokenizer: CppTokenizer) -> None:
    """Test encoding and decoding Unicode text."""
    text = "// Unicode: café, naïve, λ\n"
    tokens = tokenizer.encode(text)

    assert tokenizer.decode(tokens) == text


def test_encode_file(
    tokenizer: CppTokenizer,
    tmp_path: Path,
    sample_cpp: str,
) -> None:
    """Test tokenizing a source file."""
    source_file = tmp_path / "example.cpp"
    source_file.write_text(sample_cpp, encoding="utf-8")

    tokens = tokenizer.encode_file(str(source_file))

    assert isinstance(tokens, list)
    assert tokens
    assert tokenizer.decode(tokens) == sample_cpp


def test_save_and_load_tokens(
    tokenizer: CppTokenizer,
    tmp_path: Path,
) -> None:
    """Test saving tokens to disk and loading them again."""
    original_tokens = tokenizer.encode("int main() { return 0; }")
    token_file = tmp_path / "tokens.bin"

    tokenizer.save_tokens(original_tokens, str(token_file))
    loaded_tokens = tokenizer.load_tokens(str(token_file))

    assert np.array_equal(
        np.asarray(original_tokens, dtype=np.uint32),
        loaded_tokens,
    )


def test_saved_tokens_use_uint32(
    tokenizer: CppTokenizer,
    tmp_path: Path,
) -> None:
    """Test that token files use the expected uint32 representation."""
    tokens = tokenizer.encode("std::vector<int> values;")
    token_file = tmp_path / "tokens.bin"

    tokenizer.save_tokens(tokens, str(token_file))
    loaded_tokens = tokenizer.load_tokens(str(token_file))

    assert loaded_tokens.dtype == np.uint32


def test_save_tokens_creates_file(
    tokenizer: CppTokenizer,
    tmp_path: Path,
) -> None:
    """Test that saving tokens creates a binary file."""
    tokens = tokenizer.encode("return 0;")
    token_file = tmp_path / "tokens.bin"

    tokenizer.save_tokens(tokens, str(token_file))

    assert token_file.exists()
    assert token_file.stat().st_size > 0


def test_save_and_load_empty_tokens(
    tokenizer: CppTokenizer,
    tmp_path: Path,
) -> None:
    """Test saving and loading an empty token sequence."""
    token_file = tmp_path / "empty.bin"

    tokenizer.save_tokens([], str(token_file))
    loaded_tokens = tokenizer.load_tokens(str(token_file))

    assert isinstance(loaded_tokens, np.ndarray)
    assert loaded_tokens.size == 0


def test_count_tokens(tokenizer: CppTokenizer) -> None:
    """Test the token counting helper."""
    text = "int main() { return 0; }"
    tokens = tokenizer.encode(text)

    assert tokenizer.count_tokens(text) == len(tokens)


def test_token_ids_within_vocabulary(tokenizer: CppTokenizer) -> None:
    """Test that encoded token IDs are valid vocabulary IDs."""
    tokens = tokenizer.encode("int main() { return 0; }")

    assert all(0 <= token < tokenizer.vocab_size for token in tokens)


def test_repeated_encoding_is_deterministic(
    tokenizer: CppTokenizer,
) -> None:
    """Test that identical input produces identical token IDs."""
    text = "std::cout << value << std::endl;"

    first = tokenizer.encode(text)
    second = tokenizer.encode(text)

    assert first == second


def test_different_texts_can_produce_different_tokens(
    tokenizer: CppTokenizer,
) -> None:
    """Test that different inputs can have different encodings."""
    first = tokenizer.encode("int")
    second = tokenizer.encode("return")

    assert first != second


def test_encode_file_missing_path(
    tokenizer: CppTokenizer,
    tmp_path: Path,
) -> None:
    """Test handling a missing source file."""
    missing_file = tmp_path / "missing.cpp"

    with pytest.raises((FileNotFoundError, OSError)):
        tokenizer.encode_file(str(missing_file))


def test_load_tokens_missing_path(
    tokenizer: CppTokenizer,
    tmp_path: Path,
) -> None:
    """Test handling a missing token file."""
    missing_file = tmp_path / "missing.bin"

    with pytest.raises((FileNotFoundError, OSError)):
        tokenizer.load_tokens(str(missing_file))


def test_decode_tokens_from_numpy_array(
    tokenizer: CppTokenizer,
) -> None:
    """Test decoding token IDs stored in a NumPy array."""
    text = "int answer = 42;"
    tokens = np.asarray(tokenizer.encode(text), dtype=np.uint32)

    decoded = tokenizer.decode(tokens.tolist())

    assert decoded == text
