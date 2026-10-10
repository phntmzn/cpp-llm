"""Tests for the C++ LLM token dataset."""

import numpy as np
import pytest
import torch

from cpp_llm.data.dataset import TokenDataset


@pytest.fixture
def sample_tokens() -> np.ndarray:
    """Create a sample token sequence."""
    return np.arange(100, dtype=np.uint32)


@pytest.fixture
def dataset(sample_tokens: np.ndarray) -> TokenDataset:
    """Create a dataset with a sequence length of 8."""
    return TokenDataset(sample_tokens, sequence_length=8)


def test_dataset_initialization(sample_tokens: np.ndarray) -> None:
    """Test that the dataset initializes correctly."""
    dataset = TokenDataset(sample_tokens, sequence_length=8)

    assert len(dataset) == len(sample_tokens) - 8
    assert dataset.sequence_length == 8


def test_dataset_length(sample_tokens: np.ndarray) -> None:
    """Test the number of available training examples."""
    sequence_length = 10
    dataset = TokenDataset(
        sample_tokens,
        sequence_length=sequence_length,
    )

    assert len(dataset) == len(sample_tokens) - sequence_length


def test_dataset_returns_tensors(dataset: TokenDataset) -> None:
    """Test that each item contains PyTorch tensors."""
    inputs, targets = dataset[0]

    assert isinstance(inputs, torch.Tensor)
    assert isinstance(targets, torch.Tensor)


def test_dataset_tensor_shapes(dataset: TokenDataset) -> None:
    """Test the shapes of input and target sequences."""
    inputs, targets = dataset[0]

    assert inputs.shape == (8,)
    assert targets.shape == (8,)


def test_dataset_tensor_dtypes(dataset: TokenDataset) -> None:
    """Test that token tensors use torch.long."""
    inputs, targets = dataset[0]

    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long


def test_dataset_next_token_alignment(dataset: TokenDataset) -> None:
    """Test that targets are shifted one token ahead of inputs."""
    inputs, targets = dataset[0]

    expected_inputs = torch.arange(8, dtype=torch.long)
    expected_targets = torch.arange(1, 9, dtype=torch.long)

    assert torch.equal(inputs, expected_inputs)
    assert torch.equal(targets, expected_targets)


def test_dataset_last_item(dataset: TokenDataset) -> None:
    """Test the final valid training example."""
    inputs, targets = dataset[len(dataset) - 1]

    assert torch.equal(inputs, torch.arange(91, 99, dtype=torch.long))
    assert torch.equal(targets, torch.arange(92, 100, dtype=torch.long))


def test_dataset_different_sequence_lengths(
    sample_tokens: np.ndarray,
) -> None:
    """Test datasets with different sequence lengths."""
    for sequence_length in (1, 4, 16, 32):
        dataset = TokenDataset(
            sample_tokens,
            sequence_length=sequence_length,
        )

        inputs, targets = dataset[0]

        assert inputs.shape == (sequence_length,)
        assert targets.shape == (sequence_length,)
        assert len(dataset) == len(sample_tokens) - sequence_length


def test_dataset_accepts_uint16_tokens() -> None:
    """Test compatibility with uint16 token arrays."""
    tokens = np.arange(50, dtype=np.uint16)
    dataset = TokenDataset(tokens, sequence_length=5)

    inputs, targets = dataset[0]

    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long


def test_dataset_accepts_uint32_tokens() -> None:
    """Test compatibility with uint32 token arrays."""
    tokens = np.arange(50, dtype=np.uint32)
    dataset = TokenDataset(tokens, sequence_length=5)

    inputs, targets = dataset[0]

    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long


def test_dataset_supports_dataloader(dataset: TokenDataset) -> None:
    """Test that the dataset works with a PyTorch DataLoader."""
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    inputs, targets = next(iter(loader))

    assert inputs.shape == (4, 8)
    assert targets.shape == (4, 8)


def test_dataset_rejects_invalid_sequence_length(
    sample_tokens: np.ndarray,
) -> None:
    """Test that non-positive sequence lengths are rejected."""
    for sequence_length in (0, -1):
        with pytest.raises(ValueError):
            TokenDataset(
                sample_tokens,
                sequence_length=sequence_length,
            )


def test_dataset_rejects_insufficient_tokens() -> None:
    """Test that too few tokens for a training example are rejected."""
    tokens = np.array([1, 2, 3], dtype=np.uint32)

    with pytest.raises(ValueError):
        TokenDataset(tokens, sequence_length=3)


def test_dataset_rejects_empty_tokens() -> None:
    """Test that empty token arrays are rejected."""
    tokens = np.array([], dtype=np.uint32)

    with pytest.raises(ValueError):
        TokenDataset(tokens, sequence_length=4)


def test_dataset_rejects_non_1d_tokens() -> None:
    """Test that multidimensional token arrays are rejected."""
    tokens = np.arange(20, dtype=np.uint32).reshape(4, 5)

    with pytest.raises(ValueError):
        TokenDataset(tokens, sequence_length=3)


def test_dataset_out_of_range_index(dataset: TokenDataset) -> None:
    """Test that invalid indices raise IndexError."""
    with pytest.raises(IndexError):
        dataset[len(dataset)]

    with pytest.raises(IndexError):
        dataset[-len(dataset) - 1]
