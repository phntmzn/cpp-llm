"""Tests for the GPT causal self-attention module."""

import pytest
import torch

from cpp_llm.models.attention import CausalSelfAttention


@pytest.fixture
def attention():
    """Create a small causal self-attention layer."""
    torch.manual_seed(42)

    return CausalSelfAttention(
        embedding_dim=64,
        num_heads=4,
        dropout=0.0,
    )


@pytest.fixture
def sample_input():
    """Return a batch of sample embeddings."""
    torch.manual_seed(42)
    return torch.randn(2, 8, 64)


def test_attention_output_shape(attention, sample_input):
    """Attention should preserve batch, sequence, and embedding dimensions."""
    output = attention(sample_input)

    assert output.shape == sample_input.shape


def test_attention_output_is_finite(attention, sample_input):
    """Attention outputs should not contain NaN or infinite values."""
    output = attention(sample_input)

    assert torch.isfinite(output).all()


def test_attention_supports_single_token(attention):
    """Attention should handle a sequence containing one token."""
    inputs = torch.randn(1, 1, 64)

    output = attention(inputs)

    assert output.shape == (1, 1, 64)
    assert torch.isfinite(output).all()


def test_attention_supports_different_batch_sizes(attention):
    """Attention should support different batch sizes."""
    for batch_size in (1, 2, 4):
        inputs = torch.randn(batch_size, 8, 64)

        output = attention(inputs)

        assert output.shape == (batch_size, 8, 64)


def test_attention_supports_different_sequence_lengths(attention):
    """Attention should support different sequence lengths."""
    for sequence_length in (1, 4, 8, 16):
        inputs = torch.randn(2, sequence_length, 64)

        output = attention(inputs)

        assert output.shape == (2, sequence_length, 64)


def test_attention_supports_batch_size_one(attention):
    """Attention should work with a single example."""
    inputs = torch.randn(1, 8, 64)

    output = attention(inputs)

    assert output.shape == inputs.shape


def test_attention_backpropagation(attention, sample_input):
    """Gradients should propagate through the attention layer."""
    inputs = sample_input.clone().requires_grad_(True)

    output = attention(inputs)
    loss = output.square().mean()
    loss.backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()

    for parameter in attention.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


def test_attention_parameters_update(attention, sample_input):
    """An optimizer step should update trainable parameters."""
    optimizer = torch.optim.AdamW(attention.parameters(), lr=1e-3)

    before = [
        parameter.detach().clone()
        for parameter in attention.parameters()
        if parameter.requires_grad
    ]

    optimizer.zero_grad(set_to_none=True)
    output = attention(sample_input)
    loss = output.square().mean()
    loss.backward()
    optimizer.step()

    after = [
        parameter.detach()
        for parameter in attention.parameters()
        if parameter.requires_grad
    ]

    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, after)
    )


def test_attention_causal_masking():
    """Changing future tokens must not affect earlier outputs.

    This test assumes causal self-attention and dropout disabled in eval mode.
    """
    torch.manual_seed(42)

    attention = CausalSelfAttention(
        embedding_dim=32,
        num_heads=4,
        dropout=0.0,
    )
    attention.eval()

    original = torch.randn(1, 6, 32)
    modified = original.clone()

    # Modify tokens strictly after position 2.
    modified[:, 3:, :] = torch.randn_like(modified[:, 3:, :]) * 10

    with torch.no_grad():
        original_output = attention(original)
        modified_output = attention(modified)

    # Outputs at positions 0, 1, and 2 must remain unchanged.
    torch.testing.assert_close(
        original_output[:, :3, :],
        modified_output[:, :3, :],
        rtol=1e-5,
        atol=1e-6,
    )


def test_attention_deterministic_in_eval_mode(sample_input):
    """With dropout disabled, repeated evaluation should be deterministic."""
    attention = CausalSelfAttention(
        embedding_dim=64,
        num_heads=4,
        dropout=0.0,
    )
    attention.eval()

    with torch.no_grad():
        first_output = attention(sample_input)
        second_output = attention(sample_input)

    torch.testing.assert_close(first_output, second_output)


def test_attention_dropout_eval_mode():
    """Evaluation mode should disable dropout randomness."""
    torch.manual_seed(42)

    attention = CausalSelfAttention(
        embedding_dim=32,
        num_heads=4,
        dropout=0.5,
    )
    attention.eval()

    inputs = torch.randn(2, 8, 32)

    with torch.no_grad():
        first_output = attention(inputs)
        second_output = attention(inputs)

    torch.testing.assert_close(first_output, second_output)


@pytest.mark.parametrize(
    ("embedding_dim", "num_heads"),
    [
        (64, 4),
        (128, 8),
        (256, 8),
    ],
)
def test_attention_multiple_configurations(embedding_dim, num_heads):
    """Attention should work for valid embedding and head configurations."""
    attention = CausalSelfAttention(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        dropout=0.0,
    )

    inputs = torch.randn(2, 8, embedding_dim)
    output = attention(inputs)

    assert output.shape == inputs.shape


def test_attention_rejects_non_divisible_embedding_dimension():
    """Embedding dimension should be divisible by the number of heads."""
    with pytest.raises((ValueError, AssertionError)):
        CausalSelfAttention(
            embedding_dim=63,
            num_heads=4,
            dropout=0.0,
        )


@pytest.mark.parametrize(
    "invalid_input",
    [
        torch.randn(8, 64),
        torch.randn(2, 8, 64, 1),
    ],
)
def test_attention_rejects_invalid_input_dimensions(
    attention,
    invalid_input,
):
    """Attention should reject inputs that are not three-dimensional."""
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        attention(invalid_input)


def test_attention_empty_sequence(attention):
    """Attention should reject an empty sequence."""
    inputs = torch.randn(2, 0, 64)

    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        attention(inputs)


def test_attention_longer_sequence(attention):
    """Attention should handle a longer sequence if no fixed limit is imposed."""
    inputs = torch.randn(1, 128, 64)

    output = attention(inputs)

    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
