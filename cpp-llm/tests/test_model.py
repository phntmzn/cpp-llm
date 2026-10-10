"""Tests for the GPT language model."""

import pytest
import torch

from cpp_llm.models.gpt import GPT


@pytest.fixture
def model_config() -> dict:
    """Provide a small GPT configuration for testing."""
    return {
        "vocab_size": 256,
        "embedding_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "max_sequence_length": 32,
        "dropout": 0.0,
    }


@pytest.fixture
def model(model_config: dict) -> GPT:
    """Create a small GPT model."""
    return GPT(**model_config)


def test_model_initialization(model: GPT) -> None:
    """Test that the model initializes successfully."""
    assert isinstance(model, GPT)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_model_forward_shape(model: GPT) -> None:
    """Test the output shape of the forward pass."""
    input_ids = torch.randint(0, 256, (2, 16))

    logits = model(input_ids)

    assert logits.shape == (2, 16, 256)


def test_model_single_token_input(model: GPT) -> None:
    """Test that the model processes a single token."""
    input_ids = torch.randint(0, 256, (1, 1))

    logits = model(input_ids)

    assert logits.shape == (1, 1, 256)


def test_model_batch_size_one(model: GPT) -> None:
    """Test inference with a batch size of one."""
    input_ids = torch.randint(0, 256, (1, 8))

    logits = model(input_ids)

    assert logits.shape == (1, 8, 256)


def test_model_different_sequence_lengths(model: GPT) -> None:
    """Test the model with several valid sequence lengths."""
    for sequence_length in (1, 4, 8, 16, 32):
        input_ids = torch.randint(0, 256, (2, sequence_length))

        logits = model(input_ids)

        assert logits.shape == (2, sequence_length, 256)


def test_model_different_batch_sizes(model: GPT) -> None:
    """Test the model with several batch sizes."""
    for batch_size in (1, 2, 4):
        input_ids = torch.randint(0, 256, (batch_size, 8))

        logits = model(input_ids)

        assert logits.shape == (batch_size, 8, 256)


def test_model_outputs_finite_logits(model: GPT) -> None:
    """Test that model outputs contain no NaN or infinite values."""
    input_ids = torch.randint(0, 256, (2, 12))

    logits = model(input_ids)

    assert torch.isfinite(logits).all()


def test_model_outputs_floating_point_logits(model: GPT) -> None:
    """Test that logits are floating-point tensors."""
    input_ids = torch.randint(0, 256, (2, 8))

    logits = model(input_ids)

    assert logits.is_floating_point()


def test_model_backward_pass(model: GPT) -> None:
    """Test that gradients flow through the model."""
    input_ids = torch.randint(0, 256, (2, 8))
    targets = torch.randint(0, 256, (2, 8))

    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert any(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
        if gradient is not None
    )


def test_model_optimizer_step(model: GPT) -> None:
    """Test that an optimizer step updates model parameters."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, 256, (2, 8))
    targets = torch.randint(0, 256, (2, 8))

    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    optimizer.zero_grad()
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )
    loss.backward()
    optimizer.step()

    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.parameters())
    )


def test_model_eval_mode_is_deterministic(model: GPT) -> None:
    """Test that evaluation mode produces repeatable outputs."""
    model.eval()
    input_ids = torch.randint(0, 256, (2, 8))

    with torch.no_grad():
        first_output = model(input_ids)
        second_output = model(input_ids)

    assert torch.allclose(first_output, second_output)


def test_model_train_mode_supports_dropout(
    model_config: dict,
) -> None:
    """Test that dropout can be enabled during training."""
    model_config["dropout"] = 0.5
    model = GPT(**model_config)
    model.train()

    input_ids = torch.randint(0, 256, (2, 16))

    first_output = model(input_ids)
    second_output = model(input_ids)

    assert first_output.shape == second_output.shape
    assert torch.isfinite(first_output).all()
    assert torch.isfinite(second_output).all()


def test_model_eval_mode_disables_dropout(
    model_config: dict,
) -> None:
    """Test that dropout is disabled in evaluation mode."""
    model_config["dropout"] = 0.5
    model = GPT(**model_config)
    model.eval()

    input_ids = torch.randint(0, 256, (2, 16))

    with torch.no_grad():
        first_output = model(input_ids)
        second_output = model(input_ids)

    assert torch.allclose(first_output, second_output)


def test_model_rejects_sequence_too_long(model: GPT) -> None:
    """Test that inputs exceeding the context window are rejected."""
    input_ids = torch.randint(0, 256, (1, 33))

    with pytest.raises((ValueError, RuntimeError, IndexError)):
        model(input_ids)


def test_model_rejects_out_of_vocabulary_tokens(model: GPT) -> None:
    """Test that token IDs outside the vocabulary are rejected."""
    input_ids = torch.tensor([[0, 1, 256]])

    with pytest.raises((ValueError, RuntimeError, IndexError)):
        model(input_ids)


def test_model_parameter_count(model: GPT) -> None:
    """Test that the model exposes trainable parameters."""
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert trainable_parameters
    assert all(parameter.numel() > 0 for parameter in trainable_parameters)


def test_model_supports_cpu_inference(model: GPT) -> None:
    """Test that the model runs on CPU."""
    model = model.to("cpu")
    input_ids = torch.randint(0, 256, (1, 8), device="cpu")

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.device.type == "cpu"
    assert logits.shape == (1, 8, 256)


def test_model_causal_attention(model: GPT) -> None:
    """Test that future tokens do not affect earlier logits."""
    model.eval()

    original = torch.tensor([[1, 2, 3, 4, 5, 6]])
    modified = torch.tensor([[1, 2, 3, 99, 100, 101]])

    with torch.no_grad():
        original_logits = model(original)
        modified_logits = model(modified)

    # The first three positions must not depend on later input tokens.
    assert torch.allclose(
        original_logits[:, :3, :],
        modified_logits[:, :3, :],
        atol=1e-5,
        rtol=1e-5,
    )


def test_model_loss_is_finite(model: GPT) -> None:
    """Test that next-token prediction loss is finite."""
    input_ids = torch.randint(0, 256, (2, 12))
    targets = torch.randint(0, 256, (2, 12))

    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_model_generate_logits_for_next_token(model: GPT) -> None:
    """Test that the final sequence position provides next-token logits."""
    input_ids = torch.randint(0, 256, (1, 10))

    with torch.no_grad():
        logits = model(input_ids)

    next_token_logits = logits[:, -1, :]

    assert next_token_logits.shape == (1, 256)
