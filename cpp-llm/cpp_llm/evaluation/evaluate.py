"""Evaluate a trained cpp-llm language model."""

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from cpp_llm.models.gpt import GPT


class TokenDataset(Dataset):
    """Create input-target sequences from token IDs."""

    def __init__(
        self,
        tokens: np.ndarray,
        sequence_length: int,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive.")

        if len(tokens) <= sequence_length:
            raise ValueError(
                "The evaluation dataset must contain more tokens "
                "than the sequence length."
            )

        self.tokens = torch.as_tensor(tokens, dtype=torch.long)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.tokens) - self.sequence_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.tokens[
            index : index + self.sequence_length + 1
        ]
        return sequence[:-1], sequence[1:]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the YAML configuration."""
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    if not isinstance(config.get("model"), dict):
        raise ValueError("Configuration must contain a 'model' section.")

    return config


def load_tokens(path: str | Path) -> np.ndarray:
    """Load token IDs stored in a binary uint32 file."""
    token_path = Path(path)

    if not token_path.is_file():
        raise FileNotFoundError(
            f"Evaluation data not found: {token_path}"
        )

    if token_path.stat().st_size == 0:
        raise ValueError(
            f"Evaluation data is empty: {token_path}"
        )

    tokens = np.fromfile(token_path, dtype=np.uint32)

    if tokens.size == 0:
        raise ValueError("No tokens were found in the evaluation file.")

    return tokens.astype(np.int64)


def select_device(device_name: str = "auto") -> torch.device:
    """Select CPU, CUDA, or Apple Silicon MPS."""
    if device_name != "auto":
        device = torch.device(device_name)

        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable.")

        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Apple Silicon MPS is unavailable.")

        return device

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    device: torch.device,
) -> GPT:
    """Initialize the GPT model and restore trained weights."""
    path = Path(checkpoint_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    model = GPT(**config["model"]).to(device)

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Calculate average loss, perplexity, and next-token accuracy."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    correct_predictions = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)

        if logits.ndim != 3:
            raise ValueError(
                "Expected logits shaped "
                "[batch, sequence_length, vocab_size]."
            )

        if logits.shape[:2] != targets.shape:
            raise ValueError(
                f"Logits shape {tuple(logits.shape)} does not match "
                f"target shape {tuple(targets.shape)}."
            )

        batch_size, sequence_length, vocab_size = logits.shape

        loss_sum = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            reduction="sum",
        )

        predictions = logits.argmax(dim=-1)

        total_loss += loss_sum.item()
        total_tokens += batch_size * sequence_length
        correct_predictions += (
            predictions == targets
        ).sum().item()

    if total_tokens == 0:
        raise ValueError("The evaluation dataset contains no tokens.")

    average_loss = total_loss / total_tokens

    # Perplexity can overflow for very large loss values.
    perplexity = (
        math.exp(average_loss)
        if average_loss < math.log(float.fromhex("0x1.fffffffffffffp+1023"))
        else float("inf")
    )

    accuracy = correct_predictions / total_tokens

    return {
        "loss": average_loss,
        "perplexity": perplexity,
        "next_token_accuracy": accuracy,
    }


def evaluate(
    config_path: str | Path = "configs/model.yaml",
    checkpoint_path: str | Path = "checkpoints/latest.pt",
    data_path: str | Path | None = None,
    batch_size: int = 8,
    device_name: str = "auto",
) -> dict[str, float]:
    """Load a model and evaluate it on tokenized validation data."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    config = load_config(config_path)
    device = select_device(device_name)

    data_config = config.get("data", {})
    model_config = config["model"]

    sequence_length = int(
        model_config.get(
            "block_size",
            model_config.get("max_sequence_length", 256),
        )
    )

    if data_path is None:
        data_path = data_config.get(
            "val_path",
            "data/processed/val.bin",
        )

    tokens = load_tokens(data_path)

    dataset = TokenDataset(
        tokens=tokens,
        sequence_length=sequence_length,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = load_model(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    metrics = evaluate_model(
        model=model,
        data_loader=data_loader,
        device=device,
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Evaluation data: {data_path}")
    print(f"Validation loss: {metrics['loss']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.4f}")
    print(
        "Next-token accuracy: "
        f"{metrics['next_token_accuracy']:.2%}"
    )

    return metrics


def main() -> None:
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate the cpp-llm language model."
    )

    parser.add_argument(
        "--config",
        default="configs/model.yaml",
        help="Path to the model configuration.",
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/latest.pt",
        help="Path to the trained model checkpoint.",
    )

    parser.add_argument(
        "--data",
        default=None,
        help="Path to binary evaluation tokens.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Device used for evaluation.",
    )

    args = parser.parse_args()

    evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        batch_size=args.batch_size,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
