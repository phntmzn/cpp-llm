"""Save and load checkpoints for the cpp-llm training pipeline."""

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    step: int,
    loss: float,
    path: str | Path = "checkpoints/latest.pt",
    config: dict[str, Any] | None = None,
) -> None:
    """Save model and training state to a checkpoint file.

    Args:
        model: Model being trained.
        optimizer: Optimizer used during training.
        epoch: Current epoch number.
        step: Current training step.
        loss: Most recently recorded loss.
        path: Destination checkpoint path.
        config: Optional model or training configuration.
    """
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "config": config or {},
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optimizer | None = None,
    path: str | Path = "checkpoints/latest.pt",
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load model and optional optimizer state from a checkpoint.

    Args:
        model: Model to restore.
        optimizer: Optional optimizer to restore.
        path: Checkpoint file path.
        device: Device on which to load checkpoint tensors.

    Returns:
        Dictionary containing the saved training metadata.
    """
    checkpoint_path = Path(path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        if "optimizer_state_dict" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain optimizer state."
            )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(
        f"Checkpoint loaded: {checkpoint_path} "
        f"(epoch={checkpoint.get('epoch', 0)}, "
        f"step={checkpoint.get('step', 0)})"
    )

    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss"),
        "config": checkpoint.get("config", {}),
    }
