"""Training engine for the cpp-llm language model."""

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from cpp_llm.training.checkpoint import save_checkpoint


class Trainer:
    """Train and evaluate a GPT-style language model."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: str | torch.device = "cpu",
        gradient_clip: float = 1.0,
        scheduler: Any | None = None,
        checkpoint_dir: str | Path = "checkpoints",
        save_every_steps: int = 500,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the training engine."""
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.gradient_clip = gradient_clip
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every_steps = save_every_steps
        self.config = config or {}

        self.global_step = 0
        self.start_epoch = 0
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        self.model.to(self.device)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate next-token prediction cross-entropy loss."""
        logits = self.model(inputs)

        if logits.ndim != 3:
            raise ValueError(
                "Expected model logits with shape "
                "[batch, sequence_length, vocab_size]."
            )

        if logits.shape[:2] != targets.shape:
            raise ValueError(
                f"Logits shape {tuple(logits.shape)} is incompatible "
                f"with target shape {tuple(targets.shape)}."
            )

        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Perform one optimization step."""
        self.model.train()

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        loss = self._calculate_loss(inputs, targets)
        loss.backward()

        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.gradient_clip,
            )

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1

        return float(loss.detach().item())

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate average loss on the validation dataset."""
        if self.val_loader is None:
            raise ValueError("No validation DataLoader was provided.")

        self.model.eval()

        total_loss = 0.0
        batch_count = 0

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            loss = self._calculate_loss(inputs, targets)

            total_loss += loss.item()
            batch_count += 1

        if batch_count == 0:
            raise ValueError("The validation DataLoader is empty.")

        return total_loss / batch_count

    def save(self, epoch: int, loss: float) -> Path:
        """Save the current model and optimizer state."""
        checkpoint_path = self.checkpoint_dir / "latest.pt"

        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            step=self.global_step,
            loss=loss,
            path=checkpoint_path,
            config=self.config,
        )

        return checkpoint_path

    def fit(self, epochs: int) -> dict[str, list[float]]:
        """Run the training and validation loops."""
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")

        if len(self.train_loader) == 0:
            raise ValueError("The training DataLoader is empty.")

        for epoch in range(self.start_epoch, epochs):
            self.model.train()

            total_loss = 0.0
            batch_count = 0

            print(f"\nEpoch {epoch + 1}/{epochs}")

            for inputs, targets in self.train_loader:
                loss = self.train_step(inputs, targets)

                total_loss += loss
                batch_count += 1

                if (
                    self.save_every_steps > 0
                    and self.global_step % self.save_every_steps == 0
                ):
                    self.save(epoch, loss)

                if self.global_step % 50 == 0:
                    print(
                        f"Step {self.global_step} | "
                        f"Training loss: {loss:.4f}"
                    )

            train_loss = total_loss / batch_count
            self.history["train_loss"].append(train_loss)

            message = f"Training loss: {train_loss:.4f}"

            if self.val_loader is not None:
                val_loss = self.evaluate()
                self.history["val_loss"].append(val_loss)
                message += f" | Validation loss: {val_loss:.4f}"
                checkpoint_loss = val_loss
            else:
                checkpoint_loss = train_loss

            print(message)

            self.save(epoch + 1, checkpoint_loss)

        print("\nTraining complete.")

        return self.history
