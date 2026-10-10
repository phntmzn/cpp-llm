"""Training entry point for the cpp-llm language model."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from cpp_llm.models.gpt import GPT
from cpp_llm.training.checkpoint import load_checkpoint, save_checkpoint


class TokenDataset(Dataset):
    """Create fixed-length training examples from token IDs."""

    def __init__(self, token_ids: np.ndarray, sequence_length: int):
        if len(token_ids) <= sequence_length:
            raise ValueError(
                "Training data must contain more tokens than sequence_length."
            )

        self.token_ids = torch.as_tensor(token_ids, dtype=torch.long)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.token_ids) - self.sequence_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.token_ids[index : index + self.sequence_length + 1]
        return chunk[:-1], chunk[1:]


def set_seed(seed: int) -> None:
    """Set random seeds for more reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str | Path) -> dict:
    """Load the YAML configuration."""
    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The YAML configuration must contain a mapping.")

    return config


def load_token_data(path: str | Path) -> np.ndarray:
    """Load token IDs saved as a NumPy-compatible binary array."""
    token_path = Path(path)

    if not token_path.is_file():
        raise FileNotFoundError(
            f"Token dataset not found: {token_path}. "
            "Run the data preparation pipeline first."
        )

    if token_path.stat().st_size == 0:
        raise ValueError(
            f"Token dataset is empty: {token_path}. "
            "The .bin file must contain actual token IDs."
        )

    tokens = np.fromfile(token_path, dtype=np.uint16)

    if tokens.size == 0:
        raise ValueError(f"No token IDs found in {token_path}.")

    return tokens.astype(np.int64)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> float:
    """Calculate average next-token prediction loss."""
    model.eval()
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            total_loss += loss.item()
            batch_count += 1

            if batch_count >= max_batches:
                break

    model.train()

    if batch_count == 0:
        raise ValueError("The evaluation dataset produced no batches.")

    return total_loss / batch_count


def train(config_path: str | Path, resume: bool = False) -> None:
    """Run model training using the supplied YAML configuration."""
    config = load_config(config_path)

    training_config = config.get("training", {})
    model_config = config.get("model", {})
    data_config = config.get("data", {})

    seed = int(training_config.get("seed", 42))
    set_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    sequence_length = int(
        model_config.get("block_size", model_config.get("sequence_length", 256))
    )
    batch_size = int(training_config.get("batch_size", 8))
    epochs = int(training_config.get("epochs", 5))
    learning_rate = float(training_config.get("learning_rate", 3e-4))
    checkpoint_path = Path(
        training_config.get("checkpoint_path", "checkpoints/latest.pt")
    )

    train_path = data_config.get("train_path", "data/processed/train.bin")
    val_path = data_config.get("val_path", "data/processed/val.bin")

    train_tokens = load_token_data(train_path)
    val_tokens = load_token_data(val_path)

    train_dataset = TokenDataset(train_tokens, sequence_length)
    val_dataset = TokenDataset(val_tokens, sequence_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = GPT(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(training_config.get("weight_decay", 0.01)),
    )

    start_epoch = 0
    global_step = 0

    if resume:
        state = load_checkpoint(
            model=model,
            optimizer=optimizer,
            path=checkpoint_path,
            device=device,
        )
        start_epoch = int(state["epoch"])
        global_step = int(state["step"])

    gradient_clip = float(training_config.get("gradient_clip", 1.0))
    save_every = int(training_config.get("save_every_steps", 500))

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        for batch_index, (inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            loss.backward()

            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clip,
                )

            optimizer.step()

            global_step += 1
            running_loss += loss.item()

            if global_step % 50 == 0:
                average_loss = running_loss / min(batch_index, 50)
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Step {global_step} | "
                    f"Loss {average_loss:.4f}"
                )
                running_loss = 0.0

            if save_every > 0 and global_step % save_every == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=global_step,
                    loss=loss.item(),
                    path=checkpoint_path,
                    config=config,
                )

        validation_loss = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{epochs} complete | "
            f"Validation loss: {validation_loss:.4f}"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            step=global_step,
            loss=validation_loss,
            path=checkpoint_path,
            config=config,
        )

    print("Training complete.")


def main() -> None:
    """Parse command-line arguments and start training."""
    parser = argparse.ArgumentParser(
        description="Train the cpp-llm language model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model.yaml",
        help="Path to the YAML model configuration.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the configured checkpoint.",
    )

    args = parser.parse_args()
    train(config_path=args.config, resume=args.resume)


if __name__ == "__main__":
    main()
