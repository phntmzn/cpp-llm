"""Generate C++ code using a trained cpp-llm model."""

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from cpp_llm.models.gpt import GPT
from cpp_llm.tokenizer.tokenizer import CppTokenizer


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load model configuration from a YAML file."""
    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    if not isinstance(config.get("model"), dict):
        raise ValueError("Configuration must contain a 'model' section.")

    return config


def load_model(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    device: torch.device,
) -> GPT:
    """Initialize the model and load trained weights."""
    path = Path(checkpoint_path)

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model = GPT(**config["model"]).to(device)

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    # Supports checkpoints created by cpp_llm.training.checkpoint.
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # Also supports a checkpoint containing only the state dictionary.
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model


@torch.inference_mode()
def generate_tokens(
    model: GPT,
    input_ids: torch.Tensor,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 50,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Generate tokens autoregressively.

    Args:
        model: Trained GPT model returning logits shaped
            [batch, sequence_length, vocab_size].
        input_ids: Prompt token IDs shaped [batch, sequence_length].
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. Lower values are more
            deterministic; higher values increase randomness.
        top_k: If specified, sample only from the top-k next-token logits.
        eos_token_id: Optional token ID that stops generation.

    Returns:
        Input token IDs concatenated with newly generated token IDs.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence_length].")

    if input_ids.size(1) == 0:
        raise ValueError("The prompt must contain at least one token.")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative.")

    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")

    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1 or None.")

    model.eval()
    generated = input_ids

    # GPT's configured context window.
    max_context = getattr(
        model,
        "block_size",
        getattr(model, "max_sequence_length", None),
    )

    for _ in range(max_new_tokens):
        # Keep only the most recent tokens if the context window is full.
        if max_context is not None:
            context = generated[:, -max_context:]
        else:
            context = generated

        logits = model(context)

        if logits.ndim != 3:
            raise ValueError(
                "Expected model logits shaped "
                "[batch, sequence_length, vocab_size]."
            )

        # Use the last position to predict the next token.
        next_token_logits = logits[:, -1, :] / temperature

        if top_k is not None:
            k = min(top_k, next_token_logits.size(-1))
            top_values, _ = torch.topk(next_token_logits, k=k, dim=-1)

            threshold = top_values[:, -1].unsqueeze(-1)
            next_token_logits = next_token_logits.masked_fill(
                next_token_logits < threshold,
                float("-inf"),
            )

        probabilities = torch.softmax(next_token_logits, dim=-1)

        next_token = torch.multinomial(probabilities, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)

        if eos_token_id is not None:
            if torch.all(next_token.squeeze(-1) == eos_token_id):
                break

    return generated


def choose_device(device_name: str = "auto") -> torch.device:
    """Select the requested device or detect an available accelerator."""
    if device_name != "auto":
        device = torch.device(device_name)

        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        if device.type == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested but is not available.")

        return device

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def generate(
    prompt: str,
    config_path: str | Path = "configs/model.yaml",
    checkpoint_path: str | Path = "checkpoints/latest.pt",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 50,
    device_name: str = "auto",
) -> str:
    """Generate C++ code from a text prompt."""
    config = load_config(config_path)
    device = choose_device(device_name)

    tokenizer = CppTokenizer()
    model = load_model(config, checkpoint_path, device)

    prompt_ids = tokenizer.encode(prompt)

    if not prompt_ids:
        raise ValueError("The prompt produced no tokens.")

    input_tensor = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    output_ids = generate_tokens(
        model=model,
        input_ids=input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode only the newly generated portion.
    generated_ids = output_ids[0, len(prompt_ids):].tolist()

    return prompt + tokenizer.decode(generated_ids)


def main() -> None:
    """Parse command-line arguments and generate code."""
    parser = argparse.ArgumentParser(
        description="Generate C++ code with cpp-llm."
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="#include <iostream>\n\nint main() {\n",
        help="Prompt text to continue.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model.yaml",
        help="Path to model YAML configuration.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/latest.pt",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling limit; use 0 to disable.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device used for generation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional file path for generated C++ code.",
    )

    args = parser.parse_args()

    result = generate(
        prompt=args.prompt,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        device_name=args.device,
    )

    print(result)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result, encoding="utf-8")
        print(f"\nGenerated code saved to {output_path}")


if __name__ == "__main__":
    main()
