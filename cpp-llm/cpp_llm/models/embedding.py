"""Token and positional embeddings for cpp-llm."""

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Convert token IDs into dense vector representations."""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        """Initialize the token embedding layer."""
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs to embedding vectors.

        Args:
            token_ids: Integer tensor shaped [batch_size, sequence_length].

        Returns:
            Embeddings shaped [batch_size, sequence_length, embedding_dim].
        """
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    """Learnable positional embeddings for sequence tokens."""

    def __init__(self, max_sequence_length: int, embedding_dim: int) -> None:
        """Initialize positional embeddings."""
        super().__init__()

        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive.")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        self.max_sequence_length = max_sequence_length

        self.embedding = nn.Embedding(
            num_embeddings=max_sequence_length,
            embedding_dim=embedding_dim,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Generate positional embeddings matching the input sequence.

        Args:
            token_ids: Integer tensor shaped [batch_size, sequence_length].

        Returns:
            Positional embeddings shaped
            [batch_size, sequence_length, embedding_dim].
        """
        sequence_length = token_ids.size(1)

        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds maximum "
                f"length {self.max_sequence_length}."
            )

        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        positions = positions.unsqueeze(0).expand(token_ids.size(0), -1)

        return self.embedding(positions)


class GPTEmbedding(nn.Module):
    """Combine token and positional embeddings for a GPT model."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_sequence_length: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the combined embedding layer."""
        super().__init__()

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.position_embedding = PositionalEmbedding(
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Combine token and positional embeddings.

        Args:
            token_ids: Integer tensor shaped [batch_size, sequence_length].

        Returns:
            Embedded sequence shaped
            [batch_size, sequence_length, embedding_dim].
        """
        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(token_ids)

        embeddings = token_vectors + position_vectors

        return self.dropout(embeddings)


if __name__ == "__main__":
    batch_size = 2
    sequence_length = 16
    vocab_size = 50257
    embedding_dim = 256
    max_sequence_length = 256

    sample_tokens = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_length),
    )

    embedding_layer = GPTEmbedding(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_sequence_length=max_sequence_length,
        dropout=0.1,
    )

    output = embedding_layer(sample_tokens)

    print("Input shape:", tuple(sample_tokens.shape))
    print("Output shape:", tuple(output.shape))
    print("Expected shape:", (batch_size, sequence_length, embedding_dim))
