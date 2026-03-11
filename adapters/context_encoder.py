"""
Context Encoder (Sub-module 2a).

Encodes the previous sentence translation Y_{n-1} into a context vector c.

Architecture:
  - Input text → base model's text encoder (frozen) → mean pooling
  - → learned linear projection → context vector c

Only the projection layer is trainable.

TODO: Full implementation after baseline evaluation.
"""

import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    """Encodes previous translation text into a context vector.

    Args:
        text_encoder: Frozen text encoder (e.g., from base model's mBART).
        encoder_dim: Output dimension of the text encoder.
        context_dim: Dimension of the projected context vector.
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        encoder_dim: int = 1024,
        context_dim: int = 256,
    ):
        super().__init__()
        self.text_encoder = text_encoder

        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Trainable projection
        self.projection = nn.Linear(encoder_dim, context_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text and project to context vector.

        Args:
            input_ids: Tokenized text input (B, seq_len).
            attention_mask: Attention mask (B, seq_len).

        Returns:
            Context vector c of shape (B, context_dim).
        """
        raise NotImplementedError
