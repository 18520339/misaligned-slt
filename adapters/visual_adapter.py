"""
Boundary-Aware Visual Adapter (Sub-module 2b).

Gated cross-attention between visual features and context:
  h' = h + α * CrossAttn(h, c)

CRITICAL: α (gate scalar) is INITIALIZED TO ZERO, ensuring the adapter
starts as an identity function and the base model behavior is preserved
at the beginning of training.

TODO: Full implementation after baseline evaluation.
"""

import torch
import torch.nn as nn


class VisualAdapter(nn.Module):
    """Gated cross-attention adapter for visual features.

    Args:
        visual_dim: Dimension of visual features h.
        context_dim: Dimension of context vector c.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        gate_init: Initial value of the gate scalar α (default: 0.0).
    """

    def __init__(
        self,
        visual_dim: int = 256,
        context_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        gate_init: float = 0.0,
    ):
        super().__init__()

        # Cross-attention: queries from visual, keys/values from context
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=num_heads,
            kdim=context_dim,
            vdim=context_dim,
            dropout=dropout,
            batch_first=True,
        )

        # Gate scalar α — INITIALIZED TO ZERO
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def forward(
        self,
        visual_features: torch.Tensor,
        context_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Apply gated cross-attention.

        Args:
            visual_features: h of shape (B, T, visual_dim).
            context_vector: c of shape (B, context_dim), expanded for cross-attn.

        Returns:
            Adapted visual features h' of shape (B, T, visual_dim).
        """
        raise NotImplementedError
