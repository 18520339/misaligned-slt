"""
Boundary-Aware Context Adapter (BACA) — main wrapper.

BACA wraps a FROZEN pretrained SLT model to make it robust to temporal
misalignment. It adds three sub-modules:
  1. Context Encoder: encodes previous sentence translation Y_{n-1}
  2. Visual Adapter: gated cross-attention between visual features and context
  3. Boundary Embedding: learnable indicators for segment start/end types

All base model weights remain frozen. Only adapter weights are trained.

TODO: Full implementation after baseline evaluation is complete.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BACA(nn.Module):
    """Boundary-Aware Context Adapter.

    Wraps a frozen pretrained SLT model with lightweight adapter modules
    that learn to compensate for temporal misalignment.

    Args:
        base_model: The frozen pretrained SLT model.
        hidden_dim: Hidden dimension for adapter modules.
        num_heads: Number of attention heads in the visual adapter.
        context_dim: Dimension of the context vector.
    """

    def __init__(
        self,
        base_model: nn.Module,
        hidden_dim: int = 256,
        num_heads: int = 4,
        context_dim: int = 256,
    ):
        super().__init__()
        self.base_model = base_model

        # Freeze all base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Sub-modules (to be implemented)
        # self.context_encoder = ContextEncoder(...)
        # self.visual_adapter = VisualAdapter(...)
        # self.boundary_embedding = BoundaryEmbedding(...)

        raise NotImplementedError(
            "BACA is a stub. Full implementation coming after baseline evaluation."
        )

    def forward(
        self,
        frames: torch.Tensor,
        prev_text: Optional[str] = None,
        start_type: str = "clean",
        end_type: str = "clean",
    ) -> Dict[str, Any]:
        """Forward pass with adapter.

        Args:
            frames: Video frames tensor (B, T, C, H, W).
            prev_text: Previous sentence translation for context (or None).
            start_type: Boundary type for segment start ("clean", "truncated", "contaminated").
            end_type: Boundary type for segment end.

        Returns:
            Dict with model outputs including predicted tokens.
        """
        raise NotImplementedError
