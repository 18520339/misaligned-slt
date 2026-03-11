"""
Boundary Embedding (Sub-module 2c).

Learnable embeddings prepended/appended to the visual feature sequence
to signal whether segment boundaries are clean, truncated, or contaminated.

4 learnable embedding vectors:
  - clean_start, truncated_start (prepended)
  - clean_end, truncated_end (appended)

During training: selected based on known misalignment type.
During inference: selected based on streaming segmenter's confidence.

TODO: Full implementation after baseline evaluation.
"""

import torch
import torch.nn as nn


class BoundaryEmbedding(nn.Module):
    """Learnable boundary indicator embeddings.

    Args:
        embed_dim: Dimension matching visual features.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()

        # Start boundary embeddings
        self.clean_start = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.truncated_start = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.contaminated_start = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # End boundary embeddings
        self.clean_end = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.truncated_end = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.contaminated_end = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def forward(
        self,
        visual_features: torch.Tensor,
        start_type: str = "clean",
        end_type: str = "clean",
    ) -> torch.Tensor:
        """Prepend/append boundary embeddings to visual features.

        Args:
            visual_features: (B, T, embed_dim).
            start_type: "clean", "truncated", or "contaminated".
            end_type: "clean", "truncated", or "contaminated".

        Returns:
            Visual features with boundary embeddings: (B, T+2, embed_dim).
        """
        raise NotImplementedError
