"""
BACA adapter training loop (stub).

Trains only the adapter weights while keeping the base SLT model frozen.

Training procedure:
  1. Freeze ALL base model weights
  2. Initialize gate α = 0 (identity at start)
  3. Train on mixture of clean + misaligned pairs
  4. Loss: standard cross-entropy on full translation Y_n
  5. Robustness: inject noisy context to handle error propagation

TODO: Full implementation after baseline evaluation demonstrates the problem.
"""

from typing import Any, Dict, Optional


class AdapterTrainer:
    """Trainer for BACA adapter modules.

    Args:
        model: BACA-wrapped model.
        optimizer: Optimizer for adapter parameters only.
        scheduler: Learning rate scheduler.
        config: Training configuration dict.
    """

    def __init__(self, model: Any, optimizer: Any, scheduler: Any, config: Dict):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

    def train_epoch(self, dataloader: Any, epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dict with loss and metric values.
        """
        raise NotImplementedError("Adapter training not yet implemented.")

    def validate(self, dataloader: Any) -> Dict[str, float]:
        """Run validation.

        Returns:
            Dict with validation metrics.
        """
        raise NotImplementedError
