"""
Uni-Sign inference wrapper (stub).

Uni-Sign uses pose keypoints as input rather than raw RGB frames.
Misalignment simulation operates on pose sequences.

Repository: https://github.com/ZechengLi19/Uni-Sign
Weights: https://huggingface.co/ZechengLi19/Uni-Sign
Paper: ICLR 2025 (current state-of-the-art)

TODO: Implement after setting up Uni-Sign and extracting pose keypoints.
"""

from typing import List

import torch


class UniSignWrapper:
    """Wrapper for Uni-Sign model inference (stub).

    Uni-Sign differs from GFSLT-VLP in that it uses pose keypoints
    instead of raw RGB frames as visual input. The misalignment
    simulation will operate on the pose sequences directly.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        max_length: int = 100,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.max_length = max_length
        self.model = None

    def load_model(self):
        """Load Uni-Sign model and weights."""
        raise NotImplementedError(
            "Uni-Sign wrapper not yet implemented. "
            "Run 'bash baselines/setup_unisign.sh' and implement this wrapper."
        )

    def predict(self, pose_sequence: torch.Tensor) -> str:
        """Run inference on a pose keypoint sequence.

        Args:
            pose_sequence: Tensor of pose keypoints (T, num_keypoints, dims).

        Returns:
            Predicted German text translation.
        """
        raise NotImplementedError

    def predict_batch(self, pose_sequences: List[torch.Tensor]) -> List[str]:
        """Run batch inference on pose sequences."""
        raise NotImplementedError
