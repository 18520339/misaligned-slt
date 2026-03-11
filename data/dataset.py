"""
PyTorch Datasets for PHOENIX14T with misalignment support.

Provides three dataset classes:
  - PhoenixDataset: Base PHOENIX14T loader (frames + text labels)
  - PhoenixMisalignEvalDataset: Fixed misalignment conditions for benchmarking
  - PhoenixMisalignTrainDataset: Random on-the-fly misalignment augmentation
"""

import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.misalign import (
    MisalignmentConfig,
    apply_misalignment,
    compute_max_offset,
    generate_eval_conditions,
    get_condition_name,
    get_condition_type,
)
from data.utils import (
    count_frames_in_dir,
    create_black_frame,
    get_frame_dir,
    get_phoenix_annotations,
    load_frame_image,
    load_frame_paths,
    preprocess_text,
)


class PhoenixDataset(Dataset):
    """Base PHOENIX14T dataset.

    Loads frame paths and text labels from the standard PHOENIX14T structure.
    Frames are loaded lazily to save memory.

    Args:
        data_root: Root directory of the PHOENIX14T dataset.
        split: Dataset split ("train", "dev", "test").
        transform: Optional transform for frame images.
        frame_dir_pattern: Pattern for frame directory paths.
        max_frames_per_sample: Maximum number of frames to load per sample (None = all).
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        transform: Optional[Callable] = None,
        frame_dir_pattern: str = "features/fullFrame-210x260px/{split}/{sample_id}",
        max_frames_per_sample: Optional[int] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.frame_dir_pattern = frame_dir_pattern
        self.max_frames_per_sample = max_frames_per_sample

        # Load annotations
        self.annotations = get_phoenix_annotations(split, data_root)

        # Precompute frame directories and counts
        self.frame_dirs = []
        self.frame_counts = []
        for ann in self.annotations:
            fdir = get_frame_dir(data_root, split, ann["folder"], frame_dir_pattern)
            self.frame_dirs.append(fdir)
            self.frame_counts.append(count_frames_in_dir(fdir))

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ann = self.annotations[idx]
        frame_dir = self.frame_dirs[idx]
        text = preprocess_text(ann["text"])

        # Load frame paths (lazily — not all frames loaded into memory)
        frame_paths = load_frame_paths(frame_dir)
        if self.max_frames_per_sample is not None:
            frame_paths = frame_paths[: self.max_frames_per_sample]

        # Load frames
        frames = [load_frame_image(p, self.transform) for p in frame_paths]

        # Stack into tensor if transforms produce tensors
        if frames and isinstance(frames[0], torch.Tensor):
            frames = torch.stack(frames, dim=0)

        return {
            "frames": frames,
            "text": text,
            "gloss": ann.get("gloss", ""),
            "sample_id": ann["id"],
            "num_frames": len(frame_paths),
            "idx": idx,
        }

    def get_frame_count(self, idx: int) -> int:
        """Get the number of frames for a sample without loading them."""
        return self.frame_counts[idx]

    def get_text(self, idx: int) -> str:
        """Get the preprocessed text label for a sample."""
        return preprocess_text(self.annotations[idx]["text"])


class PhoenixMisalignEvalDataset(Dataset):
    """PHOENIX14T dataset with fixed misalignment conditions for evaluation.

    For each sample, generates all specified misalignment conditions.
    Each __getitem__ returns one (sample, condition) pair.

    The total length is len(base_dataset) * num_conditions (46 by default).

    Args:
        base_dataset: PhoenixDataset instance.
        config: MisalignmentConfig with evaluation parameters.
        conditions: Optional list of specific conditions to evaluate.
                    If None, generates all 46 conditions (clean + 9×5 severity levels).
    """

    def __init__(
        self,
        base_dataset: PhoenixDataset,
        config: Optional[MisalignmentConfig] = None,
        conditions: Optional[List[Tuple[int, int, str, float]]] = None,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.config = config or MisalignmentConfig()

        # We generate conditions per-sample since T varies, but we also
        # need a fixed set for consistent benchmarking. We'll use a
        # representative T to define the condition types, then adapt
        # magnitudes per sample.
        self.condition_specs = self._build_condition_specs()
        self._total_len = len(self.base_dataset) * len(self.condition_specs)

    def _build_condition_specs(self) -> List[Tuple[int, int, str, float]]:
        """Build the list of (sign_s, sign_e, name, severity) condition specs.

        Returns the 46 condition types. Actual offsets are computed per-sample
        using the sample's T.
        """
        specs = []
        # Clean baseline
        specs.append((0, 0, "clean", 0.0))

        sign_combos = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 0), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ]

        for severity in self.config.severity_levels:
            for s_sign, e_sign in sign_combos:
                if s_sign == 0 and e_sign == 0:
                    continue
                name = get_condition_name(s_sign, e_sign)
                specs.append((s_sign, e_sign, name, severity))

        return specs

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Map flat index to (sample_idx, condition_idx)
        num_conditions = len(self.condition_specs)
        sample_idx = idx // num_conditions
        condition_idx = idx % num_conditions

        s_sign, e_sign, cond_name, severity = self.condition_specs[condition_idx]

        # Get base sample info
        ann = self.base_dataset.annotations[sample_idx]
        frame_dir = self.base_dataset.frame_dirs[sample_idx]
        text = preprocess_text(ann["text"])
        T = self.base_dataset.frame_counts[sample_idx]

        # Compute actual offsets for this sample's T
        if severity > 0:
            offset = compute_max_offset(
                T, severity, self.config.max_ratio, self.config.max_frames
            )
        else:
            offset = 0

        delta_s = s_sign * offset
        delta_e = e_sign * offset

        # Update condition name with actual offset values for this sample
        actual_cond_name = get_condition_name(delta_s, delta_e)

        # Get adjacent sample info for contamination
        prev_num_frames = None
        prev_frame_dir = None
        prev_text = ""
        if sample_idx > 0:
            prev_num_frames = self.base_dataset.frame_counts[sample_idx - 1]
            prev_frame_dir = self.base_dataset.frame_dirs[sample_idx - 1]
            prev_text = preprocess_text(
                self.base_dataset.annotations[sample_idx - 1]["text"]
            )

        next_num_frames = None
        next_frame_dir = None
        if sample_idx < len(self.base_dataset) - 1:
            next_num_frames = self.base_dataset.frame_counts[sample_idx + 1]
            next_frame_dir = self.base_dataset.frame_dirs[sample_idx + 1]

        # Compute frame mapping
        frame_map, eff_start, eff_end = apply_misalignment(
            T, delta_s, delta_e, prev_num_frames, next_num_frames
        )

        # Load frames according to the mapping
        current_paths = load_frame_paths(frame_dir) if T > 0 else []
        prev_paths = (
            load_frame_paths(prev_frame_dir)
            if prev_frame_dir and os.path.exists(prev_frame_dir)
            else []
        )
        next_paths = (
            load_frame_paths(next_frame_dir)
            if next_frame_dir and os.path.exists(next_frame_dir)
            else []
        )

        frames = []
        for source, fidx in frame_map:
            if source == "current" and fidx < len(current_paths):
                frames.append(
                    load_frame_image(current_paths[fidx], self.base_dataset.transform)
                )
            elif source == "prev" and fidx < len(prev_paths):
                frames.append(
                    load_frame_image(prev_paths[fidx], self.base_dataset.transform)
                )
            elif source == "next" and fidx < len(next_paths):
                frames.append(
                    load_frame_image(next_paths[fidx], self.base_dataset.transform)
                )
            else:
                # Black frame
                black = create_black_frame()
                if self.base_dataset.transform is not None:
                    black = self.base_dataset.transform(black)
                frames.append(black)

        # Stack into tensor if frames are tensors
        if frames and isinstance(frames[0], torch.Tensor):
            frames = torch.stack(frames, dim=0)

        start_type, end_type = get_condition_type(delta_s, delta_e)

        return {
            "frames": frames,
            "text": text,  # Full original translation (ground truth)
            "prev_text": prev_text,  # Context from previous sentence
            "sample_id": ann["id"],
            "condition_name": actual_cond_name,
            "severity": severity,
            "delta_s": delta_s,
            "delta_e": delta_e,
            "start_type": start_type,
            "end_type": end_type,
            "num_frames_original": T,
            "num_frames_misaligned": len(frame_map),
            "idx": sample_idx,
        }


class PhoenixMisalignTrainDataset(Dataset):
    """PHOENIX14T dataset with random on-the-fly misalignment augmentation.

    With probability p_aug, applies random misalignment to each sample.
    With probability (1 - p_aug), returns the clean sample unchanged.

    Also provides the previous sample's text for context-conditioned training.

    Args:
        base_dataset: PhoenixDataset instance.
        config: MisalignmentConfig with augmentation parameters.
        p_aug: Probability of applying misalignment augmentation.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        base_dataset: PhoenixDataset,
        config: Optional[MisalignmentConfig] = None,
        p_aug: float = 0.5,
        seed: int = 42,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.config = config or MisalignmentConfig()
        self.p_aug = p_aug
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ann = self.base_dataset.annotations[idx]
        frame_dir = self.base_dataset.frame_dirs[idx]
        text = preprocess_text(ann["text"])
        T = self.base_dataset.frame_counts[idx]

        # Get previous sample text for context
        prev_text = ""
        if idx > 0:
            prev_text = preprocess_text(self.base_dataset.annotations[idx - 1]["text"])

        # Decide whether to apply misalignment
        apply_aug = self.rng.random() < self.p_aug

        if apply_aug and T > 0:
            # Sample random offsets
            max_offset = compute_max_offset(
                T, self.config.max_ratio, self.config.max_ratio, self.config.max_frames
            )

            delta_s = self.rng.randint(-max_offset, max_offset)
            delta_e = self.rng.randint(-max_offset, max_offset)

            # Get adjacent sample info
            prev_num_frames = (
                self.base_dataset.frame_counts[idx - 1] if idx > 0 else None
            )
            next_num_frames = (
                self.base_dataset.frame_counts[idx + 1]
                if idx < len(self.base_dataset) - 1
                else None
            )

            # Compute frame mapping
            frame_map, eff_start, eff_end = apply_misalignment(
                T, delta_s, delta_e, prev_num_frames, next_num_frames
            )

            # Load frames
            current_paths = load_frame_paths(frame_dir)
            prev_frame_dir = (
                self.base_dataset.frame_dirs[idx - 1] if idx > 0 else None
            )
            next_frame_dir = (
                self.base_dataset.frame_dirs[idx + 1]
                if idx < len(self.base_dataset) - 1
                else None
            )
            prev_paths = (
                load_frame_paths(prev_frame_dir)
                if prev_frame_dir and os.path.exists(prev_frame_dir)
                else []
            )
            next_paths = (
                load_frame_paths(next_frame_dir)
                if next_frame_dir and os.path.exists(next_frame_dir)
                else []
            )

            frames = []
            for source, fidx in frame_map:
                if source == "current" and fidx < len(current_paths):
                    frames.append(
                        load_frame_image(
                            current_paths[fidx], self.base_dataset.transform
                        )
                    )
                elif source == "prev" and fidx < len(prev_paths):
                    frames.append(
                        load_frame_image(
                            prev_paths[fidx], self.base_dataset.transform
                        )
                    )
                elif source == "next" and fidx < len(next_paths):
                    frames.append(
                        load_frame_image(
                            next_paths[fidx], self.base_dataset.transform
                        )
                    )
                else:
                    black = create_black_frame()
                    if self.base_dataset.transform is not None:
                        black = self.base_dataset.transform(black)
                    frames.append(black)

            condition_name = get_condition_name(delta_s, delta_e)
            start_type, end_type = get_condition_type(delta_s, delta_e)
        else:
            # Clean sample — no misalignment
            delta_s = 0
            delta_e = 0
            condition_name = "clean"
            start_type = "clean"
            end_type = "clean"

            frame_paths = load_frame_paths(frame_dir)
            frames = [
                load_frame_image(p, self.base_dataset.transform) for p in frame_paths
            ]

        # Stack into tensor if frames are tensors
        if frames and isinstance(frames[0], torch.Tensor):
            frames = torch.stack(frames, dim=0)

        return {
            "frames": frames,
            "text": text,  # Full original translation (ground truth)
            "prev_text": prev_text,  # Context Y_{n-1}
            "sample_id": ann["id"],
            "condition_name": condition_name,
            "delta_s": delta_s,
            "delta_e": delta_e,
            "start_type": start_type,
            "end_type": end_type,
            "num_frames": len(frames) if isinstance(frames, list) else frames.shape[0],
            "idx": idx,
        }
