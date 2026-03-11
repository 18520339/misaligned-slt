"""
Core misalignment simulation logic.

Implements parameterized temporal misalignment (truncation and contamination)
for evaluating sign language translation models under segmentation errors.

Misalignment is defined by two continuous offsets applied to the original
clean segment [0, T]:
  - delta_s (start offset): positive = head truncation, negative = head contamination
  - delta_e (end offset): negative = tail truncation, positive = tail contamination
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math


@dataclass
class MisalignmentConfig:
    """Configuration for misalignment simulation.

    Attributes:
        max_ratio: Maximum offset as fraction of sentence duration T.
        max_frames: Absolute maximum offset in frames.
        severity_levels: List of severity ratios for evaluation mode.
    """

    max_ratio: float = 0.25
    max_frames: int = 50
    severity_levels: List[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.15, 0.20, 0.25]
    )


def compute_max_offset(T: int, severity: float, max_ratio: float, max_frames: int) -> int:
    """Compute the maximum offset magnitude for a given severity and sentence length.

    The offset is capped by both the severity ratio * T and the absolute max_frames.
    Additionally, we enforce that max_ratio * T is respected as an upper bound.

    Args:
        T: Number of frames in the original segment.
        severity: Severity ratio (e.g., 0.10 for 10%).
        max_ratio: Maximum allowed ratio of T for the offset.
        max_frames: Absolute maximum offset in frames.

    Returns:
        Maximum offset in frames (integer, >= 0).
    """
    # The offset at this severity is severity * T, but capped by max_ratio * T and max_frames
    offset = severity * T
    cap = min(max_ratio * T, max_frames)
    return max(0, int(min(offset, cap)))


def get_condition_name(delta_s: int, delta_e: int) -> str:
    """Get a human-readable name for a misalignment condition.

    Args:
        delta_s: Start offset in frames.
        delta_e: End offset in frames.

    Returns:
        Human-readable condition name, e.g., "head_trunc+tail_contam".
    """
    if delta_s == 0 and delta_e == 0:
        return "clean"

    parts = []

    # Head component (based on sign of delta_s)
    if delta_s > 0:
        parts.append("head_trunc")
    elif delta_s < 0:
        parts.append("head_contam")

    # Tail component (based on sign of delta_e)
    if delta_e < 0:
        parts.append("tail_trunc")
    elif delta_e > 0:
        parts.append("tail_contam")

    return "+".join(parts) if parts else "clean"


def get_condition_type(delta_s: int, delta_e: int) -> Tuple[str, str]:
    """Get the type of misalignment for start and end separately.

    Args:
        delta_s: Start offset in frames.
        delta_e: End offset in frames.

    Returns:
        Tuple of (start_type, end_type) where each is one of
        "clean", "truncated", "contaminated".
    """
    if delta_s > 0:
        start_type = "truncated"
    elif delta_s < 0:
        start_type = "contaminated"
    else:
        start_type = "clean"

    if delta_e < 0:
        end_type = "truncated"
    elif delta_e > 0:
        end_type = "contaminated"
    else:
        end_type = "clean"

    return start_type, end_type


def generate_eval_conditions(
    T: int,
    severity_levels: Optional[List[float]] = None,
    max_ratio: float = 0.25,
    max_frames: int = 50,
) -> List[Tuple[int, int, str, float]]:
    """Generate all evaluation conditions for a segment of length T.

    Produces the clean baseline plus all 9 sign combinations of (delta_s, delta_e)
    at each severity level, excluding combinations where both offsets would be zero.

    The 9 sign combinations are:
        sign(delta_s) in {-, 0, +} × sign(delta_e) in {-, 0, +}

    For non-zero sign, the magnitude equals compute_max_offset(T, severity, ...).

    Args:
        T: Number of frames in the original segment.
        severity_levels: List of severity ratios. Defaults to [0.05..0.25].
        max_ratio: Maximum offset ratio.
        max_frames: Absolute maximum offset in frames.

    Returns:
        List of (delta_s, delta_e, condition_name, severity) tuples.
        The first entry is always the clean baseline.
    """
    if severity_levels is None:
        severity_levels = [0.05, 0.10, 0.15, 0.20, 0.25]

    conditions = []

    # Clean baseline
    conditions.append((0, 0, "clean", 0.0))

    # For each severity level, generate all 9 sign combinations
    # sign(delta_s) in {-1, 0, +1} × sign(delta_e) in {-1, 0, +1}
    sign_combos = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

    for severity in severity_levels:
        offset = compute_max_offset(T, severity, max_ratio, max_frames)
        if offset == 0:
            # For very short sequences, skip if offset rounds to 0
            continue

        for s_sign, e_sign in sign_combos:
            ds = s_sign * offset
            de = e_sign * offset

            # Skip the (0, 0) case — that's the clean baseline already included
            if ds == 0 and de == 0:
                continue

            name = get_condition_name(ds, de)
            conditions.append((ds, de, name, severity))

    return conditions


def apply_misalignment(
    num_frames: int,
    delta_s: int,
    delta_e: int,
    prev_num_frames: Optional[int] = None,
    next_num_frames: Optional[int] = None,
) -> Tuple[List[Tuple[str, int]], int, int]:
    """Compute the frame index mapping for a misaligned segment.

    Instead of copying frames, this returns a list of (source, index) tuples
    describing which frames to load. Sources are:
      - "current": frame from the current sample
      - "prev": frame from the previous sample
      - "next": frame from the next sample
      - "black": zero-filled black frame (used when no adjacent sample exists)

    Args:
        num_frames: Number of frames in the current (clean) segment.
        delta_s: Start offset. Positive = head truncation, negative = head contamination.
        delta_e: End offset. Negative = tail truncation, positive = tail contamination.
        prev_num_frames: Number of frames in the previous sample (None if first sample).
        next_num_frames: Number of frames in the next sample (None if last sample).

    Returns:
        Tuple of:
          - frame_map: List of (source, frame_index) tuples.
          - effective_start: Effective start index in the current segment (for diagnostics).
          - effective_end: Effective end index in the current segment (for diagnostics).
    """
    frame_map = []

    # --- Head handling ---
    if delta_s < 0:
        # Head contamination: prepend |delta_s| frames from previous sample
        contam_frames = abs(delta_s)
        if prev_num_frames is not None and prev_num_frames > 0:
            # Take the last |delta_s| frames from the previous sample
            start_in_prev = max(0, prev_num_frames - contam_frames)
            for i in range(start_in_prev, prev_num_frames):
                frame_map.append(("prev", i))
        else:
            # No previous sample — use black frames
            for _ in range(contam_frames):
                frame_map.append(("black", 0))
        effective_start = 0  # All current frames are kept
    elif delta_s > 0:
        # Head truncation: skip first delta_s frames of current segment
        effective_start = min(delta_s, num_frames)
    else:
        effective_start = 0

    # --- Current segment frames ---
    if delta_e < 0:
        # Tail truncation: end early
        effective_end = max(0, num_frames + delta_e)  # delta_e is negative
    elif delta_e > 0:
        effective_end = num_frames  # All current frames kept
    else:
        effective_end = num_frames

    # Add current segment frames
    for i in range(effective_start, effective_end):
        frame_map.append(("current", i))

    # --- Tail handling ---
    if delta_e > 0:
        # Tail contamination: append delta_e frames from next sample
        contam_frames = delta_e
        if next_num_frames is not None and next_num_frames > 0:
            # Take the first delta_e frames from the next sample
            end_in_next = min(contam_frames, next_num_frames)
            for i in range(end_in_next):
                frame_map.append(("next", i))
        else:
            # No next sample — use black frames
            for _ in range(contam_frames):
                frame_map.append(("black", 0))

    return frame_map, effective_start, effective_end
