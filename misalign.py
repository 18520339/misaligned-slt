"""Temporal misalignment simulation for sign language translation evaluation.

Simulates real-world segmentation errors by truncating or contaminating
keypoint sequences at the head (start) and/or tail (end).

Misalignment types:
  - Head truncation  (delta_s > 0): segment starts LATE, missing the beginning
  - Head contamination (delta_s < 0): segment starts EARLY, includes prev sentence
  - Tail truncation  (delta_e < 0): segment ends EARLY, missing the ending
  - Tail contamination (delta_e > 0): segment ends LATE, includes next sentence
"""

import torch
import numpy as np


# 9 condition types: (delta_s_sign, delta_e_sign)
# Sign convention: positive delta_s = head truncation, negative = head contamination
#                  negative delta_e = tail truncation, positive = tail contamination
CONDITION_TYPES = {
    'clean':                    (0,  0),
    'head_trunc':               (1,  0),
    'tail_trunc':               (0, -1),
    'head_contam':              (-1, 0),
    'tail_contam':              (0,  1),
    'head_trunc_tail_trunc':    (1, -1),
    'head_trunc_tail_contam':   (1,  1),
    'head_contam_tail_trunc':   (-1,-1),
    'head_contam_tail_contam':  (-1, 1),
}

SEVERITY_LEVELS = [10, 20, 30, 40, 50]


def apply_misalignment(keypoint, prev_keypoint, next_keypoint,
                        delta_s_pct, delta_e_pct):
    """Apply temporal misalignment to a keypoint sequence.

    Args:
        keypoint: tensor (C, T, V) — the original keypoint sequence.
        prev_keypoint: tensor (C, T_prev, V) or None — previous sample
            (used for head contamination; its LAST frames are prepended).
        next_keypoint: tensor (C, T_next, V) or None — next sample
            (used for tail contamination; its FIRST frames are appended).
        delta_s_pct: float, start offset as fraction of T.
            >0 → head truncation, <0 → head contamination.
        delta_e_pct: float, end offset as fraction of T.
            <0 → tail truncation, >0 → tail contamination.

    Returns:
        Modified keypoint tensor (C, T', V).
    """
    if delta_s_pct == 0 and delta_e_pct == 0:
        return keypoint

    C, T, V = keypoint.shape
    device = keypoint.device
    dtype = keypoint.dtype

    # Minimum frames to keep after truncation (10 % of T, at least 4 for the
    # model's stride-2 convolutions which need length divisible by 4).
    min_frames = max(4, round(0.1 * T))

    # --- compute absolute frame counts (based on *original* T) -----------
    head_trunc  = round(delta_s_pct * T) if delta_s_pct > 0 else 0
    tail_trunc  = round(abs(delta_e_pct) * T) if delta_e_pct < 0 else 0
    head_contam = round(abs(delta_s_pct) * T) if delta_s_pct < 0 else 0
    tail_contam = round(delta_e_pct * T) if delta_e_pct > 0 else 0

    # --- clamp combined truncation so at least min_frames remain ----------
    total_trunc = head_trunc + tail_trunc
    max_trunc = T - min_frames
    if total_trunc > max_trunc and total_trunc > 0:
        scale = max_trunc / total_trunc
        # int() floors toward zero — avoids round() overshooting past min_frames
        head_trunc = int(head_trunc * scale)
        tail_trunc = int(tail_trunc * scale)
        # Final guard: rounding could still leave < min_frames in edge cases
        while T - head_trunc - tail_trunc < min_frames:
            if head_trunc >= tail_trunc and head_trunc > 0:
                head_trunc -= 1
            elif tail_trunc > 0:
                tail_trunc -= 1
            else:
                break

    # --- apply truncation -------------------------------------------------
    start = head_trunc
    end = T - tail_trunc
    result = keypoint[:, start:end, :]

    # --- apply head contamination (prepend) --------------------------------
    if head_contam > 0:
        if prev_keypoint is not None:
            T_p = prev_keypoint.shape[1]
            if T_p >= head_contam:
                prefix = prev_keypoint[:, -head_contam:, :]
            else:
                pad = torch.zeros(C, head_contam - T_p, V,
                                  device=device, dtype=dtype)
                prefix = torch.cat([pad, prev_keypoint], dim=1)
        else:
            prefix = torch.zeros(C, head_contam, V, device=device, dtype=dtype)
        result = torch.cat([prefix, result], dim=1)

    # --- apply tail contamination (append) ---------------------------------
    if tail_contam > 0:
        if next_keypoint is not None:
            T_n = next_keypoint.shape[1]
            if T_n >= tail_contam:
                suffix = next_keypoint[:, :tail_contam, :]
            else:
                pad = torch.zeros(C, tail_contam - T_n, V,
                                  device=device, dtype=dtype)
                suffix = torch.cat([next_keypoint, pad], dim=1)
        else:
            suffix = torch.zeros(C, tail_contam, V, device=device, dtype=dtype)
        result = torch.cat([result, suffix], dim=1)

    # --- safety: MSKA needs length >= 4 (two stride-2 conv layers) --------
    if result.shape[1] < 4:
        pad_n = 4 - result.shape[1]
        last = result[:, -1:, :].expand(C, pad_n, V)
        result = torch.cat([result, last], dim=1)

    return result


def get_condition_deltas(condition_name, severity_pct):
    """Return (delta_s_pct, delta_e_pct) for a condition and severity."""
    if condition_name == 'clean':
        return 0.0, 0.0
    s_sign, e_sign = CONDITION_TYPES[condition_name]
    frac = severity_pct / 100.0
    return s_sign * frac, e_sign * frac


def get_all_conditions():
    """Generate the full list of 41 benchmark conditions.

    Returns:
        list of (condition_name, severity_pct, delta_s_pct, delta_e_pct)
    """
    conditions = [('clean', 0, 0.0, 0.0)]
    for name, (s_sign, e_sign) in CONDITION_TYPES.items():
        if name == 'clean':
            continue
        for sev in SEVERITY_LEVELS:
            frac = sev / 100.0
            conditions.append((name, sev, s_sign * frac, e_sign * frac))
    return conditions


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
def sanity_check(T=100, C=3, V=133):
    """Print expected output lengths for every condition to verify logic."""
    kp = torch.randn(C, T, V)
    prev_kp = torch.randn(C, 80, V)   # shorter prev to test padding
    next_kp = torch.randn(C, 120, V)  # longer next

    print(f"Original T={T}, prev T=80, next T=120")
    print(f"{'Condition':<32} {'Sev':>4} {'ds':>6} {'de':>6} {'T_out':>6} {'chg':>6}")
    print("-" * 68)
    for name, sev, ds, de in get_all_conditions():
        out = apply_misalignment(kp, prev_kp, next_kp, ds, de)
        label = name if sev == 0 else f"{name}@{sev}%"
        chg = out.shape[1] - T
        print(f"{label:<32} {sev:>3}% {ds:>+6.2f} {de:>+6.2f} "
              f"{out.shape[1]:>6} {chg:>+6}")


if __name__ == '__main__':
    sanity_check()
