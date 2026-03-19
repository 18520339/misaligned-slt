'''Temporal misalignment simulation for sign language translation evaluation.

Simulates real-world segmentation errors by truncating or contaminating
keypoint sequences at the head (start) and/or tail (end).

Misalignment types:
  - Head truncation  (delta_s > 0): segment starts LATE, missing the beginning
  - Head contamination (delta_s < 0): segment starts EARLY, includes prev sentence
  - Tail truncation  (delta_e < 0): segment ends EARLY, missing the ending
  - Tail contamination (delta_e > 0): segment ends LATE, includes next sentence
  
Condition catalogue (121 total):
  - 1 clean baseline
  - 4 single-sided types × 5 severity levels                 =  20
  - 4 compound types × 5 head severities × 5 tail severities = 100
'''
import re
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
SINGLE_CONDITIONS   = ['head_trunc', 'tail_trunc', 'head_contam', 'tail_contam']
COMPOUND_CONDITIONS = [
    'head_trunc_tail_trunc', 'head_trunc_tail_contam',
    'head_contam_tail_trunc', 'head_contam_tail_contam',
]
SEVERITY_LEVELS = [10, 20, 30, 40, 50]


def compute_frame_counts(T, delta_s_pct, delta_e_pct):
    '''Compute frame count metadata without transforming a tensor.

    This is the single source of truth for the clamping / min-frames logic.
    apply_misalignment delegates to this function rather than duplicating it.

    Args:
        T: int, original temporal length.
        delta_s_pct: float, start offset as fraction of T.
        delta_e_pct: float, end offset as fraction of T.

    Returns:
        dict with keys: T_original, T_after, head_trunc_frames,
        tail_trunc_frames, head_contam_frames, tail_contam_frames.
    '''
    if delta_s_pct == 0 and delta_e_pct == 0: return {
        'T_original': T, 'T_after': T,
        'head_trunc_frames': 0, 'tail_trunc_frames': 0,
        'head_contam_frames': 0, 'tail_contam_frames': 0,
    }
    # Minimum frames to keep after truncation (10 % of T, at least 4 for the
    # model's stride-2 convolutions which need length divisible by 4).
    min_frames = min(T, max(4, round(0.1 * T)))

    # Compute absolute frame counts (based on original T)
    head_trunc  = round(delta_s_pct * T)       if delta_s_pct > 0 else 0
    tail_trunc  = round(abs(delta_e_pct) * T)  if delta_e_pct < 0 else 0
    head_contam = round(abs(delta_s_pct) * T)  if delta_s_pct < 0 else 0
    tail_contam = round(delta_e_pct * T)       if delta_e_pct > 0 else 0

    # Clamp combined truncation so at least min_frames remain
    total_trunc = head_trunc + tail_trunc
    max_trunc = T - min_frames
    if total_trunc > max_trunc and total_trunc > 0:
        scale      = max_trunc / total_trunc
        head_trunc = int(head_trunc * scale) # int() floors — avoids overshooting
        tail_trunc = int(tail_trunc * scale)
        # Final guard: scaling/truncation could still leave < min_frames for very short T
        while T - head_trunc - tail_trunc < min_frames:
            if head_trunc >= tail_trunc and head_trunc > 0: head_trunc -= 1
            elif tail_trunc > 0: tail_trunc -= 1
            else: break

    T_after = T - head_trunc - tail_trunc + head_contam + tail_contam
    if T_after < 4: T_after = 4 # MSKA needs length >= 4
    return {
        'T_original': T, 'T_after': T_after,
        'head_trunc_frames': head_trunc, 'tail_trunc_frames': tail_trunc,
        'head_contam_frames': head_contam, 'tail_contam_frames': tail_contam,
    }


def apply_misalignment(keypoint, prev_keypoint, next_keypoint, delta_s_pct, delta_e_pct, return_meta=False):
    '''Apply temporal misalignment to a keypoint sequence.

    Args:
        keypoint: tensor (C, T, V) — the original keypoint sequence.
        prev_keypoint: tensor (C, T_prev, V) or None — previous sample (used for head contamination; its LAST frames are prepended).
        next_keypoint: tensor (C, T_next, V) or None — next sample (used for tail contamination; its FIRST frames are appended).
        delta_s_pct: float, start offset as fraction of T. >0 → head truncation, <0 → head contamination.
        delta_e_pct: float, end offset as fraction of T. <0 → tail truncation, >0 → tail contamination.
        return_meta: bool, if True return (result, meta_dict) where meta_dict
            contains frame count information matching compute_frame_counts.
    Returns:
        Modified keypoint tensor (C, T', V).
        If return_meta=True, returns (tensor, meta_dict) instead.
    '''
    C, T, V = keypoint.shape
    device = keypoint.device
    dtype = keypoint.dtype
    
    if delta_s_pct == 0 and delta_e_pct == 0:
        if return_meta:
            return keypoint, compute_frame_counts(T, 0.0, 0.0)
        return keypoint
    
    # Delegate ALL clamping logic to compute_frame_counts (single source of truth)
    frame_counts = compute_frame_counts(T, delta_s_pct, delta_e_pct)
    head_trunc   = frame_counts['head_trunc_frames']
    tail_trunc   = frame_counts['tail_trunc_frames']
    head_contam  = frame_counts['head_contam_frames']
    tail_contam  = frame_counts['tail_contam_frames']

    # Apply truncation
    result = keypoint[:, head_trunc: T - tail_trunc if tail_trunc else T, :]

    # Apply head contamination (prepend frames from previous sample)
    if head_contam > 0:
        if prev_keypoint is not None and prev_keypoint.shape[1] > 0:
            T_p = prev_keypoint.shape[1]
            if T_p >= head_contam: prefix = prev_keypoint[:, -head_contam:, :]
            else: # Not enough prev frames — pad remainder
                pad = torch.zeros(C, head_contam - T_p, V, device=device, dtype=dtype)
                prefix = torch.cat([pad, prev_keypoint], dim=1)
        else: prefix = torch.zeros(C, head_contam, V, device=device, dtype=dtype)
        result = torch.cat([prefix, result], dim=1)

    # Apply tail contamination (append frames from next sample)
    if tail_contam > 0:
        if next_keypoint is not None and next_keypoint.shape[1] > 0:
            T_n = next_keypoint.shape[1]
            if T_n >= tail_contam: suffix = next_keypoint[:, :tail_contam, :]
            else: # Not enough next frames — pad remainder
                pad = torch.zeros(C, tail_contam - T_n, V, device=device, dtype=dtype)
                suffix = torch.cat([next_keypoint, pad], dim=1)
        else: suffix = torch.zeros(C, tail_contam, V, device=device, dtype=dtype)
        result = torch.cat([result, suffix], dim=1)
        
    # Safety: MSKA needs length >= 4 (two stride-2 conv layers)
    if result.shape[1] < 4:
        pad_n = 4 - result.shape[1]
        last = result[:, -1:, :].expand(C, pad_n, V)
        result = torch.cat([result, last], dim=1)
        
    if return_meta:
        # Override T_after with the actual result shape (may differ from estimate
        # in compute_frame_counts if the safety-pad was triggered)
        return result, {**frame_counts, 'T_after': result.shape[1]}
    return result


def get_condition_deltas(condition_name, severity_pct): # Return (delta_s_pct, delta_e_pct) for a condition and severity
    if condition_name == 'clean': return 0.0, 0.0
    s_sign, e_sign = CONDITION_TYPES[condition_name]
    frac = severity_pct / 100.0
    return s_sign * frac, e_sign * frac


def get_all_conditions():
    '''Generate all 121 benchmark conditions.

    121 = 1 clean
        + 4 single-sided types x 5 severity levels      (20 conditions)
        + 4 compound types x 5 head sevs x 5 tail sevs (100 conditions)

    For compound types every ordered (head_sev, tail_sev) pair is included,
    so both symmetric (h==t) and asymmetric (h!=t) cases are covered.

    Returns:
        list of 6-tuples: (label, cond_name, head_sev, tail_sev, delta_s_pct, delta_e_pct)
        where
            label     -- unique string key used in results JSON
            cond_name -- key into CONDITION_TYPES
            head_sev  -- head-side severity percentage (0 if inactive)
            tail_sev  -- tail-side severity percentage (0 if inactive)
    '''
    conditions = [('clean', 'clean', 0, 0, 0.0, 0.0)]

    # Single-sided: only one dimension is active
    for name in SINGLE_CONDITIONS:
        s_sign, e_sign = CONDITION_TYPES[name]
        for sev in SEVERITY_LEVELS:
            frac     = sev / 100.0
            head_sev = sev if s_sign != 0 else 0
            tail_sev = sev if e_sign != 0 else 0
            conditions.append((
                f'{name}_{sev}', name, head_sev, tail_sev, 
                s_sign * frac, e_sign * frac
            ))

    # Compound: full head_sev x tail_sev grid (each side varies independently)
    for name in COMPOUND_CONDITIONS:
        s_sign, e_sign = CONDITION_TYPES[name]
        for head_sev in SEVERITY_LEVELS:
            for tail_sev in SEVERITY_LEVELS:
                conditions.append((
                    f'{name}_h{head_sev}_t{tail_sev}', name, head_sev, tail_sev, 
                    s_sign * (head_sev / 100.0), e_sign * (tail_sev / 100.0)
                ))
    return conditions


def sanity_check(T=100, C=3, V=133): # Print expected output lengths and run assertions to verify logic
    kp      = torch.randn(C, T, V)
    prev_kp = torch.randn(C, 80, V)   # shorter prev to test padding
    next_kp = torch.randn(C, 120, V)  # longer next

    conditions = get_all_conditions()
    print(f'Total conditions: {len(conditions)} (expected: 1 + {len(SINGLE_CONDITIONS)}x5 + '
          f'{len(COMPOUND_CONDITIONS)}x5x5 = {1 + len(SINGLE_CONDITIONS) * 5 + len(COMPOUND_CONDITIONS) * 25})')
    assert len(conditions) == 121, f'Expected 121, got {len(conditions)}'

    # Show first 5 + a compound sample
    header = f"{'Label':<35} {'hs':>4} {'δs':>4} {'δs':>6} {'de':>6} {'T_out':>6}"
    print(header)
    print('-' * len(header))
    # to_show = conditions[:5] + [c for c in conditions if 'h10_t30' in c[0]][:1]
    for label, cond_name, hs, ts, ds, de in conditions: # in to_show
        out = apply_misalignment(kp, prev_kp, next_kp, ds, de)
        print(f'{label:<35} {hs:>3}% {ts:>3}% {ds:>+6.2f} {de:>+6.2f} {out.shape[1]:>6}')

    # Test return_meta consistency with compute_frame_counts
    print('\n--- return_meta test ---')
    out, meta = apply_misalignment(kp, prev_kp, next_kp, 0.3, -0.2, return_meta=True)
    frame_counts = compute_frame_counts(T, 0.3, -0.2)
    assert meta['T_original'] == frame_counts['T_original']
    assert meta['T_after']    == frame_counts['T_after'] == out.shape[1]
    print(f'meta={meta}  OK')

    # Test edge case: 50%+50% truncation on very short sequence
    kp_short = torch.randn(C, 7, V)
    out_short = apply_misalignment(kp_short, None, None, 0.5, -0.5)
    assert out_short.shape[1] >= 4, f'Short seq output too small: {out_short.shape[1]}'
    print(f'Short-seq 50% + 50% trunc: T=7 -> T_out={out_short.shape[1]} (>=4)  OK')
    print('\nAll sanity checks passed.')


if __name__ == '__main__':
    sanity_check()