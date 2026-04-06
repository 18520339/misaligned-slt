'''Misalignment simulation engine for sign language translation.

Applies temporal misalignment (truncation and contamination) to keypoint sequences,
simulating imperfect segmentation boundaries in streaming scenarios.

Misalignment types:
    delta_s > 0: HEAD TRUNCATION  - segment starts late, missing the beginning
    delta_s < 0: HEAD CONTAMINATION - segment starts early, includes prev sentence tail
    delta_e < 0: TAIL TRUNCATION  - segment ends early, missing the ending
    delta_e > 0: TAIL CONTAMINATION - segment ends late, includes next sentence start
'''
import numpy as np
from typing import Optional, Tuple, Dict, List

MIN_FRAMES_DEFAULT = 8 # Default: DSTA-Net has strides [2,1,1,1,2,1,1,1] -> 4x -> min_frames=8

def compute_min_frames(temporal_strides, min_encoder_len=2):
    '''Compute minimum input frames from encoder architecture.

    DSTA-Net's temporal strides determine how many input frames map to one
    encoder timestep. With strides [2, 1, 1, 1, 2, 1, 1, 1] the total
    downsampling is 4x, so T=8 -> T_enc=2 (minimum for relative position).

    Args:
        temporal_strides: List of stride values from DSTA-Net layer config. Each entry is 
             the 5th element of that layer's config tuple (e.g., [64,64,16,7,2] has stride=2).
        min_encoder_len: Minimum encoder output length (default 2).
    Returns:
        int: minimum input frames.
    '''
    total_downsample = 1
    for s in temporal_strides: total_downsample *= s
    return total_downsample * min_encoder_len


def apply_misalignment(
    keypoints: np.ndarray, delta_s_ratio: float, delta_e_ratio: float,
    prev_keypoints: Optional[np.ndarray] = None,
    next_keypoints: Optional[np.ndarray] = None,
    max_length: Optional[int] = None,
    min_frames: int = MIN_FRAMES_DEFAULT,
) -> Tuple[np.ndarray, Dict]:
    '''Apply temporal misalignment to a keypoint sequence.

    Args:
        keypoints: Shape (T, ...) - keypoint sequence (first dim is time).
        delta_s_ratio: Start offset as fraction of T.
            Positive = head truncation, Negative = head contamination.
        delta_e_ratio: End offset as fraction of T.
            Negative = tail truncation, Positive = tail contamination.
        prev_keypoints: Previous sample keypoints for head contamination.
        next_keypoints: Next sample keypoints for tail contamination.
        max_length: Maximum allowed output length (model constraint).
        min_frames: Minimum frames required after truncation; skip if below.

    Returns:
        (misaligned_keypoints, info_dict) where info_dict contains metadata.
    '''
    T = keypoints.shape[0]
    info = {
        'original_length': T, 'result_length': T,
        'frames_removed': 0, 'frames_added': 0,
        'skipped': False, 'skip_reason': None,
        'head_truncated': 0, 'tail_truncated': 0,
        'head_contaminated': 0, 'tail_contaminated': 0,
        'zero_padded': 0,
    }
    if delta_s_ratio == 0.0 and delta_e_ratio == 0.0: 
        return keypoints.copy(), info # Clean condition - fast path

    # Compute frame offsets from ratios (always based on original length T)
    head_trunc = int(np.floor(delta_s_ratio * T)) if delta_s_ratio > 0 else 0
    head_contam = int(np.floor(abs(delta_s_ratio) * T)) if delta_s_ratio < 0 else 0
    tail_trunc = int(np.floor(abs(delta_e_ratio) * T)) if delta_e_ratio < 0 else 0
    tail_contam = int(np.floor(delta_e_ratio * T)) if delta_e_ratio > 0 else 0

    # Check if truncation leaves enough frames
    remaining = T - head_trunc - tail_trunc
    if remaining < min_frames:
        info['skipped'] = True
        info['skip_reason'] = f'Too few frames ({remaining}) after truncation (need >= {min_frames})'
        return keypoints.copy(), info

    # Step 1: Apply truncation to original segment
    start_idx = head_trunc
    end_idx = T - tail_trunc if tail_trunc > 0 else T
    result = keypoints[start_idx:end_idx]
    info['head_truncated'] = head_trunc
    info['tail_truncated'] = tail_trunc
    info['frames_removed'] = head_trunc + tail_trunc

    # Step 2: Apply head contamination (prepend from previous sample)
    if head_contam > 0:
        contam_frames = _get_contamination_frames(
            prev_keypoints, head_contam, 
            keypoints.shape[1:], keypoints.dtype, source='tail'
        )
        zero_padded = head_contam - (prev_keypoints.shape[0] if prev_keypoints is not None else 0)
        info['zero_padded'] += max(0, zero_padded)
        result = np.concatenate([contam_frames, result], axis=0)
        info['head_contaminated'] = head_contam
        info['frames_added'] += head_contam

    # Step 3: Apply tail contamination (append from next sample)
    if tail_contam > 0:
        contam_frames = _get_contamination_frames(
            next_keypoints, tail_contam, 
            keypoints.shape[1:], keypoints.dtype, source='head'
        )
        zero_padded = tail_contam - (next_keypoints.shape[0] if next_keypoints is not None else 0)
        info['zero_padded'] += max(0, zero_padded)
        result = np.concatenate([result, contam_frames], axis=0)
        info['tail_contaminated'] = tail_contam
        info['frames_added'] += tail_contam

    # Step 4: Max length guard - trim contamination if sequence too long
    if max_length is not None and result.shape[0] > max_length:
        excess = result.shape[0] - max_length
        # Prefer trimming tail contamination first, then head
        if tail_contam > 0 and excess > 0:
            trim = min(excess, info['tail_contaminated'])
            result = result[:result.shape[0] - trim]
            info['tail_contaminated'] -= trim
            info['frames_added'] -= trim
            excess -= trim
        if head_contam > 0 and excess > 0:
            trim = min(excess, info['head_contaminated'])
            result = result[trim:]
            info['head_contaminated'] -= trim
            info['frames_added'] -= trim

    info['result_length'] = result.shape[0]
    return result, info


def _get_contamination_frames(
    source_kp: Optional[np.ndarray], num_frames: int, frame_shape: tuple,
    dtype: np.dtype, source: str,  # 'head' = take first N frames, 'tail' = take last N frames
) -> np.ndarray:
    # Extract contamination frames from an adjacent sample, with zero-pad fallback
    if source_kp is None: return np.zeros((num_frames,) + frame_shape, dtype=dtype)
    available = source_kp.shape[0]
    take = min(num_frames, available)
    frames = source_kp[-take:] if source == 'tail' else source_kp[:take]

    if take < num_frames:
        pad_shape = (num_frames - take,) + frame_shape
        pad = np.zeros(pad_shape, dtype=dtype)
        if source == 'tail': frames = np.concatenate([pad, frames], axis=0)
        else: frames = np.concatenate([frames, pad], axis=0)
    return frames


def generate_conditions(severity_levels: List[float], include_compound: bool = True) -> List[Tuple[str, float, float]]:
    conditions = [('clean', 0.0, 0.0)]
    for s in severity_levels: # Basic conditions: one offset active, other zero
        pct = int(s * 100)
        conditions.append((f'HT_{pct:02d}', s, 0.0))
        conditions.append((f'TT_{pct:02d}', 0.0, -s))
        conditions.append((f'HC_{pct:02d}', -s, 0.0))
        conditions.append((f'TC_{pct:02d}', 0.0, s))

    if include_compound:
        for s1 in severity_levels:
            for s2 in severity_levels:
                p1, p2 = int(s1 * 100), int(s2 * 100)
                conditions.append((f'HT_{p1:02d}+TT_{p2:02d}', s1, -s2))
                conditions.append((f'HC_{p1:02d}+TC_{p2:02d}', -s1, s2))
                conditions.append((f'HT_{p1:02d}+TC_{p2:02d}', s1, s2))
                conditions.append((f'HC_{p1:02d}+TT_{p2:02d}', -s1, -s2))
    return conditions # List of (name, delta_s_ratio, delta_e_ratio) tuples


def condition_count(k: int, include_compound: bool = True) -> int: 
    return 1 + 4 * k + (4 * k**2 if include_compound else 0) # N_total = 1 + 4k + 4k^2 (or 1 + 4k if basic only)


def parse_condition_name(name: str) -> Dict: # Parse a condition name into its components
    if name == 'clean': return {'type': 'clean', 'components': [], 'severities': []}
    parts = name.split('+')
    components, severities = [], []
    
    for p in parts:
        ctype, sev = p.split('_')
        components.append(ctype)
        severities.append(int(sev) / 100.0)
    return {
        'type': '+'.join(components), 'components': components,
        'severities': severities, 'is_compound': len(parts) > 1,
    }