'''Knee point detection using a clean-baseline threshold rule.

Defines the knee as the first severity where the metric crosses a
user-defined fraction of the clean baseline.
'''
import numpy as np
from typing import List, Optional


def detect_knee_point(
    severities: List[float], values: List[float], direction: str = 'decreasing',
    clean_baseline: Optional[float] = None, retention_ratio: float = 0.70,
) -> Optional[dict]:
    '''Detect the knee point via threshold crossing from clean baseline.

    Args:
        severities: X-axis values (severity percentages).
        values: Y-axis values (metric values at each severity).
        direction: 'decreasing' for BLEU-like (lower = worse),
                   'increasing' for WER-like (higher = worse).
        clean_baseline: Clean-condition metric value. If None, uses values[0].
        retention_ratio: Fraction of clean baseline used as threshold. Example: 0.70 means "70% of clean baseline".

    Returns:
        Dict with knee severity and value at knee, or None.
    '''
    if not severities or not values or len(severities) != len(values): return None
    if not (0 < retention_ratio <= 1.0): raise ValueError('retention_ratio must be in (0, 1].')

    baseline = values[0] if clean_baseline is None else clean_baseline
    if baseline is None: return None

    if direction == 'decreasing':
        threshold = baseline * retention_ratio
        crossed_idx = next((i for i, v in enumerate(values) if v <= threshold), None)
    elif direction == 'increasing':
        # Symmetric counterpart for "worse-is-higher" metrics.
        threshold = baseline * (2.0 - retention_ratio)
        crossed_idx = next((i for i, v in enumerate(values) if v >= threshold), None)
    else: raise ValueError("direction must be 'decreasing' or 'increasing'.")

    knee_idx = crossed_idx if crossed_idx is not None else len(values) - 1
    return {
        'knee_severity': severities[knee_idx],
        'knee_value': values[knee_idx],
        'clean_baseline': baseline,
        'threshold': threshold,
        'retention_ratio': retention_ratio,
    }


def compute_degradation_rate(severities: List[float], values: List[float]) -> float:
    # Compute linear degradation rate (slope) via least-squares fit
    if len(severities) < 2: return 0.0
    x = np.array(severities)
    y = np.array(values)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def detect_all_knee_points(results_json: dict, severity_levels: list) -> dict:
    '''Detect knee points for all basic condition types.

    Args:
        results_json: Full results JSON from evaluator.
        severity_levels: List of severity values used.

    Returns:
        Dict mapping condition_type -> {metric -> knee_info}
    '''
    basic_types = ['HT', 'TT', 'HC', 'TC']
    knee_results = {}
    clean_bleu = results_json.get('clean', {}).get('metrics', {}).get('bleu4')

    for ctype in basic_types:
        bleu_vals = []
        valid_sevs = []
        for sev in severity_levels:
            cond_name = f'{ctype}_{int(sev * 100):02d}'
            if cond_name in results_json:
                m = results_json[cond_name].get('metrics', {})
                if 'bleu4' in m:
                    bleu_vals.append(m['bleu4'])
                    valid_sevs.append(sev)

        if not valid_sevs: continue
        knee_results[ctype] = {}
        bleu_knee = detect_knee_point(
            valid_sevs, bleu_vals,
            direction='decreasing',
            clean_baseline=clean_bleu,
            retention_ratio=0.70,
        )
        if bleu_knee: bleu_knee['degradation_rate'] = compute_degradation_rate(valid_sevs, bleu_vals)
        knee_results[ctype]['bleu4'] = bleu_knee
    return knee_results