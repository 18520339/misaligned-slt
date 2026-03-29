"""Knee point detection using the Kneedle algorithm.

Identifies the severity level at which degradation accelerates —
the point beyond which performance drops sharply.
"""
import numpy as np
from typing import List, Tuple, Optional
from kneed import KneeLocator


def detect_knee_point(severities: List[float], values: List[float], direction: str = 'decreasing') -> Optional[dict]:
    """Detect the knee point in a degradation curve.

    Args:
        severities: X-axis values (severity percentages).
        values: Y-axis values (metric values at each severity).
        direction: 'decreasing' for BLEU-like (lower = worse),
                   'increasing' for WER-like (higher = worse).
        curve: 'convex' or 'concave'.
        sensitivity: Kneedle sensitivity parameter.

    Returns:
        Dict with knee severity, value at knee, and related info, or None.
    """
    if len(severities) < 3: return None
    kl = KneeLocator(severities, values, curve='convex', direction=direction, S=1.0, online=True)
    if kl.knee is None: return _fallback_knee_detection(severities, values, direction)
    knee_idx = severities.index(kl.knee) if kl.knee in severities else None
    return {'knee_severity': kl.knee, 'knee_value': kl.knee_y}


def _fallback_knee_detection(severities, values, direction):
    # Simple fallback: find maximum second derivative (curvature)
    if len(severities) < 3: return None
    x, y = np.array(severities), np.array(values)
    dy2 = np.diff(y, n=2) # Second derivative (finite differences)
    
    if direction == 'decreasing': # Look for most negative second derivative (steepest drop)
        idx = np.argmin(dy2)
    else: # Look for most positive second derivative (steepest rise)
        idx = np.argmax(dy2)
        
    knee_idx = idx + 1  # offset due to double differencing
    return {'knee_severity': severities[knee_idx], 'knee_value': values[knee_idx]}
    

def compute_degradation_rate(severities: List[float], values: List[float]) -> float:
    # Compute linear degradation rate (slope) via least-squares fit
    if len(severities) < 2: return 0.0
    x = np.array(severities)
    y = np.array(values)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def detect_all_knee_points(results_json: dict, severity_levels: list) -> dict:
    """Detect knee points for all basic condition types.

    Args:
        results_json: Full results JSON from evaluator.
        severity_levels: List of severity values used.

    Returns:
        Dict mapping condition_type -> {metric -> knee_info}
    """
    basic_types = ['HT', 'TT', 'HC', 'TC']
    knee_results = {}

    for ctype in basic_types:
        bleu_vals, wer_vals = [], []
        valid_sevs = []
        for sev in severity_levels:
            cond_name = f'{ctype}_{int(sev * 100):02d}'
            if cond_name in results_json:
                m = results_json[cond_name].get('metrics', {})
                if 'bleu4' in m:
                    bleu_vals.append(m['bleu4'])
                    wer_vals.append(m.get('wer', 0))
                    valid_sevs.append(sev)

        if not valid_sevs: continue
        knee_results[ctype] = {}
        bleu_knee = detect_knee_point(valid_sevs, bleu_vals, direction='decreasing')
        if bleu_knee: bleu_knee['degradation_rate'] = compute_degradation_rate(valid_sevs, bleu_vals)
        knee_results[ctype]['bleu4'] = bleu_knee
    return knee_results