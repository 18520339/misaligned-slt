'''Failure mode classification for misaligned SLT predictions.

Classifies each (sample, condition) prediction into one of:
  - Acceptable:       sentence-BLEU >= 0.4
  - Under-generation: output_length_ratio < 0.5 AND sentence-BLEU < 0.4
  - Hallucination:    novel_token_rate > 0.5 AND sentence-BLEU < 0.2
  - Partial match:    0.1 <= sentence-BLEU < 0.4 AND 0.5 <= length_ratio <= 1.5
  - Repetition:       any token repeated >= 3 times consecutively
  - Incoherent:       everything else with sentence-BLEU < 0.1
'''
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter

FAILURE_MODES = [
    'Acceptable', 'Under-generation', 'Hallucination',
    'Partial match', 'Repetition', 'Incoherent'
]
FAILURE_COLORS = {
    'Acceptable': '#2ecc71', 'Under-generation': '#3498db', 'Hallucination': '#e74c3c',
    'Partial match': '#f1c40f', 'Repetition': '#9b59b6', 'Incoherent': '#7f8c8d'
}

def classify_failure_mode(
    sentence_bleu: float, novel_token_rate: float,
    output_length_ratio: float, has_repetition: bool,
) -> str:# Classify a single prediction into a failure mode
    if sentence_bleu >= 0.4: return 'Acceptable'
    if has_repetition: return 'Repetition'
    if novel_token_rate > 0.5 and sentence_bleu < 0.2: return 'Hallucination'
    if output_length_ratio < 0.5 and sentence_bleu < 0.4: return 'Under-generation'
    if 0.1 <= sentence_bleu < 0.4 and 0.5 <= output_length_ratio <= 1.5: return 'Partial match'
    if sentence_bleu < 0.1: return 'Incoherent'
    return 'Partial match'  # fallback for edge cases


def classify_all_predictions(results_json: dict) -> dict:
    '''Classify all predictions across all conditions.

    Args:
        results_json: The full results JSON from evaluator.py

    Returns:
        Dict mapping condition_name -> {sample_name -> failure_mode}
    '''
    classifications = {}
    for cond_name, cond_data in results_json.items():
        if cond_name == 'meta': continue
        metrics = cond_data.get('metrics', {})
        per_sample = metrics.get('per_sample', {})
        cond_class = {}
        for sample_name, sm in per_sample.items():
            mode = classify_failure_mode(
                sentence_bleu=sm['sentence_bleu'],
                novel_token_rate=sm['novel_token_rate'],
                output_length_ratio=sm['output_length_ratio'],
                has_repetition=sm['has_repetition'],
            )
            cond_class[sample_name] = mode
        classifications[cond_name] = cond_class
    return classifications


def failure_mode_distribution(classifications: dict) -> dict: # Compute failure mode distribution per condition
    distributions = {}
    for cond_name, sample_modes in classifications.items():
        counts = Counter(sample_modes.values())
        total = len(sample_modes)
        dist = {mode: counts.get(mode, 0) / max(total, 1) * 100 for mode in FAILURE_MODES}
        dist['_total'] = total
        distributions[cond_name] = dist
    return distributions # Dict mapping condition_name -> {mode: count}


def failure_mode_transitions(classifications: dict, condition_type: str, severity_levels: list) -> dict:
    '''Track how samples transition between failure modes as severity increases.

    Args:
        classifications: Output of classify_all_predictions
        condition_type: One of 'HT', 'TT', 'HC', 'TC'
        severity_levels: List of severity values

    Returns:
        Dict mapping severity -> {mode: count}
    '''
    transitions = {}
    for sev in severity_levels:
        pct = int(sev * 100)
        cond_name = f'{condition_type}_{pct:02d}'
        if cond_name in classifications:
            counts = Counter(classifications[cond_name].values())
            transitions[sev] = {mode: counts.get(mode, 0) for mode in FAILURE_MODES}
    return transitions


def compute_transition_matrix(classifications: dict, from_cond: str, to_cond: str) -> Tuple[Optional[np.ndarray], int]:
    '''Compute per-sample failure-mode transition matrix between two conditions.

    Rows   = failure mode at `from_cond` (e.g. 'clean').
    Columns = failure mode at `to_cond`  (e.g. 'HT_30').
    Cell [i, j] = number of samples that were in mode i at from_cond and transitioned to mode j at to_cond.
    Only samples present in BOTH conditions contribute.

    Returns:
        (matrix, n_common) where matrix is shape (n_modes, n_modes) int ndarray,
        and n_common is the number of samples tracked.  Returns (None, 0) if
        either condition has no classifications.
    '''
    from_classes = classifications.get(from_cond, {})
    to_classes   = classifications.get(to_cond,   {})
    common = set(from_classes.keys()) & set(to_classes.keys())
    if not common: return None, 0

    mode_to_idx = {m: i for i, m in enumerate(FAILURE_MODES)}
    n = len(FAILURE_MODES)
    matrix = np.zeros((n, n), dtype=int)
    for sample in common:
        i = mode_to_idx.get(from_classes[sample], n - 1)
        j = mode_to_idx.get(to_classes[sample],   n - 1)
        matrix[i, j] += 1
    return matrix, len(common)