"""
Failure analysis for SLT model outputs under misalignment.

Classifies each translation output into failure categories:
  - Hallucination: output contains content not supported by visible video
  - Under-generation: output significantly shorter than reference
  - Incoherent: output is grammatically broken or nonsensical
  - Acceptable: output captures the core meaning despite misalignment

Uses rule-based heuristics (length ratio, BLEU threshold).
"""

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

from evaluation.metrics import compute_bleu


# Failure categories
HALLUCINATION = "hallucination"
UNDER_GENERATION = "under_generation"
INCOHERENT = "incoherent"
ACCEPTABLE = "acceptable"

ALL_FAILURE_TYPES = [HALLUCINATION, UNDER_GENERATION, INCOHERENT, ACCEPTABLE]


def classify_output(
    hypothesis: str,
    reference: str,
    clean_hypothesis: Optional[str] = None,
    undergen_ratio: float = 0.5,
    bleu_acceptable_threshold: float = 0.15,
) -> str:
    """Classify a single translation output into a failure category.

    Heuristic classification rules (applied in order):
    1. Under-generation: output length < undergen_ratio × reference length
    2. Hallucination: if clean output exists, check if misaligned output
       introduces content not in clean or reference (approximated by
       significant length increase with very low BLEU against reference)
    3. Incoherent: very low BLEU (< 5) against reference
    4. Acceptable: everything else (BLEU >= threshold)

    Args:
        hypothesis: The model's predicted translation.
        reference: The ground-truth reference translation.
        clean_hypothesis: The model's output on clean (aligned) input, if available.
        undergen_ratio: Length ratio threshold for under-generation.
        bleu_acceptable_threshold: BLEU-4 threshold (in sacrebleu scale 0-100)
                                    for acceptable classification.

    Returns:
        One of: "hallucination", "under_generation", "incoherent", "acceptable".
    """
    hyp = hypothesis.strip().lower()
    ref = reference.strip().lower()

    hyp_words = hyp.split()
    ref_words = ref.split()

    # Empty hypothesis
    if len(hyp_words) == 0:
        return UNDER_GENERATION

    # 1. Under-generation check
    if len(ref_words) > 0 and len(hyp_words) / len(ref_words) < undergen_ratio:
        return UNDER_GENERATION

    # Compute sentence-level BLEU
    bleu_scores = compute_bleu([hyp], [ref])
    bleu4 = bleu_scores.get("bleu4", 0.0)

    # 2. Hallucination check
    # Significant length increase over reference with low BLEU suggests hallucination
    if len(ref_words) > 0 and len(hyp_words) / len(ref_words) > 1.5 and bleu4 < 10.0:
        return HALLUCINATION

    # If clean output is available, check if misaligned output diverged significantly
    if clean_hypothesis is not None:
        clean_words = clean_hypothesis.strip().lower().split()
        # If misaligned output is much longer than clean and very low BLEU with ref
        if (
            len(clean_words) > 0
            and len(hyp_words) / len(clean_words) > 1.5
            and bleu4 < 10.0
        ):
            return HALLUCINATION

    # 3. Incoherent check - very low BLEU
    if bleu4 < 5.0:
        return INCOHERENT

    # 4. Acceptable
    if bleu4 >= bleu_acceptable_threshold:
        return ACCEPTABLE

    # Default: incoherent for low but not zero BLEU
    return INCOHERENT


def analyze_failures(
    hypotheses: List[str],
    references: List[str],
    clean_hypotheses: Optional[List[str]] = None,
    condition_name: str = "unknown",
    severity: float = 0.0,
    undergen_ratio: float = 0.5,
    bleu_acceptable_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Analyze failure types for a set of predictions.

    Args:
        hypotheses: List of model predictions.
        references: List of reference translations.
        clean_hypotheses: Optional list of clean predictions for hallucination detection.
        condition_name: Name of the misalignment condition.
        severity: Severity level.
        undergen_ratio: Length ratio threshold for under-generation.
        bleu_acceptable_threshold: BLEU threshold for acceptable.

    Returns:
        Dict with condition info and counts of each failure type.
    """
    counts = {ft: 0 for ft in ALL_FAILURE_TYPES}
    classifications = []

    for i, (hyp, ref) in enumerate(zip(hypotheses, references)):
        clean_hyp = clean_hypotheses[i] if clean_hypotheses else None
        category = classify_output(
            hyp, ref, clean_hyp, undergen_ratio, bleu_acceptable_threshold
        )
        counts[category] += 1
        classifications.append(category)

    return {
        "condition": condition_name,
        "severity": severity,
        "total": len(hypotheses),
        **counts,
        "classifications": classifications,
    }


def save_failure_analysis(
    results: List[Dict[str, Any]],
    output_path: str,
) -> str:
    """Save failure analysis results to CSV.

    Args:
        results: List of failure analysis dicts from analyze_failures().
        output_path: Path to save the CSV.

    Returns:
        Path to the saved file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "condition",
        "severity",
        "total",
        *ALL_FAILURE_TYPES,
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {k: result[k] for k in fieldnames}
            writer.writerow(row)

    return output_path
