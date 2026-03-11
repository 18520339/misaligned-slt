"""
Sample output extraction and formatting for qualitative review.

Extracts side-by-side examples of clean vs misaligned translations
for human inspection and paper presentation.
"""

import csv
import os
from typing import Any, Dict, List, Optional


def extract_samples(
    references: List[str],
    clean_hypotheses: List[str],
    misaligned_hypotheses: List[str],
    sample_ids: List[str],
    condition_name: str,
    severity: float,
    n_samples: int = 5,
    selection: str = "worst",
) -> List[Dict[str, str]]:
    """Extract sample translations for qualitative comparison.

    Args:
        references: Reference translations.
        clean_hypotheses: Model outputs on clean data.
        misaligned_hypotheses: Model outputs on misaligned data.
        sample_ids: Sample identifiers.
        condition_name: Misalignment condition name.
        severity: Severity level.
        n_samples: Number of samples to extract.
        selection: Selection strategy:
            "worst" - samples with largest quality degradation
            "best" - samples with smallest degradation
            "random" - random selection

    Returns:
        List of dicts with sample info and translations.
    """
    from evaluation.metrics import compute_bleu

    # Compute per-sample degradation (sentence-level BLEU is approximate)
    degradations = []
    for i in range(len(references)):
        # Sentence-level BLEU (sacrebleu on single sentence)
        clean_bleu = compute_bleu([clean_hypotheses[i]], [references[i]])["bleu4"]
        mis_bleu = compute_bleu(
            [misaligned_hypotheses[i]], [references[i]]
        )["bleu4"]
        degradations.append(clean_bleu - mis_bleu)

    # Select indices
    if selection == "worst":
        # Largest degradation (clean much better than misaligned)
        indices = sorted(range(len(degradations)), key=lambda i: -degradations[i])
    elif selection == "best":
        # Smallest degradation
        indices = sorted(range(len(degradations)), key=lambda i: degradations[i])
    else:
        import random

        indices = random.sample(range(len(references)), min(n_samples, len(references)))

    selected = indices[:n_samples]

    samples = []
    for i in selected:
        samples.append(
            {
                "sample_id": sample_ids[i],
                "condition": condition_name,
                "severity": f"{severity:.0%}",
                "reference": references[i],
                "clean_output": clean_hypotheses[i],
                "misaligned_output": misaligned_hypotheses[i],
                "degradation": f"{degradations[i]:.1f}",
            }
        )

    return samples


def save_samples_text(
    samples: List[Dict[str, str]],
    output_path: str,
) -> str:
    """Save sample translations as a formatted text file.

    Args:
        samples: List of sample dicts from extract_samples().
        output_path: Path to save the text file.

    Returns:
        Path to the saved file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(samples, 1):
            f.write(f"{'='*80}\n")
            f.write(f"Sample {i}: {sample['sample_id']}\n")
            f.write(f"Condition: {sample['condition']} | Severity: {sample['severity']}\n")
            f.write(f"BLEU-4 Degradation: {sample['degradation']}\n")
            f.write(f"{'-'*80}\n")
            f.write(f"REFERENCE:  {sample['reference']}\n")
            f.write(f"CLEAN:      {sample['clean_output']}\n")
            f.write(f"MISALIGNED: {sample['misaligned_output']}\n")
            f.write(f"{'='*80}\n\n")

    return output_path


def save_samples_csv(
    samples: List[Dict[str, str]],
    output_path: str,
) -> str:
    """Save sample translations as CSV.

    Args:
        samples: List of sample dicts from extract_samples().
        output_path: Path to save the CSV file.

    Returns:
        Path to the saved file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not samples:
        return output_path

    fieldnames = list(samples[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)

    return output_path
