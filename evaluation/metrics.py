"""
Evaluation metrics for sign language translation.

Implements BLEU (1-4) and ROUGE-L computation following PHOENIX14T conventions:
  - All text is lowercased
  - BLEU uses sacrebleu with 13a tokenization
  - ROUGE-L uses the rouge-score package
"""

from typing import Dict, List, Optional

import sacrebleu
from rouge_score import rouge_scorer


def compute_bleu(
    hypotheses: List[str],
    references: List[str],
    lowercase: bool = True,
    tokenize: str = "13a",
) -> Dict[str, float]:
    """Compute BLEU-1 through BLEU-4 scores.

    Args:
        hypotheses: List of predicted translations.
        references: List of reference translations.
        lowercase: Whether to lowercase both hyps and refs.
        tokenize: Tokenization method for sacrebleu.

    Returns:
        Dict with keys "bleu1", "bleu2", "bleu3", "bleu4", each a float in [0, 100].
    """
    if lowercase:
        hypotheses = [h.lower() for h in hypotheses]
        references = [r.lower() for r in references]

    results = {}
    for n in range(1, 5):
        bleu_metric = sacrebleu.BLEU(
            max_ngram_order=n,
            smooth_method="exp",
            tokenize=tokenize,
        )
        bleu = bleu_metric.corpus_score(
            hypotheses,
            [references],  # sacrebleu expects list of reference lists
        )
        results[f"bleu{n}"] = bleu.score

    return results


def compute_rouge(
    hypotheses: List[str],
    references: List[str],
    lowercase: bool = True,
) -> Dict[str, float]:
    """Compute ROUGE-L score.

    Args:
        hypotheses: List of predicted translations.
        references: List of reference translations.
        lowercase: Whether to lowercase both hyps and refs.

    Returns:
        Dict with key "rougeL" containing the F1 score in [0, 1].
    """
    if lowercase:
        hypotheses = [h.lower() for h in hypotheses]
        references = [r.lower() for r in references]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    scores = []
    for hyp, ref in zip(hypotheses, references):
        score = scorer.score(ref, hyp)
        scores.append(score["rougeL"].fmeasure)

    avg_rouge = sum(scores) / len(scores) if scores else 0.0

    return {"rougeL": avg_rouge}


def compute_all_metrics(
    hypotheses: List[str],
    references: List[str],
    lowercase: bool = True,
) -> Dict[str, float]:
    """Compute all translation metrics (BLEU-1..4 + ROUGE-L).

    Args:
        hypotheses: List of predicted translations.
        references: List of reference translations.
        lowercase: Whether to lowercase both hyps and refs.

    Returns:
        Dict with keys: bleu1, bleu2, bleu3, bleu4 (in [0, 100]) and rougeL (in [0, 1]).
    """
    metrics = {}
    metrics.update(compute_bleu(hypotheses, references, lowercase=lowercase))
    metrics.update(compute_rouge(hypotheses, references, lowercase=lowercase))
    return metrics


def compute_degradation(
    clean_metrics: Dict[str, float],
    misaligned_metrics: Dict[str, float],
) -> Dict[str, float]:
    """Compute degradation percentage for each metric.

    degradation = (clean - misaligned) / clean * 100

    Positive values mean the misaligned score is worse (expected).

    Args:
        clean_metrics: Metrics computed on clean (aligned) data.
        misaligned_metrics: Metrics computed on misaligned data.

    Returns:
        Dict with same keys, values = degradation percentage.
    """
    degradation = {}
    for key in clean_metrics:
        clean_val = clean_metrics[key]
        mis_val = misaligned_metrics.get(key, 0.0)
        if clean_val > 0:
            degradation[f"{key}_degradation"] = (
                (clean_val - mis_val) / clean_val * 100.0
            )
        else:
            degradation[f"{key}_degradation"] = 0.0
    return degradation
