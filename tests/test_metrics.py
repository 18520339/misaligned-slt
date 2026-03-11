"""
Unit tests for evaluation metrics (evaluation/metrics.py).
"""

import pytest

from evaluation.metrics import (
    compute_all_metrics,
    compute_bleu,
    compute_degradation,
    compute_rouge,
)


class TestComputeBleu:
    """Tests for compute_bleu()."""

    def test_perfect_match(self):
        """Identical hypothesis and reference should give high BLEU."""
        hyps = ["heute wird es sonnig und warm"]
        refs = ["heute wird es sonnig und warm"]
        scores = compute_bleu(hyps, refs)
        assert scores["bleu4"] > 90.0  # Near perfect (sacrebleu may not give exactly 100)

    def test_no_match(self):
        """Completely different text should give very low BLEU."""
        hyps = ["xyz abc def"]
        refs = ["heute wird es sonnig und warm"]
        scores = compute_bleu(hyps, refs)
        assert scores["bleu4"] < 5.0

    def test_empty_hypothesis(self):
        """Empty hypothesis should give 0 BLEU."""
        hyps = [""]
        refs = ["heute wird es sonnig und warm"]
        scores = compute_bleu(hyps, refs)
        assert scores["bleu4"] == 0.0

    def test_returns_all_four_ngrams(self):
        hyps = ["der regen kommt morgen"]
        refs = ["der regen kommt morgen"]
        scores = compute_bleu(hyps, refs)
        assert "bleu1" in scores
        assert "bleu2" in scores
        assert "bleu3" in scores
        assert "bleu4" in scores

    def test_lowercase_normalization(self):
        """Case should not matter with lowercase=True."""
        hyps = ["HEUTE WIRD ES SONNIG"]
        refs = ["heute wird es sonnig"]
        scores_lower = compute_bleu(hyps, refs, lowercase=True)
        assert scores_lower["bleu4"] > 90.0

    def test_multiple_sentences(self):
        """Should handle corpus-level BLEU with multiple sentences."""
        hyps = ["heute regnet es", "morgen scheint die sonne"]
        refs = ["heute regnet es", "morgen scheint die sonne"]
        scores = compute_bleu(hyps, refs)
        assert scores["bleu4"] > 90.0

    def test_partial_match(self):
        """Partial overlap should give intermediate BLEU."""
        hyps = ["heute wird es kalt"]
        refs = ["heute wird es warm und sonnig"]
        scores = compute_bleu(hyps, refs)
        assert 0 < scores["bleu4"] < 100
        # BLEU-1 should be higher than BLEU-4 for partial matches
        assert scores["bleu1"] >= scores["bleu4"]


class TestComputeRouge:
    """Tests for compute_rouge()."""

    def test_perfect_match(self):
        """Identical texts should give ROUGE-L = 1.0."""
        hyps = ["heute wird es sonnig und warm"]
        refs = ["heute wird es sonnig und warm"]
        scores = compute_rouge(hyps, refs)
        assert abs(scores["rougeL"] - 1.0) < 0.01

    def test_no_match(self):
        """Completely different texts should give low ROUGE-L."""
        hyps = ["xyz abc def"]
        refs = ["heute wird es sonnig und warm"]
        scores = compute_rouge(hyps, refs)
        assert scores["rougeL"] < 0.1

    def test_empty_hypothesis(self):
        """Empty hypothesis should give 0 ROUGE-L."""
        hyps = [""]
        refs = ["heute wird es sonnig und warm"]
        scores = compute_rouge(hyps, refs)
        assert scores["rougeL"] == 0.0

    def test_partial_subsequence(self):
        """Shared subsequence should give intermediate ROUGE-L."""
        hyps = ["heute wird es"]
        refs = ["heute wird es sonnig und warm"]
        scores = compute_rouge(hyps, refs)
        assert 0 < scores["rougeL"] < 1.0


class TestComputeAllMetrics:
    """Tests for compute_all_metrics()."""

    def test_returns_all_keys(self):
        hyps = ["test sentence"]
        refs = ["test sentence"]
        metrics = compute_all_metrics(hyps, refs)
        assert "bleu1" in metrics
        assert "bleu2" in metrics
        assert "bleu3" in metrics
        assert "bleu4" in metrics
        assert "rougeL" in metrics


class TestComputeDegradation:
    """Tests for compute_degradation()."""

    def test_no_degradation(self):
        """Identical metrics should give 0% degradation."""
        clean = {"bleu4": 50.0, "rougeL": 0.7}
        mis = {"bleu4": 50.0, "rougeL": 0.7}
        deg = compute_degradation(clean, mis)
        assert abs(deg["bleu4_degradation"]) < 0.01
        assert abs(deg["rougeL_degradation"]) < 0.01

    def test_full_degradation(self):
        """Zero misaligned metrics should give 100% degradation."""
        clean = {"bleu4": 50.0, "rougeL": 0.7}
        mis = {"bleu4": 0.0, "rougeL": 0.0}
        deg = compute_degradation(clean, mis)
        assert abs(deg["bleu4_degradation"] - 100.0) < 0.01
        assert abs(deg["rougeL_degradation"] - 100.0) < 0.01

    def test_half_degradation(self):
        """Half the score should give 50% degradation."""
        clean = {"bleu4": 40.0}
        mis = {"bleu4": 20.0}
        deg = compute_degradation(clean, mis)
        assert abs(deg["bleu4_degradation"] - 50.0) < 0.01

    def test_zero_clean_score(self):
        """Zero clean score should give 0% degradation (avoid division by zero)."""
        clean = {"bleu4": 0.0}
        mis = {"bleu4": 0.0}
        deg = compute_degradation(clean, mis)
        assert deg["bleu4_degradation"] == 0.0
