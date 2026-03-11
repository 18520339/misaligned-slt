"""
Complete benchmark orchestrator.

This is the main entry point for running the full misalignment benchmark.
It ties together:
  1. Loading model and data
  2. Running inference across all 46 conditions
  3. Computing metrics per condition
  4. Saving structured CSV results
  5. Generating degradation plots and heatmaps
  6. Running failure analysis
  7. Extracting sample translations

Usage:
    python -m evaluation.benchmark \
        --model gfslt_vlp \
        --gfslt_dir baselines/GFSLT-VLP \
        --checkpoint baselines/GFSLT-VLP/out/Gloss-Free/best_checkpoint.pth \
        --data_root data/phoenix14t \
        --split test \
        --output_dir results
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from data.misalign import MisalignmentConfig
from evaluation.metrics import compute_all_metrics, compute_degradation


# All 8 non-clean condition types
ALL_CONDITION_TYPES = [
    "head_trunc",
    "tail_trunc",
    "head_contam",
    "tail_contam",
    "head_trunc+tail_trunc",
    "head_trunc+tail_contam",
    "head_contam+tail_trunc",
    "head_contam+tail_contam",
]


def run_benchmark_gfslt(
    gfslt_dir: str,
    checkpoint_path: str,
    data_root: str,
    split: str = "test",
    output_dir: str = "results",
    batch_size: int = 4,
    device: str = "cuda",
    severity_levels: Optional[List[float]] = None,
) -> str:
    """Run the full benchmark for GFSLT-VLP.

    Args:
        gfslt_dir: Path to GFSLT-VLP repo.
        checkpoint_path: Path to trained checkpoint.
        data_root: PHOENIX14T data root.
        split: Dataset split.
        output_dir: Output directory for results.
        batch_size: Inference batch size.
        device: Device string.
        severity_levels: List of severity ratios to evaluate.

    Returns:
        Path to the results CSV.
    """
    from baselines.run_gfslt_vlp import (
        load_gfslt_model,
        load_misaligned_data,
        run_inference,
        save_predictions,
    )

    device = torch.device(device)
    config = MisalignmentConfig()
    if severity_levels:
        config.severity_levels = severity_levels

    # Directories
    tables_dir = os.path.join(output_dir, "tables")
    predictions_dir = os.path.join(output_dir, "predictions", "gfslt_vlp")
    plots_dir = os.path.join(output_dir, "plots")
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    # Load model
    print("=" * 70)
    print("  MISALIGNMENT BENCHMARK — GFSLT-VLP")
    print("=" * 70)
    model, tokenizer, gfslt_config = load_gfslt_model(gfslt_dir, checkpoint_path, device)

    # Build condition list: clean + 8 types × N severity levels
    conditions = [("clean", 0.0)]
    for severity in config.severity_levels:
        for cond_type in ALL_CONDITION_TYPES:
            conditions.append((cond_type, severity))

    print(f"\nTotal conditions to evaluate: {len(conditions)}")
    print(f"Severity levels: {config.severity_levels}")
    print()

    # Run each condition
    results = []
    clean_metrics = None
    clean_predictions = None
    clean_references = None

    for cond_idx, (cond_name, severity) in enumerate(conditions):
        print(f"\n[{cond_idx+1}/{len(conditions)}] {cond_name} @ {severity:.0%}")
        print("-" * 50)

        # Load data
        start_time = time.time()
        videos, references, sample_ids = load_misaligned_data(
            data_root, split, gfslt_dir,
            condition_name=cond_name, severity=severity,
        )
        load_time = time.time() - start_time

        # Run inference
        start_time = time.time()
        predictions = run_inference(
            model, tokenizer, videos, device,
            batch_size=batch_size,
        )
        inference_time = time.time() - start_time

        # Lowercase predictions for evaluation
        predictions_lower = [p.lower() for p in predictions]
        references_lower = [r.lower() for r in references]

        # Save predictions
        save_predictions(
            predictions, references, sample_ids,
            predictions_dir, cond_name, severity,
        )

        # Compute metrics
        metrics = compute_all_metrics(predictions_lower, references_lower)

        # Compute degradation
        if cond_name == "clean":
            clean_metrics = metrics.copy()
            clean_predictions = predictions_lower.copy()
            clean_references = references_lower.copy()
            degradation = {f"{k}_degradation": 0.0 for k in metrics}
        elif clean_metrics is not None:
            degradation = compute_degradation(clean_metrics, metrics)
        else:
            degradation = {}

        result_row = {
            "model": "gfslt_vlp",
            "condition": cond_name,
            "severity": severity,
            "num_samples": len(sample_ids),
            "load_time_s": round(load_time, 1),
            "inference_time_s": round(inference_time, 1),
            **metrics,
            **degradation,
        }
        results.append(result_row)

        print(f"  BLEU-4: {metrics['bleu4']:.2f}  ROUGE-L: {metrics['rougeL']:.4f}")
        if cond_name != "clean" and clean_metrics:
            print(f"  Degradation: {degradation.get('bleu4_degradation', 0):.1f}%")

    # Save all results to CSV
    csv_path = os.path.join(tables_dir, "gfslt_vlp_results.csv")
    _save_results_csv(results, csv_path)
    print(f"\n{'='*70}")
    print(f"Results saved to: {csv_path}")

    # Generate plots
    print("\nGenerating plots...")
    try:
        from evaluation.visualize import plot_degradation_curves, plot_heatmap

        plot_degradation_curves(csv_path, plots_dir, model_name="GFSLT-VLP")
        for sev in config.severity_levels:
            plot_heatmap(csv_path, plots_dir, severity=sev, model_name="GFSLT-VLP")
        print(f"Plots saved to: {plots_dir}")
    except Exception as e:
        print(f"Plot generation failed: {e}")

    # Failure analysis
    print("\nRunning failure analysis...")
    try:
        from evaluation.failure_analysis import analyze_failures, save_failure_analysis

        failure_results = []
        for cond_idx, (cond_name, severity) in enumerate(conditions):
            if cond_name == "clean":
                continue

            # Load prediction files
            sev_str = f"{int(severity * 100):02d}"
            hyp_file = os.path.join(predictions_dir, f"{cond_name}_{sev_str}_hyp.txt")
            if not os.path.exists(hyp_file):
                continue

            with open(hyp_file, 'r', encoding='utf-8') as f:
                hyps = [line.strip() for line in f]

            ref_file = os.path.join(predictions_dir, f"{cond_name}_{sev_str}_ref.txt")
            with open(ref_file, 'r', encoding='utf-8') as f:
                refs = [line.strip() for line in f]

            fa_result = analyze_failures(
                hyps, refs,
                clean_hypotheses=clean_predictions,
                condition_name=cond_name,
                severity=severity,
            )
            failure_results.append(fa_result)

        if failure_results:
            failure_csv = os.path.join(tables_dir, "gfslt_vlp_failures.csv")
            save_failure_analysis(failure_results, failure_csv)
            print(f"Failure analysis saved to: {failure_csv}")

            from evaluation.visualize import plot_failure_distribution
            plot_failure_distribution(failure_csv, plots_dir, model_name="GFSLT-VLP")
    except Exception as e:
        print(f"Failure analysis failed: {e}")

    # Sample translations at 20% severity
    print("\nExtracting sample translations...")
    try:
        from evaluation.sample_outputs import (
            extract_samples, save_samples_text, save_samples_csv,
        )

        all_samples = []
        target_severity = 0.20
        for cond_type in ALL_CONDITION_TYPES:
            sev_str = f"{int(target_severity * 100):02d}"
            hyp_file = os.path.join(predictions_dir, f"{cond_type}_{sev_str}_hyp.txt")
            ref_file = os.path.join(predictions_dir, f"{cond_type}_{sev_str}_ref.txt")
            ids_file = os.path.join(predictions_dir, f"{cond_type}_{sev_str}_ids.txt")

            if not all(os.path.exists(f) for f in [hyp_file, ref_file, ids_file]):
                continue

            with open(hyp_file, 'r', encoding='utf-8') as f:
                hyps = [line.strip() for line in f]
            with open(ref_file, 'r', encoding='utf-8') as f:
                refs = [line.strip() for line in f]
            with open(ids_file, 'r', encoding='utf-8') as f:
                ids = [line.strip() for line in f]

            # Load clean predictions
            clean_hyp_file = os.path.join(predictions_dir, "clean_00_hyp.txt")
            with open(clean_hyp_file, 'r', encoding='utf-8') as f:
                clean_hyps = [line.strip() for line in f]

            samples = extract_samples(
                refs, clean_hyps[:len(refs)], hyps, ids,
                condition_name=cond_type,
                severity=target_severity,
                n_samples=5,
                selection="worst",
            )
            all_samples.extend(samples)

        if all_samples:
            save_samples_text(
                all_samples,
                os.path.join(samples_dir, "gfslt_vlp_samples.txt"),
            )
            save_samples_csv(
                all_samples,
                os.path.join(samples_dir, "gfslt_vlp_samples.csv"),
            )
            print(f"Sample translations saved to: {samples_dir}")
    except Exception as e:
        print(f"Sample extraction failed: {e}")

    print(f"\n{'='*70}")
    print("  BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"\nAll outputs in: {output_dir}/")
    print(f"  Tables:  {tables_dir}/")
    print(f"  Plots:   {plots_dir}/")
    print(f"  Samples: {samples_dir}/")

    return csv_path


def _save_results_csv(results: List[Dict[str, Any]], csv_path: str):
    """Save results list to CSV."""
    if not results:
        return

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = list(results[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def load_results_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load results from a CSV file.

    Args:
        csv_path: Path to the CSV results file.

    Returns:
        List of result dicts with numeric values converted to floats.
    """
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in row:
                try:
                    row[key] = float(row[key])
                except (ValueError, TypeError):
                    pass
            results.append(row)
    return results


def print_results_summary(csv_path: str):
    """Print a formatted summary of benchmark results.

    Args:
        csv_path: Path to the results CSV.
    """
    results = load_results_csv(csv_path)

    print(f"\n{'='*80}")
    print(f"{'Condition':<30} {'Severity':>8} {'BLEU-4':>8} {'ROUGE-L':>8} {'Degrad%':>8}")
    print(f"{'='*80}")

    for row in results:
        cond = row.get('condition', 'unknown')
        sev = row.get('severity', 0)
        bleu4 = row.get('bleu4', 0)
        rougeL = row.get('rougeL', 0)
        deg = row.get('bleu4_degradation', 0)

        sev_str = f"{float(sev):.0%}" if float(sev) > 0 else "—"
        deg_str = f"{float(deg):.1f}%" if cond != 'clean' else "—"

        print(f"{cond:<30} {sev_str:>8} {float(bleu4):>8.2f} {float(rougeL):>8.4f} {deg_str:>8}")

    print(f"{'='*80}")


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run misalignment benchmark on SLT models"
    )
    parser.add_argument(
        "--model",
        default="gfslt_vlp",
        choices=["gfslt_vlp"],
        help="Model to benchmark",
    )
    parser.add_argument(
        "--gfslt_dir",
        default="baselines/GFSLT-VLP",
        help="Path to GFSLT-VLP repo",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_root",
        default="data/phoenix14t",
        help="PHOENIX14T data root",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["dev", "test"],
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Output directory for all results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--severity_levels",
        type=float,
        nargs="+",
        default=None,
        help="Custom severity levels (default: 0.05 0.10 0.15 0.20 0.25)",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Print summary of existing results CSV (skip inference)",
    )
    args = parser.parse_args()

    # If just printing summary
    if args.summary:
        print_results_summary(args.summary)
        return

    if args.model == "gfslt_vlp":
        run_benchmark_gfslt(
            gfslt_dir=args.gfslt_dir,
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            split=args.split,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
            severity_levels=args.severity_levels,
        )


if __name__ == "__main__":
    main()
