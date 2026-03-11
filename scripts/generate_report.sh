#!/bin/bash
# ============================================================
# Generate all plots, tables, and reports from existing results
# ============================================================
#
# This script generates visualization outputs from already-computed
# benchmark results (CSV files). Use this to regenerate plots
# without re-running inference.
#
# Usage:
#   bash scripts/generate_report.sh
#   bash scripts/generate_report.sh --results_csv results/tables/gfslt_vlp_results.csv
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-results}"
RESULTS_CSV="${1:-${OUTPUT_DIR}/tables/gfslt_vlp_results.csv}"

echo "============================================="
echo "  Generating Report"
echo "============================================="
echo ""
echo "  Results CSV: $RESULTS_CSV"
echo "  Output: $OUTPUT_DIR"
echo ""

if [ ! -f "$RESULTS_CSV" ]; then
    echo "ERROR: Results CSV not found: $RESULTS_CSV"
    echo "  Run the benchmark first: bash scripts/run_benchmark.sh"
    exit 1
fi

python << PYTHON_SCRIPT
import os, sys
sys.path.insert(0, '.')

from evaluation.visualize import (
    plot_degradation_curves, plot_heatmap,
    plot_failure_distribution,
)
from evaluation.benchmark import print_results_summary

results_csv = '${RESULTS_CSV}'
plots_dir = '${OUTPUT_DIR}/plots'
tables_dir = '${OUTPUT_DIR}/tables'
os.makedirs(plots_dir, exist_ok=True)

model_name = os.path.basename(results_csv).replace('_results.csv', '').upper().replace('_', '-')
print(f'Generating plots for {model_name}...')

# Degradation curves for multiple metrics
for metric in ['bleu4', 'bleu1', 'rougeL']:
    path = plot_degradation_curves(results_csv, plots_dir, metric=metric, model_name=model_name)
    print(f'  Created: {path}')

# Heatmaps at different severity levels
for sev in [0.10, 0.15, 0.20, 0.25]:
    path = plot_heatmap(results_csv, plots_dir, severity=sev, model_name=model_name)
    print(f'  Created: {path}')

# Failure distribution (if failure CSV exists)
failure_csv = results_csv.replace('_results.csv', '_failures.csv')
if os.path.exists(failure_csv):
    path = plot_failure_distribution(failure_csv, plots_dir, model_name=model_name)
    print(f'  Created: {path}')

# Print summary table
print()
print_results_summary(results_csv)
PYTHON_SCRIPT

echo ""
echo "============================================="
echo "  Report complete!"
echo "============================================="
echo "  Plots saved to: ${OUTPUT_DIR}/plots/"
