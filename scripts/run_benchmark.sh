#!/bin/bash
# ============================================================
# Run the full misalignment benchmark
# ============================================================
#
# This script runs the complete benchmark pipeline:
#   1. GFSLT-VLP inference on all 46 conditions
#   2. Metric computation (BLEU-1..4, ROUGE-L)
#   3. Plot generation (degradation curves, heatmaps)
#   4. Failure analysis
#   5. Sample translation extraction
#
# Usage:
#   bash scripts/run_benchmark.sh
#
# Environment variables:
#   GFSLT_CHECKPOINT  Path to trained checkpoint (required)
#   GFSLT_DIR         Path to GFSLT-VLP repo (default: baselines/GFSLT-VLP)
#   DATA_ROOT         Path to PHOENIX14T data (default: data/phoenix14t)
#   SPLIT             Dataset split (default: test)
#   OUTPUT_DIR        Output directory (default: results)
#   BATCH_SIZE        Inference batch size (default: 4)
#   DEVICE            cuda or cpu (default: cuda)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
GFSLT_DIR="${GFSLT_DIR:-baselines/GFSLT-VLP}"
DATA_ROOT="${DATA_ROOT:-data/phoenix14t}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda}"

# Find checkpoint
if [ -z "$GFSLT_CHECKPOINT" ]; then
    # Try default location
    GFSLT_CHECKPOINT="${GFSLT_DIR}/out/Gloss-Free/best_checkpoint.pth"
fi

echo "============================================="
echo "  Misalignment Benchmark"
echo "============================================="
echo ""
echo "  Model:      GFSLT-VLP"
echo "  Checkpoint: $GFSLT_CHECKPOINT"
echo "  Data root:  $DATA_ROOT"
echo "  Split:      $SPLIT"
echo "  Output:     $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Device:     $DEVICE"
echo ""

if [ ! -f "$GFSLT_CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at: $GFSLT_CHECKPOINT"
    echo ""
    echo "Please set GFSLT_CHECKPOINT environment variable or train the model first."
    echo "See: bash baselines/setup_gfslt_vlp.sh"
    exit 1
fi

# Run benchmark
python -m evaluation.benchmark \
    --model gfslt_vlp \
    --gfslt_dir "$GFSLT_DIR" \
    --checkpoint "$GFSLT_CHECKPOINT" \
    --data_root "$DATA_ROOT" \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE"

echo ""
echo "============================================="
echo "  Benchmark complete!"
echo "============================================="
echo ""
echo "  Results: ${OUTPUT_DIR}/tables/"
echo "  Plots:   ${OUTPUT_DIR}/plots/"
echo "  Samples: ${OUTPUT_DIR}/samples/"
echo ""
echo "  To view results summary:"
echo "    python -m evaluation.benchmark --summary ${OUTPUT_DIR}/tables/gfslt_vlp_results.csv"
