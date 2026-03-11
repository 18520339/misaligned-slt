#!/bin/bash
# ============================================================
# Download and prepare PHOENIX14T dataset
# ============================================================
#
# PHOENIX14T (RWTH-PHOENIX-Weather-2014T) is a German Sign Language
# dataset with text translations. It requires manual download from
# the RWTH website after registration.
#
# Usage:
#   bash scripts/prepare_phoenix14t.sh --phoenix_root /path/to/downloaded/data
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data/phoenix14t"

# Parse args
PHOENIX_ROOT="${1:-}"
if [ -z "$PHOENIX_ROOT" ]; then
    echo "============================================="
    echo "  PHOENIX14T Dataset Preparation"
    echo "============================================="
    echo ""
    echo "Usage:"
    echo "  bash scripts/prepare_phoenix14t.sh /path/to/PHOENIX-2014-T"
    echo ""
    echo "Steps:"
    echo ""
    echo "  1. Register and download PHOENIX14T from:"
    echo "     https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/"
    echo ""
    echo "  2. Extract the downloaded archive"
    echo ""
    echo "  3. Run this script pointing to the extracted directory"
    echo ""
    echo "  Expected directory structure after download:"
    echo "    PHOENIX-2014-T/"
    echo "    ├── annotations/"
    echo "    │   ├── PHOENIX-2014-T.train.corpus.csv"
    echo "    │   ├── PHOENIX-2014-T.dev.corpus.csv"
    echo "    │   └── PHOENIX-2014-T.test.corpus.csv"
    echo "    └── features/"
    echo "        └── fullFrame-210x260px/"
    echo "            ├── train/"
    echo "            │   └── <sample_folders>/"
    echo "            │       └── images0001.png, images0002.png, ..."
    echo "            ├── dev/"
    echo "            └── test/"
    exit 0
fi

echo "============================================="
echo "  Preparing PHOENIX14T Dataset"
echo "============================================="
echo ""
echo "Source: $PHOENIX_ROOT"
echo "Target: $DATA_DIR"

# Create symlink to the data
mkdir -p "$(dirname "$DATA_DIR")"

if [ -L "$DATA_DIR" ]; then
    echo ""
    echo "Symlink already exists at: $DATA_DIR"
elif [ -d "$DATA_DIR" ]; then
    echo ""
    echo "Directory already exists at: $DATA_DIR"
else
    echo ""
    echo "Creating symlink..."
    ln -s "$PHOENIX_ROOT" "$DATA_DIR"
    echo "  $DATA_DIR -> $PHOENIX_ROOT"
fi

# Verify data
echo ""
echo "Verifying dataset..."
ERRORS=0

for split in train dev test; do
    CSV_FILE="${DATA_DIR}/annotations/PHOENIX-2014-T.${split}.corpus.csv"
    if [ -f "$CSV_FILE" ]; then
        COUNT=$(wc -l < "$CSV_FILE")
        echo "  ✓ ${split}: ${COUNT} lines in annotation CSV"
    else
        echo "  ✗ ${split}: annotation CSV not found at $CSV_FILE"
        ERRORS=$((ERRORS + 1))
    fi

    FRAME_DIR="${DATA_DIR}/features/fullFrame-210x260px/${split}"
    if [ -d "$FRAME_DIR" ]; then
        NUM_SAMPLES=$(ls -d "$FRAME_DIR"/*/ 2>/dev/null | wc -l)
        echo "  ✓ ${split}: ${NUM_SAMPLES} sample directories"
    else
        echo "  ✗ ${split}: frame directory not found at $FRAME_DIR"
        ERRORS=$((ERRORS + 1))
    fi
done

if [ $ERRORS -gt 0 ]; then
    echo ""
    echo "⚠ Found $ERRORS issues. Please check your data paths."
    exit 1
fi

# Create GFSLT-VLP label files
echo ""
echo "Creating GFSLT-VLP label files..."
GFSLT_DATA_DIR="${PROJECT_ROOT}/baselines/GFSLT-VLP/data/Phonexi-2014T"

python "${SCRIPT_DIR}/create_gfslt_labels.py" \
    --phoenix_root "$DATA_DIR" \
    --output_dir "$GFSLT_DATA_DIR" \
    --img_root "${DATA_DIR}/features/fullFrame-210x260px"

echo ""
echo "============================================="
echo "  Dataset preparation complete!"
echo "============================================="
echo ""
echo "  Next: Run the GFSLT-VLP setup:"
echo "    bash baselines/setup_gfslt_vlp.sh"
