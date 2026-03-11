#!/bin/bash
# Setup script for Uni-Sign baseline (stretch goal)
# Repository: https://github.com/ZechengLi19/Uni-Sign
# Weights: https://huggingface.co/ZechengLi19/Uni-Sign
# Paper: ICLR 2025

set -e

echo "============================================="
echo "  Setting up Uni-Sign baseline (stub)"
echo "============================================="

BASELINES_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$BASELINES_DIR")"
UNISIGN_DIR="${PROJECT_ROOT}/baselines/Uni-Sign"

# Clone repo
if [ ! -d "$UNISIGN_DIR" ]; then
    echo "[1/3] Cloning Uni-Sign repository..."
    git clone https://github.com/ZechengLi19/Uni-Sign.git "$UNISIGN_DIR"
else
    echo "[1/3] Uni-Sign repository already exists."
fi

# Download weights
echo "[2/3] Downloading weights from HuggingFace..."
echo "  pip install huggingface_hub"
echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download('ZechengLi19/Uni-Sign', local_dir='${UNISIGN_DIR}/weights')\""

# Note about pose extraction
echo "[3/3] Uni-Sign uses pose keypoints, not raw RGB frames."
echo "  You may need to extract poses for PHOENIX14T using their preprocessing pipeline."
echo "  See the Uni-Sign README for details."

echo ""
echo "============================================="
echo "  Uni-Sign setup stub complete"
echo "============================================="
