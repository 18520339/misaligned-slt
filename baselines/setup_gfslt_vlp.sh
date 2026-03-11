#!/bin/bash
# ============================================================
# Setup script for GFSLT-VLP baseline
# ============================================================
#
# This script:
#   1. Clones the GFSLT-VLP repository
#   2. Installs its dependencies
#   3. Downloads/prepares MBart weights
#   4. Documents how to train or where to get pretrained checkpoints
#
# Repository: https://github.com/zhoubenjia/GFSLT-VLP
# Paper: ICCV 2023
#
# Usage:
#   bash baselines/setup_gfslt_vlp.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GFSLT_DIR="${SCRIPT_DIR}/GFSLT-VLP"

echo "============================================="
echo "  Setting up GFSLT-VLP baseline"
echo "============================================="

# -------------------------------------------------------
# Step 1: Clone repository
# -------------------------------------------------------
if [ ! -d "$GFSLT_DIR" ]; then
    echo ""
    echo "[1/5] Cloning GFSLT-VLP repository..."
    git clone https://github.com/zhoubenjia/GFSLT-VLP.git "$GFSLT_DIR"
    echo "  Done."
else
    echo ""
    echo "[1/5] GFSLT-VLP already cloned at: $GFSLT_DIR"
fi

# -------------------------------------------------------
# Step 2: Install GFSLT-VLP dependencies
# -------------------------------------------------------
echo ""
echo "[2/5] Installing GFSLT-VLP Python dependencies..."
cd "$GFSLT_DIR"

pip install torch torchvision --quiet 2>/dev/null || true
pip install transformers sentencepiece protobuf --quiet 2>/dev/null || true
pip install timm einops loguru hpman hpargparse --quiet 2>/dev/null || true
pip install vidaug opencv-python --quiet 2>/dev/null || true
pip install sacrebleu wandb --quiet 2>/dev/null || true

# Also install our project's deps
pip install -r "${PROJECT_ROOT}/requirements.txt" --quiet 2>/dev/null || true

echo "  Dependencies installed."

# -------------------------------------------------------
# Step 3: Prepare MBart weights (trim_model.py)
# -------------------------------------------------------
echo ""
echo "[3/5] Preparing pretrained MBart weights..."

PRETRAIN_DIR="${GFSLT_DIR}/pretrain_models"
mkdir -p "$PRETRAIN_DIR"

if [ -f "${PRETRAIN_DIR}/MBart_trimmed/pytorch_model.bin" ] && \
   [ -f "${PRETRAIN_DIR}/mytran/pytorch_model.bin" ]; then
    echo "  MBart weights already prepared."
else
    echo "  Running trim_model.py to prepare MBart weights from HuggingFace..."
    echo "  (This downloads facebook/mbart-large-cc25 and trims it)"
    cd "$GFSLT_DIR"
    python trim_model.py 2>/dev/null || {
        echo ""
        echo "  ⚠ trim_model.py failed. You can manually download weights from:"
        echo "    Baidu Netdisk: https://pan.baidu.com/s/15h9dsHMPH8dXH7glZvZnng?pwd=4s1p"
        echo "    Extraction code: 4s1p"
        echo ""
        echo "  After downloading, extract to: ${PRETRAIN_DIR}/"
        echo "  Expected structure:"
        echo "    pretrain_models/"
        echo "    ├── MBart_trimmed/"
        echo "    │   ├── config.json"
        echo "    │   ├── pytorch_model.bin"
        echo "    │   ├── sentencepiece.bpe.model"
        echo "    │   ├── special_tokens_map.json"
        echo "    │   └── tokenizer_config.json"
        echo "    └── mytran/"
        echo "        ├── config.json"
        echo "        └── pytorch_model.bin"
    }
fi

# -------------------------------------------------------
# Step 4: Prepare PHOENIX14T data for GFSLT-VLP
# -------------------------------------------------------
echo ""
echo "[4/5] Preparing PHOENIX14T data..."

GFSLT_DATA_DIR="${GFSLT_DIR}/data/Phonexi-2014T"
mkdir -p "$GFSLT_DATA_DIR"

if [ -f "${GFSLT_DATA_DIR}/labels.train" ] && \
   [ -f "${GFSLT_DATA_DIR}/labels.dev" ] && \
   [ -f "${GFSLT_DATA_DIR}/labels.test" ]; then
    echo "  PHOENIX14T label files already exist."
else
    echo ""
    echo "  PHOENIX14T label files not found at: ${GFSLT_DATA_DIR}/"
    echo ""
    echo "  You need to create the label files. GFSLT-VLP uses pickle files"
    echo "  containing a dict with keys: 'name', 'text', 'imgs_path', 'length'."
    echo ""
    echo "  To create them, run the following Python script:"
    echo ""
    cat << 'PYTHON_SCRIPT'
    # Save this as: create_gfslt_labels.py
    # Run: python create_gfslt_labels.py --phoenix_root /path/to/PHOENIX-2014-T

    import os, pickle, csv, glob, argparse

    def create_labels(phoenix_root, output_dir, split):
        # PHOENIX14T CSV annotation files use '|' delimiter
        csv_file = os.path.join(phoenix_root, 'annotations',
                                f'PHOENIX-2014-T.{split}.corpus.csv')
        if not os.path.exists(csv_file):
            # Try alternate location
            csv_file = os.path.join(phoenix_root,
                                    f'PHOENIX-2014-T.{split}.corpus.csv')

        img_root = os.path.join(phoenix_root, 'features', 'fullFrame-210x260px')

        data = {}
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='|')
            for idx, row in enumerate(reader):
                name = row.get('name', row.get('id', str(idx)))
                folder = row.get('folder', name)
                text = row.get('translation', row.get('text', row.get('orth', '')))

                # Find frame images
                frame_dir = os.path.join(img_root, split, folder)
                if not os.path.exists(frame_dir):
                    frame_dir = os.path.join(img_root, folder)

                frame_files = sorted(glob.glob(os.path.join(frame_dir, '*.png')))
                if not frame_files:
                    frame_files = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))

                # Store paths relative to img_root
                imgs_path = [os.path.join(split, folder, os.path.basename(f))
                            for f in frame_files]

                data[f'{split}_{idx}'] = {
                    'name': name,
                    'text': text,
                    'imgs_path': [f'/{p}' for p in imgs_path],
                    'length': len(frame_files),
                }

        output_path = os.path.join(output_dir, f'labels.{split}')
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f'Created {output_path} with {len(data)} samples')

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--phoenix_root', required=True)
        parser.add_argument('--output_dir', default='data/Phonexi-2014T')
        args = parser.parse_args()

        os.makedirs(args.output_dir, exist_ok=True)
        for split in ['train', 'dev', 'test']:
            create_labels(args.phoenix_root, args.output_dir, split)
PYTHON_SCRIPT
    echo ""
fi

# -------------------------------------------------------
# Step 5: Update config paths
# -------------------------------------------------------
echo ""
echo "[5/5] Updating config file paths..."

# Create a local config that points to correct paths
CONFIG_FILE="${GFSLT_DIR}/configs/config_gloss_free.yaml"
cat > "$CONFIG_FILE" << EOF
name: GFSLT-VLP
data:
  train_label_path: ${GFSLT_DATA_DIR}/labels.train
  dev_label_path: ${GFSLT_DATA_DIR}/labels.dev
  test_label_path: ${GFSLT_DATA_DIR}/labels.test
  img_path: ${PROJECT_ROOT}/data/phoenix14t/features/fullFrame-210x260px
  max_length: 300
training:
  wandb: disabled
  scale_embedding: False
model:
  tokenizer: ${PRETRAIN_DIR}/MBart_trimmed
  transformer: ${PRETRAIN_DIR}/MBart_trimmed
  visual_encoder: ${PRETRAIN_DIR}/mytran
  sign_proj: True
EOF

echo "  Config updated: $CONFIG_FILE"

# -------------------------------------------------------
# Done
# -------------------------------------------------------
echo ""
echo "============================================="
echo "  GFSLT-VLP Setup Complete!"
echo "============================================="
echo ""
echo "  Next steps:"
echo ""
echo "  1. Download PHOENIX14T data (if not done):"
echo "     https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/"
echo "     Extract to: ${PROJECT_ROOT}/data/phoenix14t/"
echo ""
echo "  2. Create label files (if not done):"
echo "     python create_gfslt_labels.py --phoenix_root ${PROJECT_ROOT}/data/phoenix14t"
echo ""
echo "  3. Train the model (4 GPU, ~2 days):"
echo "     cd ${GFSLT_DIR}"
echo ""
echo "     # Step A: VLP Pre-training"
echo "     CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \\"
echo "       --nproc_per_node=4 --master_port=1236 --use_env train_vlp.py \\"
echo "       --batch-size 4 --epochs 80 --opt sgd --lr 0.01 --output_dir out/vlp"
echo ""
echo "     # Step B: SLT Fine-tuning"
echo "     CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \\"
echo "       --nproc_per_node=4 --master_port=1236 --use_env train_slt.py \\"
echo "       --batch-size 2 --epochs 200 --opt sgd --lr 0.01 \\"
echo "       --output_dir out/Gloss-Free --finetune ./out/vlp/checkpoint.pth"
echo ""
echo "  4. Run the benchmark:"
echo "     cd ${PROJECT_ROOT}"
echo "     python -m evaluation.benchmark \\"
echo "       --model gfslt_vlp \\"
echo "       --gfslt_dir ${GFSLT_DIR} \\"
echo "       --checkpoint ${GFSLT_DIR}/out/Gloss-Free/best_checkpoint.pth \\"
echo "       --data_root data/phoenix14t"
echo ""
