# Temporal Misalignment Benchmark for Sign Language Translation

Systematic evaluation of how temporal misalignment (segmentation errors) degrades
pretrained SLT models. Uses MSKA-SLT on Phoenix-2014T as the primary testbed.

## Project Structure

```
.
├── configs/benchmark.yaml   # Paths, severity levels, conditions
├── misalign.py              # Misalignment simulation (truncation + contamination)
├── run_benchmark.py         # Run inference on all 41 conditions
├── analyze.py               # Generate all visualizations and analysis
├── requirements.txt         # Additional Python dependencies
├── results/                 # Output directory (created by scripts)
│   ├── benchmark_results.json
│   ├── heatmap.png
│   ├── degradation_curves.png
│   ├── knee_points.csv
│   ├── scores.csv
│   ├── sample_translations.md
│   └── failure_distribution.png
└── MSKA/                    # Cloned MSKA repo (https://github.com/sutwangyan/MSKA)
```

## Setup

### 1. Clone MSKA (already done if you cloned this repo)

```bash
git clone https://github.com/sutwangyan/MSKA.git
```

### 2. Install dependencies

```bash
# MSKA environment (Python 3.10, PyTorch + TensorFlow + transformers)
conda create -n mska python=3.10
conda activate mska
pip install -r MSKA/requirements.txt

# Our additional dependencies
pip install -r requirements.txt
```

### 3. Download required files

Place all files inside the `MSKA/` directory:

| File | Source | Location |
|------|--------|----------|
| Phoenix-2014T keypoints | [Google Drive](https://drive.google.com/drive/folders/1XBBqsxJqM4M64iGxhVCNuqUInhaACUwi) | `MSKA/data/Phoenix-2014T/Phoenix-2014T.{train,dev,test}` |
| mBart-de language model | [Google Drive](https://drive.google.com/drive/folders/1u7uhrwaBL6sNqscFerJLUHjwt1kuwWw9) | `MSKA/pretrained_models/mBart_de/` |
| MSKA-SLT checkpoint | [Google Drive](https://drive.google.com/drive/folders/1kQhvT-gJBfarkV2jtigBnO24Ial95znc) | `MSKA/pretrained_models/Phoenix-2014T_SLT/best.pth` |

`MSKA/data/Phoenix-2014T/gloss2ids.pkl` is already in the repo.

## Running

### Step 1: Verify clean baseline

```bash
python run_benchmark.py --clean-only
```

Expected: BLEU-4 ~ 29.03, ROUGE-L ~ 53.54 (matching MSKA paper).

### Step 2: Run full benchmark (41 conditions)

```bash
python run_benchmark.py
```

This runs 41 inference passes on the test set. Progress is printed.

### Step 3: Generate analysis

```bash
python analyze.py
```

Produces all plots and tables in `results/`.

### Optional: Sanity-check misalignment logic

```bash
python misalign.py
```

Prints expected output lengths for every condition (no model needed).

## Misalignment Conditions

9 types x 5 severity levels (10-50%) + 1 clean = 41 conditions.

| delta_s \\ delta_e | Tail Trunc (de<0) | Clean (de=0) | Tail Contam (de>0) |
|--------------------|-------------------|--------------|---------------------|
| Head Contam (ds<0) | HC + TT           | HC only      | HC + TC             |
| Clean (ds=0)       | TT only           | **CLEAN**    | TC only             |
| Head Trunc (ds>0)  | HT + TT           | HT only      | HT + TC             |

Offsets are percentages of the original sequence length T.
