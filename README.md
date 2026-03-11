# Temporal Misalignment Robustness for Gloss-Free Sign Language Translation

This project studies how **temporal misalignment** (segmentation errors) affects gloss-free sign language translation (SLT) models in streaming/online scenarios.

## Key Idea

In real-time SLT, an automatic segmenter splits a continuous video stream into sentence-level segments. These segments are never perfectly aligned — they may start too late (missing beginning signs), end too early (cutting off ending signs), or include frames from adjacent sentences. We systematically simulate these errors and measure how badly they degrade translation quality.

## Components

1. **Misalignment Simulation Benchmark** — Parameterised truncation and contamination of video segments
2. **Boundary-Aware Context Adapter (BACA)** — A lightweight adapter that wraps frozen pretrained SLT models to make them robust to misalignment *(coming soon)*
3. **Evaluation & Analysis Tools** — Metrics, degradation curves, heatmaps, failure analysis

## Quick Start

```bash
# 1. Create conda environment
conda create -n misalign-slt python=3.10
conda activate misalign-slt

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare PHOENIX14T dataset (see scripts/prepare_phoenix14t.sh)
bash scripts/prepare_phoenix14t.sh

# 4. Set up GFSLT-VLP baseline
bash baselines/setup_gfslt_vlp.sh

# 5. Run full benchmark (all 46 conditions)
bash scripts/run_benchmark.sh

# 6. Generate plots and reports
bash scripts/generate_report.sh
```

## Project Structure

```
misaligned-slt/
├── configs/              # YAML configuration files
├── data/                 # Misalignment simulation and dataset loaders
├── adapters/             # BACA adapter modules (stubs)
├── baselines/            # Baseline model wrappers (GFSLT-VLP, Uni-Sign)
├── evaluation/           # Metrics, benchmarking, visualization, failure analysis
├── training/             # Adapter training loop (stubs)
├── scripts/              # Shell scripts for data prep, benchmarking, reporting
├── tests/                # Unit tests
└── results/              # Output: tables, plots, sample translations
```

## Dataset

**PHOENIX14T** (RWTH-PHOENIX-Weather-2014T): German Sign Language weather forecasts with German text translations. ~8,257 samples across train/dev/test splits.

## Misalignment Model

Each segment `[0, T]` is perturbed by two offsets:
- **δ_s** (start): positive = head truncation, negative = head contamination
- **δ_e** (end): negative = tail truncation, positive = tail contamination

This produces 9 condition types × 5 severity levels = 45 misaligned + 1 clean = **46 total evaluation conditions**.

## Citation

*Paper in preparation.*
