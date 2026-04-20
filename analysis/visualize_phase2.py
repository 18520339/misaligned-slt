'''Phase 2 visualization: compare 4 models in the {AR, BD} x {Clean, Aug} ablation matrix.

Figures:
  1. Robustness Degradation Curves — 2x2 grid (one subplot per model), BLEU-4 vs severity
  2. Clean vs Robust Tradeoff — scatter of clean BLEU-4 vs mean misaligned BLEU-4
  3. Failure Mode Comparison — stacked bars across models at representative severities
  4. Improvement Heatmap — Model C (bd_aug) minus Baseline delta across conditions
  5. Training Curves — loss and dev BLEU-4 over epochs for each trained model
'''
import os, json
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from analysis.visualize_phase1 import (
    BASIC_ORDER, CONDITION_COLORS, CONDITION_MARKERS, CONDITION_LABELS,
    COMPOUND_PAIR_COLORS, COMPOUND_PAIR_LABELS,
    _extract_basic_curves, _extract_compound_mean_curves,
    _save_fig, _relative_degradation,
)
from analysis.failure_modes import (
    classify_all_predictions, failure_mode_distribution,
    FAILURE_COLORS, FAILURE_MODES,
)
from analysis.tables import _canonical_pair, _parse_compound_name

# ── Model display config ─────────────────────────────────────────────────────
MODEL_ORDER = ['baseline', 'ar_aug', 'bd_clean', 'bd_aug']
MODEL_LABELS = {'baseline': 'Baseline (AR, Clean)', 'ar_aug': 'AR + Aug', 'bd_clean': 'BD, Clean', 'bd_aug': 'BD + Aug'}
MODEL_COLORS = {'baseline': '#7f8c8d', 'ar_aug': '#e74c3c', 'bd_clean': '#3498db', 'bd_aug': '#2ecc71'}
MODEL_MARKERS = {'baseline': 'o', 'ar_aug': 's', 'bd_clean': '^', 'bd_aug': 'D'}

matplotlib.use('Agg')
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})

def _load_benchmark(results_dir, model_name):
    # Load benchmark results for a model. Baseline uses benchmark.json, others use benchmark_{name}.json
    path = os.path.join(results_dir, 'raw', 'benchmark.json' if model_name == 'baseline' else f'benchmark_{model_name}.json')
    if not os.path.exists(path): return None
    with open(path) as f:
        return json.load(f)

def _load_training_log(results_dir, model_name):
    path = os.path.join(results_dir, 'checkpoints', model_name, 'training_log.jsonl')
    if not os.path.exists(path): return None
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line: entries.append(json.loads(line))
    return entries

def _get_severity_levels(results): # Extract severity levels from condition keys.
    sevs = set()
    for key in results:
        if key in ('meta', 'clean') or '+' in key: continue
        parts = key.split('_')
        if len(parts) == 2:
            try: sevs.add(int(parts[1]) / 100)
            except ValueError: pass
    return sorted(sevs)

def _mean_misaligned_metric(results, metric='bleu4'): # Compute mean metric across all misaligned (non-clean) conditions.
    vals = []
    for key, data in results.items():
        if key in ('meta', 'clean'): continue
        v = data.get('metrics', {}).get(metric)
        if v is not None: vals.append(v)
    return np.mean(vals) if vals else 0.0


# Figure 1: Robustness Degradation Curves (2x2 grid), each subplot shows BLEU-4 degradation curves for one model
def fig01_degradation_grid(all_results, output_dir):
    available = [m for m in MODEL_ORDER if all_results.get(m) is not None]
    if len(available) < 2:
        print('  fig01: need at least 2 models, skipping')
        return

    n = len(available)
    ncols, nrows = 2, (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5.5 * nrows), squeeze=False)

    for idx, model_name in enumerate(available):
        ax = axes[idx // ncols][idx % ncols]
        results = all_results[model_name]
        severity_levels = _get_severity_levels(results)
        sevs_pct = [s * 100 for s in severity_levels]
        curves = _extract_basic_curves(results, severity_levels, 'bleu4')
        clean_bleu = results.get('clean', {}).get('metrics', {}).get('bleu4', 0)

        for ctype in BASIC_ORDER:
            vals = curves[ctype]
            valid = [(s, v) for s, v in zip(sevs_pct, vals) if v is not None]
            if not valid: continue
            xs, ys = zip(*valid)
            ax.plot(xs, ys, color=CONDITION_COLORS[ctype], marker=CONDITION_MARKERS[ctype],
                    label=CONDITION_LABELS[ctype], linewidth=2, markersize=6)

        if clean_bleu: 
            ax.axhline(clean_bleu, color='#555', linestyle='--', linewidth=1.2, alpha=0.7, label=f'Clean ({clean_bleu:.1f})')

        ax.set_xlabel('Severity (%)')
        ax.set_ylabel('BLEU-4')
        ax.set_title(MODEL_LABELS.get(model_name, model_name))
        ax.set_xlim(left=0)
        if idx == 0: ax.legend(loc='lower left', fontsize=9)

    for idx in range(n, nrows * ncols): # Hide unused subplots
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle('BLEU-4 Degradation Under Temporal Misalignment (per model)', fontsize=14, y=1.01)
    fig.tight_layout()
    _save_fig(fig, output_dir, 'phase2_fig01_degradation_grid')


# Figure 2: Scatter: x = clean BLEU-4, y = mean misaligned BLEU-4, per model
def fig02_clean_vs_robust(all_results, output_dir): 
    fig, ax = plt.subplots(figsize=(8, 7))

    for model_name in MODEL_ORDER:
        results = all_results.get(model_name)
        if results is None: continue
        clean_bleu = results.get('clean', {}).get('metrics', {}).get('bleu4', 0)
        mean_mis = _mean_misaligned_metric(results, 'bleu4')
        ax.scatter(clean_bleu, mean_mis, s=200, zorder=5, color=MODEL_COLORS[model_name], marker=MODEL_MARKERS[model_name],
                   edgecolors='white', linewidth=1.5, label=MODEL_LABELS.get(model_name, model_name))
        ax.annotate(MODEL_LABELS.get(model_name, model_name), (clean_bleu, mean_mis),
                    textcoords='offset points', xytext=(10, 8), fontsize=10)

    # Ideal direction arrow
    ax.annotate('', xy=(1.0, 1.0), xytext=(0.7, 0.7), xycoords='axes fraction', 
                textcoords='axes fraction', arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(0.88, 0.88, 'Ideal', transform=ax.transAxes, fontsize=10, color='gray', ha='center')

    ax.set_xlabel('Clean BLEU-4')
    ax.set_ylabel('Mean Misaligned BLEU-4')
    ax.set_title('Clean vs Robustness Tradeoff')
    ax.legend(loc='lower right')
    fig.tight_layout()
    _save_fig(fig, output_dir, 'phase2_fig02_clean_vs_robust')


# Figure 3: Heatmap of BLEU-4 improvement from baseline to bd_aug across conditions
def fig03_improvement_heatmap(all_results, output_dir):
    baseline = all_results.get('baseline')
    bd_aug = all_results.get('bd_aug')
    if baseline is None or bd_aug is None:
        print('  fig03: need baseline and bd_aug, skipping')
        return

    severity_levels = _get_severity_levels(baseline)
    n_types = len(BASIC_ORDER)
    n_sevs = len(severity_levels)

    # Build improvement matrix
    matrix = np.full((n_types, n_sevs), np.nan)
    for ti, ctype in enumerate(BASIC_ORDER):
        for si, sev in enumerate(severity_levels):
            cond = f'{ctype}_{int(sev * 100):02d}'
            base_val = baseline.get(cond, {}).get('metrics', {}).get('bleu4')
            aug_val = bd_aug.get(cond, {}).get('metrics', {}).get('bleu4')
            if base_val is not None and aug_val is not None:
                matrix[ti, si] = aug_val - base_val

    fig, ax = plt.subplots(figsize=(12, 4))
    sev_labels = [f'{int(s * 100)}%' for s in severity_levels]
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-np.nanmax(np.abs(matrix)), vmax=np.nanmax(np.abs(matrix)))

    ax.set_xticks(range(n_sevs))
    ax.set_xticklabels(sev_labels)
    ax.set_yticks(range(n_types))
    ax.set_yticklabels(BASIC_ORDER)
    ax.set_xlabel('Severity')
    ax.set_ylabel('Condition Type')
    ax.set_title('BLEU-4 Improvement: BD+Aug vs Baseline\n(green = improvement, red = regression)')

    # Annotate cells
    for ti in range(n_types):
        for si in range(n_sevs):
            val = matrix[ti, si]
            if not np.isnan(val):
                color = 'white' if abs(val) > np.nanmax(np.abs(matrix)) * 0.6 else 'black'
                ax.text(si, ti, f'{val:+.1f}', ha='center', va='center', fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label='BLEU-4 delta (pp)', shrink=0.8)
    fig.tight_layout()
    _save_fig(fig, output_dir, 'phase2_fig03_improvement_heatmap')


# Figure 4: Loss and dev BLEU-4 over epochs for each trained model
def fig04_training_curves(results_dir, output_dir):
    trained_models = ['ar_aug', 'bd_clean', 'bd_aug']
    logs = {}
    for m in trained_models:
        log = _load_training_log(results_dir, m)
        if log: logs[m] = log

    if not logs:
        print('  fig04: no training logs found, skipping')
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for model_name, entries in logs.items():
        epochs = [e['epoch'] for e in entries]
        losses = [e['train_loss'] for e in entries]
        bleu4s = [e.get('dev_bleu4', 0) for e in entries]

        color = MODEL_COLORS.get(model_name, '#333')
        label = MODEL_LABELS.get(model_name, model_name)
        ax1.plot(epochs, losses, color=color, linewidth=2, label=label)
        ax2.plot(epochs, bleu4s, color=color, linewidth=2, label=label)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss')
    ax1.legend()

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dev BLEU-4')
    ax2.set_title('Dev BLEU-4')
    ax2.legend()

    fig.suptitle('Training Curves', fontsize=14)
    fig.tight_layout()
    _save_fig(fig, output_dir, 'phase2_fig04_training_curves')


# Figure 5 (bonus): Overlay all models on same axes per condition ──────────
def fig05_overlay_per_condition(all_results, output_dir):
    available = [m for m in MODEL_ORDER if all_results.get(m) is not None]
    if len(available) < 2: return

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    for ci, ctype in enumerate(BASIC_ORDER):
        ax = axes[ci]
        for model_name in available:
            results = all_results[model_name]
            severity_levels = _get_severity_levels(results)
            sevs_pct = [s * 100 for s in severity_levels]
            curves = _extract_basic_curves(results, severity_levels, 'bleu4')
            vals = curves[ctype]
            valid = [(s, v) for s, v in zip(sevs_pct, vals) if v is not None]
            if not valid: continue
            xs, ys = zip(*valid)
            ax.plot(xs, ys, color=MODEL_COLORS[model_name], marker=MODEL_MARKERS[model_name],
                    label=MODEL_LABELS.get(model_name, model_name), linewidth=2, markersize=5)

        ax.set_xlabel('Severity (%)')
        ax.set_title(CONDITION_LABELS[ctype])
        if ci == 0:
            ax.set_ylabel('BLEU-4')
            ax.legend(fontsize=8)

    fig.suptitle('BLEU-4 Degradation: All Models Compared', fontsize=14)
    fig.tight_layout()
    _save_fig(fig, output_dir, 'phase2_fig05_overlay_per_condition')


# Main entry point
def generate_phase2_figures(results_dir, output_dir):
    '''Generate all Phase 2 comparison figures.

    Args:
        results_dir: directory containing raw/ and checkpoints/ subdirs
        output_dir: directory for output figures
    '''
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f'\n=== Generating Phase 2 Figures ===')
    print(f'  Results: {results_dir}')
    print(f'  Output:  {output_dir}\n')

    # Load all available model benchmarks
    all_results = {}
    for model_name in MODEL_ORDER:
        data = _load_benchmark(results_dir, model_name)
        if data is not None:
            all_results[model_name] = data
            clean = data.get('clean', {}).get('metrics', {})
            print(f"  Loaded {model_name}: BLEU-4={clean.get('bleu4', '?'):.2f}, ROUGE-L={clean.get('rouge_l', '?'):.2f}")
        else: print(f'  {model_name}: not found (skipping)')

    if len(all_results) < 2:
        print('\nNeed at least 2 model results to generate comparison figures.')
        return

    print('  Generating fig01: degradation grid...')
    fig01_degradation_grid(all_results, output_dir)

    print('  Generating fig02: clean vs robust tradeoff...')
    fig02_clean_vs_robust(all_results, output_dir)

    print('  Generating fig03: improvement heatmap...')
    fig03_improvement_heatmap(all_results, output_dir)

    print('  Generating fig04: training curves...')
    fig04_training_curves(results_dir, output_dir)

    print('  Generating fig05: overlay per condition...')
    fig05_overlay_per_condition(all_results, output_dir)