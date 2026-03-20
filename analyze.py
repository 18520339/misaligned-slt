'''Analyze benchmark results and produce all visualizations.

Reads results/benchmark_results.json and generates:
  1.  Degradation curves               (degradation_curves.png)
  2.  Knee-point analysis              (knee_points.csv)
  3.  Per-condition scores             (scores.csv)
  4.  Sample translations              (sample_translations.md)
  5.  Violin/box plots                 (violin_plots.png)
  6.  Length vs BLEU drop scatter      (length_vs_drop.png)
  7.  Multi-metric heatmap grid        (multi_metric_heatmap.png)
  8.  Radar/spider chart                (radar_chart.png)
  9.  Failure transition matrix         (failure_transitions.csv + failure_transitions.png)
  10. Statistical significance tests   (significance_tests.csv)
  11. Truncation vs contamination      (trunc_vs_contam.png)
  12. Sample robustness analysis       (sample_robustness.csv + sample_robustness.png)
  13. Sensitivity ranking              (sensitivity_ranking.csv)
  14. Failure distribution (detailed)  (failure_distribution.png)
  15. Compound severity grid           (compound_severity_grid.png)
  16. Compound interaction analysis    (compound_interaction.png + compound_analysis.csv)

  
Compound conditions (4 types × 5×5 = 100 entries) are represented throughout using
a max-severity aggregation: at severity level s, all (head_sev, tail_sev) pairs
where max(head_sev, tail_sev) == s are pooled and averaged.  This covers the full
asymmetric grid rather than only the diagonal (head_sev == tail_sev).

Usage:
    python analyze.py                           # default paths
    python analyze.py --results path/to/json    # custom input
'''
import os
import csv
import json
import random
import argparse
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from kneed import KneeLocator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MISALIGN_ORDER = [
    'head_trunc', 'tail_trunc', 'head_contam', 'tail_contam',
    'head_trunc_tail_trunc', 'head_trunc_tail_contam',
    'head_contam_tail_trunc', 'head_contam_tail_contam',
]
COMPOUND_CONDITIONS = [ # Must mirror misalign.py — used to pick the right storage key format
    'head_trunc_tail_trunc', 'head_trunc_tail_contam',
    'head_contam_tail_trunc', 'head_contam_tail_contam',
]
COMPOUND_TO_SINGLES = { # Maps each compound condition to its (head, tail) single-sided components
    'head_trunc_tail_trunc':   ('head_trunc',  'tail_trunc'),
    'head_trunc_tail_contam':  ('head_trunc',  'tail_contam'),
    'head_contam_tail_trunc':  ('head_contam', 'tail_trunc'),
    'head_contam_tail_contam': ('head_contam', 'tail_contam'),
}
PRETTY = {
    'head_trunc': 'Head Truncation', 'tail_trunc': 'Tail Truncation',
    'head_contam': 'Head Contamination', 'tail_contam': 'Tail Contamination',
    'head_trunc_tail_trunc': 'Head Trunc + Tail Trunc',
    'head_trunc_tail_contam': 'Head Trunc + Tail Contam',
    'head_contam_tail_trunc': 'Head Contam + Tail Trunc',
    'head_contam_tail_contam': 'Head Contam + Tail Contam',
}
SEVERITY_LEVELS = [10, 20, 30, 40, 50]

# Category groupings for conditions
TRUNC_ONLY = ['head_trunc', 'tail_trunc', 'head_trunc_tail_trunc']
CONTAM_ONLY = ['head_contam', 'tail_contam', 'head_contam_tail_contam']
MIXED = ['head_trunc_tail_contam', 'head_contam_tail_trunc']

# Module-level constants reused across multiple plotting functions
FAILURE_TYPES = ['acceptable', 'under-generation', 'hallucination', 'incoherent']
FAILURE_COLOURS = {
    'acceptable': '#4CAF50', 'under-generation': '#FF9800',
    'hallucination': '#F44336', 'incoherent': '#9C27B0',
}
CAT_COLORS = {'Truncation': '#1f77b4', 'Contamination': '#ff7f0e', 'Mixed': '#2ca02c'}
CAT_MAP = {c: 'Truncation' for c in TRUNC_ONLY}
CAT_MAP.update({c: 'Contamination' for c in CONTAM_ONLY})
CAT_MAP.update({c: 'Mixed' for c in MIXED})
COND_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']


def compound_2d_matrix(metrics, cond_name, metric_name='bleu4'):
    # Return (5×5) numpy array indexed by [head_sev_idx, tail_sev_idx]
    mat = np.full((len(SEVERITY_LEVELS), len(SEVERITY_LEVELS)), np.nan)
    cond_data = metrics.get(cond_name, {})
    for ri, head_sev in enumerate(SEVERITY_LEVELS):
        for ci, tail_sev in enumerate(SEVERITY_LEVELS):
            val = cond_data.get(f'h{head_sev}_t{tail_sev}', {}).get(metric_name)
            if val is not None: mat[ri, ci] = val
    return mat


def _compound_agg_metric(metrics, cond, metric_name, sev):
    '''Mean of metric_name for compound "cond" across all (h,t) with max(h,t)==sev.

    This covers the full off-diagonal compound grid rather than just the
    diagonal entry (h==t==sev).  Returns None when no data is available.
    '''
    cond_data = metrics.get(cond, {})
    vals = []
    for head_sev in SEVERITY_LEVELS:
        for tail_sev in SEVERITY_LEVELS:
            if max(head_sev, tail_sev) == sev:
                val = cond_data.get(f'h{head_sev}_t{tail_sev}', {}).get(metric_name)
                if val is not None: vals.append(val)
    return float(np.mean(vals)) if vals else None


def _compound_agg_samples(translations, cond, sev):
    '''Pool translation samples for compound "cond" where max(head_sev,tail_sev)==sev.

    Returns a flat list of sample dicts from all (h,t) pairs with max(h,t)==sev.
    '''
    samples = []
    for head_sev in SEVERITY_LEVELS:
        for tail_sev in SEVERITY_LEVELS:
            if max(head_sev, tail_sev) == sev:
                samples.extend(translations.get(f'{cond}_h{head_sev}_t{tail_sev}', []))
    return samples


def metric_matrix(metrics, metric_name):
    '''Return (8×5) numpy array: rows=conditions, cols=severity levels.

    Single conditions use the plain str(sev) key. Compound conditions use
    _compound_agg_metric (mean over all (h,t) pairs with max(h,t)==sev), so
    each cell represents the "average" performance across the full asymmetric
    grid at that maximum-severity level — not only the diagonal.
    '''
    mat = np.full((len(MISALIGN_ORDER), len(SEVERITY_LEVELS)), np.nan)
    for ri, cond in enumerate(MISALIGN_ORDER):
        if cond in COMPOUND_CONDITIONS:
            for ci, sev in enumerate(SEVERITY_LEVELS):
                v = _compound_agg_metric(metrics, cond, metric_name, sev)
                if v is not None: mat[ri, ci] = v
        else:
            cond_data = metrics.get(cond, {})
            for ci, sev in enumerate(SEVERITY_LEVELS):
                entry = cond_data.get(str(sev), {})
                val = entry.get(metric_name)
                if val is not None: mat[ri, ci] = val
    return mat


def find_knee(x, y):
    '''Detect knee point using the Kneedle algorithm (if kneed is installed)
    or fall back to a maximum-absolute-first-derivative (steepest-drop) heuristic.'''
    try:
        kl = KneeLocator(x, y, curve='convex', direction='decreasing', online=True, S=1.0)
        if kl.knee is not None:
            idx = list(x).index(kl.knee)
            return kl.knee, y[idx]
    except Exception: 
        print('  [KneeLocator failed, falling back to steepest drop heuristic]')
        pass
    
    # Fallback: steepest drop (max absolute first derivative).
    # More robust than second-derivative for monotonic or flat curves.
    if len(y) < 2: return x[0], y[0]
    d1 = np.abs(np.diff(y))
    if np.max(d1) == 0: return x[0], y[0] # flat curve
    idx = np.argmax(d1)
    return x[idx + 1], y[idx + 1]


def sentence_bleu_approx(ref, hyp): # Very simple unigram-precision proxy for sentence-level quality
    ref_toks = ref.lower().split()
    hyp_toks = hyp.lower().split()
    if not hyp_toks: return 0.0
    matches = sum(1 for t in hyp_toks if t in ref_toks)
    return matches / len(hyp_toks)


def classify_failure(ref, hyp): # Classify a single (ref, hyp) pair into a failure type
    ref_toks = ref.split()
    hyp_toks = hyp.split()
    ref_len = len(ref_toks)
    hyp_len = len(hyp_toks)

    # Check for repeated tokens (3+ consecutive same word, case-insensitive)
    hyp_lower = [t.lower() for t in hyp_toks]
    for i in range(len(hyp_lower) - 2):
        if hyp_lower[i] == hyp_lower[i+1] == hyp_lower[i+2]:
            return 'incoherent'

    # Under-generation: hyp < 50% of ref length
    if ref_len > 0 and hyp_len < 0.5 * ref_len:
        return 'under-generation'

    # Hallucination: normal/long length but very low overlap
    overlap = sentence_bleu_approx(ref, hyp)
    if hyp_len >= 0.5 * ref_len and overlap < 0.2:
        return 'hallucination'
    return 'acceptable'


def _count_failures(samples):
    '''Count failure types for a list of sample dicts with 'ref' and 'hyp' keys.

    Returns:
        dict mapping each failure type in FAILURE_TYPES to its count.
    '''
    dist = {ft: 0 for ft in FAILURE_TYPES}
    for s in samples:
        dist[classify_failure(s['ref'], s['hyp'])] += 1
    return dist


def _add_regression_line(ax, x_arr, y_arr, color='r', linestyle='--', linewidth=2, alpha=0.8):
    '''Fit a linear trend line to (x_arr, y_arr) and draw it on ax.

    Returns:
        (r, p): Pearson correlation coefficient and p-value.
                p is nan if scipy is not available.
                Both are None if fewer than 3 valid (finite) points.
    '''
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() < 3: return None, None
    
    xv, yv = x_arr[valid], y_arr[valid]
    coeffs = np.polyfit(xv, yv, 1)
    x_line = np.linspace(xv.min(), xv.max(), 100)
    ax.plot(x_line, np.poly1d(coeffs)(x_line), color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
    r, p = scipy_stats.pearsonr(xv, yv)
    return r, p


def _draw_stacked_bars(ax, x, data_by_type, width=0.7):
    '''Draw a stacked bar chart on ax using FAILURE_TYPES / FAILURE_COLOURS.

    Args:
        ax: matplotlib Axes.
        x: array-like of x positions.
        data_by_type: dict {failure_type: list/array of percentages}.
        width: bar width.
    '''
    bottom = np.zeros(len(x))
    for ft in FAILURE_TYPES:
        vals = np.array(data_by_type[ft])
        ax.bar(x, vals, bottom=bottom, label=ft.capitalize(), color=FAILURE_COLOURS[ft], width=width)
        bottom += vals


def has_per_sample_scores(data): # Check if per-sample sent_bleu data is available in translations
    translations = data.get('translations', {})
    clean_samples = translations.get('clean', [])
    if not clean_samples: return False
    # Check if at least one sample has sent_bleu
    return any(s.get('sent_bleu') is not None for s in clean_samples)


def has_length_data(data): # Check if per-sample T_original data is available
    translations = data.get('translations', {})
    clean_samples = translations.get('clean', [])
    if not clean_samples: return False
    return any(s.get('T_original') is not None for s in clean_samples)


def get_per_sample_scores(translations, key, score_field='sent_bleu'):
    # Extract per-sample scores as a dict {name: score} for a given translation key
    samples = translations.get(key, [])
    result = {}
    for s in samples:
        val = s.get(score_field)
        if val is not None: result[s['name']] = val
    return result


def get_per_sample_scores_compound(translations, cond, sev, score_field='sent_bleu'):
    '''Per-sample mean score for compound *cond* across all (h,t) with max(h,t)==sev.

    When a sample appears in multiple (h,t) compound variants, its scores are
    averaged — giving one value per sample for paired statistical tests.
    '''
    pooled = {}  # {name: [scores]}
    for head_sev in SEVERITY_LEVELS:
        for tail_sev in SEVERITY_LEVELS:
            if max(head_sev, tail_sev) == sev:
                for s in translations.get(f'{cond}_h{head_sev}_t{tail_sev}', []):
                    val = s.get(score_field)
                    if val is not None: pooled.setdefault(s['name'], []).append(val)
    return {name: float(np.mean(vals)) for name, vals in pooled.items()}


def _save_compound_csv(csv_rows, out_dir): # Write compound_analysis.csv if there are rows to write
    if not csv_rows: return
    df = pd.DataFrame(csv_rows)
    path = os.path.join(out_dir, 'compound_analysis.csv')
    df.to_csv(path, index=False)
    print(f'  Saved {path}')


# OUTPUT 1: Degradation Curves
def plot_degradation_curves(metrics, clean_bleu, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = [0] + SEVERITY_LEVELS
    colors = plt.cm.tab10.colors
    ax.axhline(clean_bleu, color='grey', linestyle='--', linewidth=1, label=f'Clean ({clean_bleu:.1f})')

    for i, cond in enumerate(MISALIGN_ORDER):
        cond_data = metrics.get(cond, {})
        y = [clean_bleu]
        for sev in SEVERITY_LEVELS:
            if cond in COMPOUND_CONDITIONS:
                val = _compound_agg_metric(metrics, cond, 'bleu4', sev)
            else:
                val = metrics.get(cond, {}).get(str(sev), {}).get('bleu4')
            y.append(val if val is not None else np.nan)
            
        y = np.array(y, dtype=float)
        ax.plot(
            x, y, label=PRETTY[cond], linewidth=1.5, markersize=6,
            marker=COND_MARKERS[i % len(COND_MARKERS)], color=colors[i % len(colors)], 
        )
        valid = ~np.isnan(y[1:])
        if valid.any(): # mark knee
            kx, ky = find_knee(np.array(SEVERITY_LEVELS)[valid], y[1:][valid])
            ax.plot(kx, ky, marker='|', color=colors[i % len(colors)], markersize=18, markeredgewidth=2)

    ax.set_xlabel('Max Severity (%)', fontsize=11)
    ax.set_ylabel('BLEU-4', fontsize=11)
    ax.set_title("BLEU-4 Degradation Under Temporal Misalignment\n"
                 "(Compound: mean over all h×t at max severity)", fontsize=12)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'{v}%' for v in x])
    ax.legend(fontsize=8, loc='lower left', ncol=2)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    path = os.path.join(out_dir, 'degradation_curves.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f'  Saved {path}')


# OUTPUT 2: Knee-Point Analysis CSV
def compute_knee_points(metrics, clean_bleu, out_dir):
    rows = []
    for cond in MISALIGN_ORDER:
        cond_data = metrics.get(cond, {})
        sevs, vals = [], []
        for sev in SEVERITY_LEVELS:
            if cond in COMPOUND_CONDITIONS:
                val = _compound_agg_metric(metrics, cond, 'bleu4', sev)
            else:
                val = metrics.get(cond, {}).get(str(sev), {}).get('bleu4')
                
            if val is not None:
                sevs.append(sev)
                vals.append(val)
                
        if not sevs: continue
        kx, ky = find_knee(np.array(sevs), np.array(vals))
        drop = (clean_bleu - ky) / clean_bleu * 100 if clean_bleu > 0 else 0
        rows.append({
            'misalignment_type': cond,
            'knee_severity': int(kx),
            'bleu_at_knee': round(ky, 2),
            'bleu_at_clean': round(clean_bleu, 2),
            'relative_drop_pct': round(drop, 1),
            'n_points_used': len(sevs),
        })
    rows.sort(key=lambda r: r['knee_severity'])

    path = os.path.join(out_dir, 'knee_points.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f'  Saved {path}')


# OUTPUT 3: Per-Condition Summary Table CSV
def generate_scores_csv(metrics, clean_bleu, out_dir):
    has_bertscore = any(
        v.get('bertscore_f1') is not None
        for cond in MISALIGN_ORDER
        for k, v in metrics.get(cond, {}).items()
        if isinstance(v, dict)
    )
    def _make_row(cond, head_sev, tail_sev, entry):
        b = entry['bleu4']
        drop = (clean_bleu - b) / clean_bleu * 100 if clean_bleu else 0
        row = {
            'misalignment_type': cond,
            'head_severity': head_sev, 'tail_severity': tail_sev,
            'bleu4': b, 'bleu4_drop': f'-{drop:.1f}%',
            'rouge_l': entry.get('rouge_l'),
            'meteor': entry.get('meteor'),
            'ter': entry.get('ter'),
        }
        if has_bertscore: row['bertscore_f1'] = entry.get('bertscore_f1')
        return row
    
    c = metrics['clean']
    rows = [{ # Clean row
        'misalignment_type': 'clean', 
        'head_severity': 0, 'tail_severity': 0, 
        'bleu4': c['bleu4'], 'bleu4_drop': '—',
        'rouge_l': c.get('rouge_l'), 'meteor': c.get('meteor'), 'ter': c.get('ter'),
    }]
    if has_bertscore: rows[0]['bertscore_f1'] = c.get('bertscore_f1')

    for cond in MISALIGN_ORDER:
        cond_data = metrics.get(cond, {})
        if cond in COMPOUND_CONDITIONS: # Expand all 25 (head_sev, tail_sev) entries
            for head_sev in SEVERITY_LEVELS:
                for tail_sev in SEVERITY_LEVELS:
                    entry = cond_data.get(f'h{head_sev}_t{tail_sev}')
                    if entry is not None: rows.append(_make_row(cond, head_sev, tail_sev, entry))
        else:
            for sev in SEVERITY_LEVELS:
                entry = cond_data.get(str(sev))
                if entry is not None: rows.append(_make_row(cond, sev, sev, entry))

    path = os.path.join(out_dir, 'scores.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {path}")


# OUTPUT 4: Sample Translation Comparison (Markdown)
def generate_sample_translations(data, out_dir, n_samples=5):
    translations = data.get('translations', {})
    clean_trans = {t['name']: t for t in translations.get('clean', [])}
    lines = ['# Sample Translation Comparison\n']

    for cond in MISALIGN_ORDER:
        if cond in COMPOUND_CONDITIONS:
            cases = [(20, 20), (10, 40), (40, 10)] # Show diagonal (20%, 20%) and two asymmetric cases
            lines.append(f'\n## {PRETTY[cond]}\n')
            for h_sev, t_sev in cases:
                key = f'{cond}_h{h_sev}_t{t_sev}'
                samples = translations.get(key, [])
                if not samples: continue
                
                lines.append(f'\n### Head={h_sev}%, Tail={t_sev}%\n')
                chosen = random.sample(samples, min(n_samples, len(samples)))
                for s in chosen:
                    name, ref, hyp_mis = s['name'], s['ref'], s['hyp']
                    hyp_clean = clean_trans.get(name, {}).get('hyp', '—')
                    ftype = classify_failure(ref, hyp_mis)
                    lines.append(f'#### Sample: `{name}`\n')
                    lines.append('| | Text |')
                    lines.append('|---|---|')
                    lines.append(f'| **Reference** | {ref} |')
                    lines.append(f'| **Clean output** | {hyp_clean} |')
                    lines.append(f'| **Misaligned output** | {hyp_mis} |')
                    lines.append(f'| **Failure type** | {ftype} |')
                    lines.append('')
        else:
            key = f'{cond}_20'
            samples = translations.get(key, [])
            if not samples: continue
            
            lines.append(f'\n## {PRETTY[cond]} (20%)\n')
            chosen = random.sample(samples, min(n_samples, len(samples)))
            for s in chosen:
                name, ref, hyp_mis = s['name'], s['ref'], s['hyp']
                hyp_clean = clean_trans.get(name, {}).get('hyp', '—')
                ftype = classify_failure(ref, hyp_mis)
                lines.append(f'### Sample: `{name}`\n')
                lines.append('| | Text |')
                lines.append('|---|---|')
                lines.append(f'| **Reference** | {ref} |')
                lines.append(f'| **Clean output** | {hyp_clean} |')
                lines.append(f'| **Misaligned output** | {hyp_mis} |')
                lines.append(f'| **Failure type** | {ftype} |')
                lines.append('')

    path = os.path.join(out_dir, 'sample_translations.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  Saved {path}')
    

# OUTPUT 5: Violin/Box Plots of Per-Sample Score Distributions
def plot_violin_plots(data, out_dir):
    translations = data.get('translations', {})
    if not has_per_sample_scores(data):
        print('  [Skipping — no per-sample sent_bleu data available]')
        return

    # Get clean baseline scores
    clean_scores = get_per_sample_scores(translations, 'clean', 'sent_bleu')
    if not clean_scores:
        print('  [Skipping — no clean sent_bleu data]')
        return
    
    clean_median = np.median(list(clean_scores.values()))
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, cond in enumerate(MISALIGN_ORDER):
        ax, plot_data, plot_labels = axes[idx], [], []
        for sev in SEVERITY_LEVELS:
            if cond in COMPOUND_CONDITIONS:
                pooled = {} # Collect all scores from pooled compound samples
                for head_sev in SEVERITY_LEVELS:
                    for tail_sev in SEVERITY_LEVELS:
                        if max(head_sev, tail_sev) == sev:
                            for s in translations.get(f'{cond}_h{head_sev}_t{tail_sev}', []):
                                v = s.get('sent_bleu')
                                if v is not None: # Average across compound variants per sample
                                    pooled.setdefault(s['name'], []).append(v)
                vals = [float(np.mean(vs)) for vs in pooled.values()]
            else:
                scores = get_per_sample_scores(translations, f'{cond}_{sev}', 'sent_bleu')
                vals = list(scores.values())

            if vals:
                plot_data.append(vals)
                plot_labels.append(f"{sev}%")

        if not plot_data:
            ax.set_title(PRETTY[cond], fontsize=10)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=10, color='grey')
            continue
        
        try: # Try violin plot, fall back to box plot
            parts = ax.violinplot(plot_data, positions=range(len(plot_data)), showmeans=True, showmedians=True)
            for pc in parts.get('bodies', []):
                pc.set_alpha(0.7)
        except Exception:
            ax.boxplot(plot_data, positions=range(len(plot_data)))

        ax.axhline(clean_median, color='green', linestyle='--', linewidth=1, alpha=0.7, label=f'Clean median ({clean_median:.1f})')
        ax.set_xticks(range(len(plot_labels)))
        ax.set_xticklabels(plot_labels)
        ax.set_title(PRETTY[cond], fontsize=10)
        ax.set_ylabel('Sentence BLEU')
        ax.set_xlabel('Max Severity' if cond in COMPOUND_CONDITIONS else 'Severity')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Per-Sample Sentence BLEU Distributions by Misalignment Type\n'
                 '(Compound: per-sample mean across all h×t at max severity)',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'violin_plots.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')
    
    
# OUTPUT 6: Sequence Length vs BLEU Drop Scatter
def plot_length_vs_drop(data, out_dir):
    translations = data.get('translations', {})
    if not has_per_sample_scores(data) or not has_length_data(data):
        print('  [Skipping — requires per-sample sent_bleu and T_original data]')
        return

    clean_scores = get_per_sample_scores(translations, 'clean', 'sent_bleu')
    if not clean_scores:
        print('  [Skipping — no clean sent_bleu data]')
        return

    # Build per-sample T_original lookup from clean translations
    clean_samples = translations.get('clean', [])
    t_original_map = {
        s['name']: s['T_original'] for s in clean_samples 
        if s.get('T_original') is not None
    }
    target_sev = 30  # Representative severity
    fig, ax = plt.subplots(figsize=(10, 7))
    all_x, all_y = [], []

    for cond in MISALIGN_ORDER:
        if cond in COMPOUND_CONDITIONS:
            mis_scores = get_per_sample_scores_compound(translations, cond, target_sev, 'sent_bleu')
        else:
            mis_scores = get_per_sample_scores(translations, f'{cond}_{target_sev}', 'sent_bleu')
        if not mis_scores: continue

        cat = CAT_MAP.get(cond, 'Mixed')
        xs, ys = [], []
        for name, mis_bleu in mis_scores.items():
            if name in clean_scores and name in t_original_map:
                t_orig = t_original_map[name]
                drop = clean_scores[name] - mis_bleu
                xs.append(t_orig)
                ys.append(drop)

        if xs:
            ax.scatter(xs, ys, c=CAT_COLORS[cat], alpha=0.3, s=15, label=f'{cat} ({PRETTY[cond]})')
            all_x.extend(xs)
            all_y.extend(ys)

    if all_x: # Linear regression trend line + Pearson correlation
        r, p = _add_regression_line(ax, np.array(all_x), np.array(all_y))
        if r is not None: ax.plot([], [], ' ', label=f'Pearson r = {r:.3f} (p = {p:.2e})')

    ax.set_xlabel('Original Sequence Length (T_original)', fontsize=11)
    ax.set_ylabel('BLEU Drop (clean - misaligned)', fontsize=11)
    ax.set_title(f'Sequence Length vs BLEU Drop at {target_sev}% Severity\n'
                 f'(Compound: per-sample mean across all h×t with max={target_sev}%)', fontsize=11)
    ax.axhline(0, color='grey', linestyle='-', linewidth=0.5)
    
    # Deduplicate legend entries by label
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    path = os.path.join(out_dir, 'length_vs_drop.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f'  Saved {path}')

    
# OUTPUT 7: Multi-Metric Heatmap Grid
def plot_multi_metric_heatmap(metrics, clean_bleu, out_dir):
    # Determine which metrics are available
    metric_configs = [
        ('bleu4', 'BLEU-4', 'RdYlGn', False),
        ('rouge_l', 'ROUGE-L', 'RdYlGn', False),
        ('meteor', 'METEOR', 'RdYlGn', False),
        ('ter', 'TER', 'RdYlGn_r', True), # Reversed colormap for TER since lower is better
    ]
    has_bertscore = any( # Check if BERTScore is available
        v.get('bertscore_f1') is not None
        for cond in MISALIGN_ORDER
        for sev in SEVERITY_LEVELS
        for v in [metrics.get(cond, {}).get(str(sev), {})]
    )
    if has_bertscore: metric_configs.append(('bertscore_f1', 'BERTScore', 'RdYlGn', False))
    
    # Filter to only metrics that actually have data
    available_configs = []
    for mname, pretty_name, cmap_name, is_reversed in metric_configs:
        mat = metric_matrix(metrics, mname)
        knee_points = []
        
        if not np.all(np.isnan(mat)):
            available_configs.append((mname, pretty_name, cmap_name, is_reversed, mat))
            for ri, cond in enumerate(MISALIGN_ORDER):
                row = mat[ri]
                valid = ~np.isnan(row)
                if valid.any():
                    x = np.array(SEVERITY_LEVELS)[valid]
                    y = row[valid]
                    kx, _ = find_knee(x, y)
                    knee_points.append(int(kx))
                else:
                    knee_points.append(0)

    if not available_configs:
        print('  [Skipping — no metrics available for multi-metric heatmap]')
        return

    n_panels = len(available_configs)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1: axes = [axes]
    row_labels = [PRETTY[c] for c in MISALIGN_ORDER]
    col_labels = [f'{s}%' for s in SEVERITY_LEVELS]

    for panel_idx, (mname, pretty_name, cmap_name, is_reversed, mat) in enumerate(available_configs):
        ax = axes[panel_idx]
        clean_val = metrics['clean'].get(mname) # Get clean baseline for this metric
        if clean_val is not None and clean_val > 0:
            if is_reversed: # TER: lower is better, reversed colormap
                vmin = clean_val * 0.5
                vmax = clean_val * 2.5
            else:
                vmin = max(0, clean_val * 0.1)
                vmax = clean_val
        else:
            vmin = np.nanmin(mat) if not np.all(np.isnan(mat)) else 0
            vmax = np.nanmax(mat) if not np.all(np.isnan(mat)) else 100

        cmap = sns.color_palette(cmap_name, as_cmap=True)
        show_ylabels = (panel_idx == 0)
        sns.heatmap(
            mat, annot=True, fmt='.1f', cmap=cmap, vmin=vmin, vmax=vmax,
            xticklabels=col_labels, yticklabels=row_labels if show_ylabels else False,
            linewidths=0.5, ax=ax, # cbar_kws={'label': pretty_name}
        )
        for idx, knee in enumerate(knee_points): # Knee column on the right
            ax.text(knee / 10 - 0.5, idx + 0.6, '___', ha='center', va='center', fontsize=14, fontweight='bold')
        
        subtitle = f'(Clean = {clean_val:.2f})' if clean_val is not None else ''
        ax.set_title(f'{pretty_name} {subtitle}', fontsize=11)
        ax.set_xlabel('Max Severity')
        if show_ylabels: ax.set_ylabel('')

    fig.suptitle('Multi-Metric Vulnerability Heatmaps with Knee Points Underlined\n'
                 '(Compound rows: mean over all h×t at max severity)', fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'multi_metric_heatmap.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# OUTPUT 8: Radar/Spider Chart at Fixed Severity
def plot_radar_chart(metrics, out_dir):
    target_sev = 30
    metric_names = ['bleu4', 'rouge_l', 'meteor']
    metric_labels = ['BLEU-4', 'ROUGE-L', 'METEOR']

    # Collect values for each condition at target severity
    raw_values = {}  # metric -> list of values (one per condition)
    for mname in metric_names:
        vals = []
        for cond in MISALIGN_ORDER:
            if cond in COMPOUND_CONDITIONS:
                val = _compound_agg_metric(metrics, cond, mname, target_sev)
            else:
                val = metrics.get(cond, {}).get(str(target_sev), {}).get(mname)
            vals.append(val)
        raw_values[mname] = vals

    # Check we have enough data
    has_data = any(v is not None for mname in metric_names for v in raw_values[mname])
    if not has_data:
        print(f'  [Skipping — no metric data available at severity {target_sev}%]')
        return

    # Normalize each metric to [0, 1] based on min/max across all conditions
    normalized = {}  # metric -> list of normalized values
    for mname in metric_names:
        vals = raw_values[mname]
        numeric_vals = [v for v in vals if v is not None]
        if not numeric_vals:
            normalized[mname] = [0.0] * len(MISALIGN_ORDER)
            continue
        
        vmin, vmax = min(numeric_vals), max(numeric_vals)
        rng = vmax - vmin if vmax > vmin else 1.0
        normalized[mname] = [(v - vmin) / rng if v is not None else 0.0 for v in vals]

    # Number of axes = number of conditions
    N = len(MISALIGN_ORDER)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors_radar = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for mi, mname in enumerate(metric_names):
        vals = normalized[mname] + [normalized[mname][0]]  # close polygon
        ax.plot(angles, vals, 'o-', linewidth=2, label=metric_labels[mi], color=colors_radar[mi], markersize=6)
        ax.fill(angles, vals, alpha=0.1, color=colors_radar[mi])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([PRETTY[c] for c in MISALIGN_ORDER], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=7)
    ax.set_title(f'Model Vulnerability Profile at max {target_sev}% Severity\n'
                 f'(Compound: mean over all h×t | Normalized to [0,1] per metric)',
                 fontsize=11, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    path = os.path.join(out_dir, 'radar_chart.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')
    

# OUTPUT 9: Failure Transition Matrix
def compute_failure_transitions(data, out_dir):
    translations = data.get('translations', {})
    csv_rows = [] # Build transitions CSV
    repr_cond = 'head_trunc' # For the heatmap, pick a representative condition

    # Classify all samples for clean
    clean_samples = translations.get('clean', [])
    if not clean_samples:
        print('  [Skipping — no clean translation data]')
        return

    clean_classes = {s['name']: classify_failure(s['ref'], s['hyp']) for s in clean_samples}
    sev_chain = [0] + SEVERITY_LEVELS # Build severity chain: clean -> 10 -> 20 -> ... -> 50

    # Collect all transitions
    repr_matrices = {}  # (from_sev, to_sev) -> transition matrix for repr_cond
    for cond in MISALIGN_ORDER:
        prev_classes = dict(clean_classes)  # start with clean classifications

        for si in range(1, len(sev_chain)):
            from_sev, to_sev = sev_chain[si - 1], sev_chain[si]
            if cond in COMPOUND_CONDITIONS:
                samples = _compound_agg_samples(translations, cond, to_sev)
            else:
                samples = translations.get(f'{cond}_{to_sev}', [])
                
            if not samples: continue
            curr_classes = {s['name']: classify_failure(s['ref'], s['hyp']) for s in samples}

            # For pooled compound samples, a name may appear multiple times;
            # take the majority failure type (most common across h×t variants).
            if cond in COMPOUND_CONDITIONS:
                name_votes = {}
                for s in samples:
                    ft = classify_failure(s['ref'], s['hyp'])
                    name_votes.setdefault(s['name'], []).append(ft)
                curr_classes = {
                    name: max(set(votes), key=votes.count)
                    for name, votes in name_votes.items()
                }
            
            # Build transition counts
            trans_counts = {}
            for name, curr_ft in curr_classes.items():
                prev_ft = prev_classes.get(name)
                if prev_ft is None: continue
                pair = (prev_ft, curr_ft)
                trans_counts[pair] = trans_counts.get(pair, 0) + 1

            for (from_ft, to_ft), count in trans_counts.items():
                csv_rows.append({
                    'condition': cond,
                    'from_severity': from_sev, 'to_severity': to_sev,
                    'from_type': from_ft, 'to_type': to_ft,
                    'count': count,
                })

            if cond == repr_cond: # Store matrix for representative condition
                mat = np.zeros((len(FAILURE_TYPES), len(FAILURE_TYPES)))
                for (from_ft, to_ft), count in trans_counts.items():
                    ri = FAILURE_TYPES.index(from_ft)
                    ci = FAILURE_TYPES.index(to_ft)
                    mat[ri, ci] = count
                repr_matrices[(from_sev, to_sev)] = mat
            prev_classes = curr_classes # Update prev_classes: for compound use majority vote at new sev

    if csv_rows: # Save CSV
        df = pd.DataFrame(csv_rows)
        path_csv = os.path.join(out_dir, 'failure_transitions.csv')
        df.to_csv(path_csv, index=False)
        print(f'  Saved {path_csv}')
    else:
        print('  [Skipping CSV — no transition data]')
        return
    
    if not repr_matrices: # Plot transition heatmaps for representative condition
        print('  [Skipping plot — no transition matrices for representative condition]')
        return

    n_transitions = len(repr_matrices)
    fig, axes = plt.subplots(1, n_transitions, figsize=(5 * n_transitions, 4.5))
    if n_transitions == 1: axes = [axes]

    for ax_idx, ((from_sev, to_sev), mat) in enumerate(sorted(repr_matrices.items())):
        ax = axes[ax_idx]
        # Normalize rows to percentages
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        mat_pct = mat / row_sums * 100

        sns.heatmap(
            mat_pct, annot=True, fmt='.0f', cmap='Blues',
            xticklabels=[ft[:6] for ft in FAILURE_TYPES], 
            yticklabels=[ft[:6] for ft in FAILURE_TYPES],
            ax=ax, vmin=0, vmax=100, linewidths=0.5, cbar_kws={'label': '%'}
        )
        from_label = 'clean' if from_sev == 0 else f'{from_sev}%'
        ax.set_title(f'{from_label} -> {to_sev}%', fontsize=10)
        ax.set_xlabel('To')
        ax.set_ylabel('From')

    fig.suptitle(f'Failure Transition Matrices ({PRETTY[repr_cond]})', fontsize=12, y=1.02)
    fig.tight_layout()
    path_png = os.path.join(out_dir, 'failure_transitions.png')
    fig.savefig(path_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path_png}')


# OUTPUT 10: Statistical Significance Tests
def compute_significance_tests(data, out_dir):
    translations = data.get('translations', {})
    if not has_per_sample_scores(data):
        print('  [Skipping — no per-sample sent_bleu data]')
        return

    clean_scores = get_per_sample_scores(translations, 'clean', 'sent_bleu')
    if not clean_scores:
        print('  [Skipping — no clean sent_bleu data]')
        return

    np.random.seed(42)  # For bootstrap reproducibility
    rows = []
    for cond in MISALIGN_ORDER:
        for sev in SEVERITY_LEVELS:
            if cond in COMPOUND_CONDITIONS:
                mis_scores = get_per_sample_scores_compound(translations, cond, sev, 'sent_bleu')
            else:
                mis_scores = get_per_sample_scores(translations, f'{cond}_{sev}', 'sent_bleu')
            if not mis_scores: continue

            # Get paired samples (only samples present in both clean and misaligned)
            paired_names = sorted(set(clean_scores.keys()) & set(mis_scores.keys()))
            if len(paired_names) < 5: continue

            clean_arr = np.array([clean_scores[n] for n in paired_names])
            mis_arr = np.array([mis_scores[n] for n in paired_names])
            diffs = clean_arr - mis_arr
            mean_drop = np.mean(diffs)

            # Wilcoxon signed-rank test
            nonzero = diffs[diffs != 0] # Remove zero differences for Wilcoxon
            p_value = scipy_stats.wilcoxon(nonzero).pvalue if len(nonzero) > 0 else 1.0

            # Paired bootstrap CI (1000 resamples)
            n_boot = 1000
            boot_means = np.empty(n_boot)
            n_samples = len(diffs)
            boot_means = np.array([
                np.mean(diffs[np.random.randint(0, n_samples, size=n_samples)])
                for _ in range(n_boot)
            ])
            ci_lower = np.percentile(boot_means, 2.5)
            ci_upper = np.percentile(boot_means, 97.5)

            rows.append({
                'condition': cond, 'severity': sev,
                'n_paired': len(paired_names), 'mean_drop': round(mean_drop, 4),
                'ci_lower': round(ci_lower, 4), 'ci_upper': round(ci_upper, 4),
                'wilcoxon_p': round(p_value, 6) if np.isfinite(p_value) else 'NA',
                'significant_at_005': 'yes' if (np.isfinite(p_value) and p_value < 0.05) else 'no',
                'compound_agg': 'max_sev_mean' if cond in COMPOUND_CONDITIONS else 'single',
            })

    if not rows:
        print('  [Skipping — no paired data for significance tests]')
        return

    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, 'significance_tests.csv')
    df.to_csv(path, index=False)
    print(f'  Saved {path}')


# OUTPUT 11: Truncation vs Contamination Comparison
def plot_trunc_vs_contam(metrics, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    categories = {'Truncation': TRUNC_ONLY, 'Contamination': CONTAM_ONLY, 'Mixed': MIXED}
    cat_means = {}
    
    for cat_name, conds in categories.items(): # Compute average BLEU-4 for each category at each severity
        means = []
        for sev in SEVERITY_LEVELS:
            vals = []
            for cond in conds:
                if cond in COMPOUND_CONDITIONS:
                    val = _compound_agg_metric(metrics, cond, 'bleu4', sev)
                else:
                    val = metrics.get(cond, {}).get(str(sev), {}).get('bleu4')
                if val is not None: vals.append(val)
            means.append(np.mean(vals) if vals else np.nan)
        cat_means[cat_name] = means

    # Panel 1: Category comparison (grouped bar)
    ax1 = axes[0] 
    bar_width = 0.25
    x = np.arange(len(SEVERITY_LEVELS))

    for ci, (cat_name, means) in enumerate(cat_means.items()):
        offset = (ci - 1) * bar_width
        ax1.bar(x + offset, means, bar_width, label=cat_name, color=CAT_COLORS[cat_name], alpha=0.8)

    ax1.set_xlabel('Max Severity (%)', fontsize=11)
    ax1.set_ylabel('Average BLEU-4', fontsize=11)
    ax1.set_title('Truncation vs Contamination vs Mixed\n'
                  '(Compound: mean over all h×t at max severity)', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{s}%' for s in SEVERITY_LEVELS])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Head-only vs Tail-only comparison
    ax2 = axes[1]
    head_conds = ['head_trunc', 'head_contam']
    tail_conds = ['tail_trunc', 'tail_contam']

    for label, conds, color in [('Head-only', head_conds, '#e377c2'), ('Tail-only', tail_conds, '#7f7f7f')]:
        means = []
        for sev in SEVERITY_LEVELS:
            vals = []
            for cond in conds:
                val = metrics.get(cond, {}).get(str(sev), {}).get('bleu4')
                if val is not None: vals.append(val)
            means.append(np.mean(vals) if vals else np.nan)

        offset = -0.15 if label == 'Head-only' else 0.15
        ax2.bar(x + offset, means, 0.3, label=label, color=color, alpha=0.8)

    ax2.set_xlabel('Severity (%)', fontsize=11)
    ax2.set_ylabel('Average BLEU-4', fontsize=11)
    ax2.set_title('Head-Only vs Tail-Only Misalignment', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{s}%' for s in SEVERITY_LEVELS])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    path = os.path.join(out_dir, 'trunc_vs_contam.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f'  Saved {path}')


# OUTPUT 12: Sample Robustness Analysis
def compute_sample_robustness(data, out_dir):
    translations = data.get('translations', {})
    clean_samples = translations.get('clean', [])
    if not clean_samples:
        print('  [Skipping — no clean translation data]')
        return

    # Build sample info from clean
    sample_info = {s['name']: {
        'name': s['name'], 'T_original': s.get('T_original')
    } for s in clean_samples}

    # For each sample, find max severity where it remains acceptable across
    # all single-type conditions
    single_conds = ['head_trunc', 'tail_trunc', 'head_contam', 'tail_contam']
    rows = []

    for name, info in sample_info.items():
        fails_first_cond, fails_first_sev = None, None
        for cond in single_conds:
            for sev in SEVERITY_LEVELS:
                samples = translations.get(f'{cond}_{sev}', [])
                sample_entry = next((s for s in samples if s['name'] == name), None)
                if sample_entry is None: continue

                ft = classify_failure(sample_entry['ref'], sample_entry['hyp'])
                if ft != 'acceptable':
                    if fails_first_sev is None or sev < fails_first_sev:
                        fails_first_sev, fails_first_cond = sev, cond
                    break  # This condition fails at this severity, move to next cond

        if fails_first_sev is not None:
            # Robustness = severity level just before failure
            # If fails at 10%, robustness = 0 (fails at the lowest)
            sev_idx = SEVERITY_LEVELS.index(fails_first_sev) if fails_first_sev in SEVERITY_LEVELS else 0
            robustness = SEVERITY_LEVELS[sev_idx - 1] if sev_idx > 0 else 0
        else:
            robustness = 50  # Never fails

        rows.append({
            'sample_name': name, 'T_original': info['T_original'], 'robustness_score': robustness,
            'fails_first_at_condition': fails_first_cond if fails_first_cond else 'none',
            'fails_first_at_severity': fails_first_sev if fails_first_sev else 'none',
        })

    if not rows:
        print('  [Skipping — no robustness data computed]')
        return

    df = pd.DataFrame(rows)
    path_csv = os.path.join(out_dir, 'sample_robustness.csv')
    df.to_csv(path_csv, index=False)
    print(f'  Saved {path_csv}')

    # Plot: histogram + scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of robustness scores
    ax1 = axes[0]
    rob_scores = df['robustness_score'].values
    bins = [-5] + [s - 2.5 for s in SEVERITY_LEVELS] + [52.5]
    ax1.hist(rob_scores, bins=bins, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.set_xlabel('Robustness Score (max severity before failure)', fontsize=10)
    ax1.set_ylabel('Number of Samples', fontsize=10)
    ax1.set_title('Distribution of Sample Robustness Scores', fontsize=11)
    ax1.set_xticks([0] + SEVERITY_LEVELS)
    ax1.set_xticklabels(['0'] + [f'{s}%' for s in SEVERITY_LEVELS])
    ax1.grid(True, alpha=0.3, axis='y')

    # Scatter: T_original vs robustness (if T_original available)
    ax2 = axes[1]
    t_orig = df['T_original'].dropna()
    if len(t_orig) > 0:
        valid_df = df.dropna(subset=['T_original'])
        ax2.scatter(valid_df['T_original'], valid_df['robustness_score'], alpha=0.4, s=20, color='steelblue')
        ax2.set_xlabel('Original Sequence Length (T_original)', fontsize=10)
        ax2.set_ylabel('Robustness Score', fontsize=10)
        ax2.set_title('Sequence Length vs Robustness', fontsize=11)

        # Correlation + trend line
        x_arr = valid_df['T_original'].values.astype(float)
        y_arr = valid_df['robustness_score'].values.astype(float)
        r, p = _add_regression_line(ax2, x_arr, y_arr)
        if r is not None: ax2.set_title(f'Sequence Length vs Robustness (r={r:.3f})', fontsize=11)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No T_original data', ha='center', va='center', transform=ax2.transAxes, fontsize=12, color='grey')
        ax2.set_title('Sequence Length vs Robustness', fontsize=11)

    fig.tight_layout()
    path_png = os.path.join(out_dir, 'sample_robustness.png')
    fig.savefig(path_png, dpi=200)
    plt.close(fig)
    print(f'  Saved {path_png}')


# OUTPUT 13: Sensitivity Ranking
def compute_sensitivity_ranking(metrics, out_dir):
    '''Rank conditions by slope of BLEU-4 vs severity regression.

    For compound conditions all 25 (h,t) data points are used, with x = max(h,t),
    giving a more robust regression than the 5 diagonal points alone.
    '''
    rows = []
    for cond in MISALIGN_ORDER:
        cond_data = metrics.get(cond, {})
        sevs, vals = [], []
        
        for head_sev in SEVERITY_LEVELS: # Use all 25 entries; x = max(head_sev, tail_sev)
            for tail_sev in SEVERITY_LEVELS:
                val = cond_data.get(f'h{head_sev}_t{tail_sev}', {}).get('bleu4')
                if val is not None:
                    sevs.append(max(head_sev, tail_sev))
                    vals.append(val)
        else:
            for sev in SEVERITY_LEVELS:
                val = cond_data.get(str(sev), {}).get('bleu4')
                if val is not None:
                    sevs.append(sev)
                    vals.append(val)
                    
        if len(sevs) < 2: continue
        x = np.array(sevs, dtype=float)
        y = np.array(vals, dtype=float)

        # Linear regression: y = slope * x + intercept
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]

        # R-squared
        y_pred = np.poly1d(coeffs)(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rows.append({'condition': cond, 'slope': round(slope, 4), 'r_squared': round(r_squared, 4), 'n_points': len(sevs)})

    if not rows:
        print('  [Skipping — insufficient data for sensitivity ranking]')
        return

    # Rank by slope (most negative = most sensitive = rank 1)
    rows.sort(key=lambda r: r['slope'])
    for rank, row in enumerate(rows, 1):
        row['rank'] = rank

    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, 'sensitivity_ranking.csv')
    df.to_csv(path, index=False)
    print(f'  Saved {path}')


# OUTPUT 14: Failure Distribution — Small Multiples
def plot_failure_distribution(data, out_dir):
    translations = data.get('translations', {})
    fig, axes = plt.subplots(2, 4, figsize=(20, 20))
    axes = axes.flatten()
    any_data = False

    for idx, cond in enumerate(MISALIGN_ORDER):
        ax, has_cond_data = axes[idx], False
        sev_labels, sev_counts = [], {ft: [] for ft in FAILURE_TYPES}

        for sev in SEVERITY_LEVELS:
            if cond in COMPOUND_CONDITIONS:
                samples = _compound_agg_samples(translations, cond, sev)
            else:
                samples = translations.get(f'{cond}_{sev}', [])
                
            if not samples: continue
            has_cond_data = True
            dist = _count_failures(samples)
            total = len(samples)
            sev_labels.append(f'{sev}%')
            for ft in FAILURE_TYPES:
                sev_counts[ft].append(dist[ft] / total * 100)

        if not has_cond_data:
            ax.set_title(PRETTY[cond], fontsize=10)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=10, color='grey')
            continue

        any_data = True
        x = np.arange(len(sev_labels))
        _draw_stacked_bars(ax, x, sev_counts, width=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(sev_labels, fontsize=9)
        ax.set_title(PRETTY[cond], fontsize=10)
        ax.set_ylim(0, 105)
        ax.set_ylabel('% of samples' if idx % 4 == 0 else '')
        ax.set_xlabel('Max Severity' if cond in COMPOUND_CONDITIONS else 'Severity')
        ax.grid(True, alpha=0.2, axis='y')

    if not any_data:
        plt.close(fig)
        print('  [Skipping — no per-sample translations]')
        return

    # Add shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAILURE_COLOURS[ft]) for ft in FAILURE_TYPES]
    fig.legend(
        handles, [ft.capitalize() for ft in FAILURE_TYPES],
        loc='lower center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.02)
    )
    fig.suptitle('Failure-Type Distribution by Condition — Small Multiples\n'
                 '(Compound bars pool all h×t at max severity)', fontsize=14, y=1.02)
    
    fig.tight_layout()
    path = os.path.join(out_dir, 'failure_distribution.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# OUTPUT 15: Compound Severity Grid (full h × t heatmap per type)
def plot_compound_severity_grid(metrics, clean_bleu, out_dir):
    '''2×2 layout of 5×5 heatmaps — one per compound condition type.

    Rows = head severity, cols = tail severity.  Every cell is annotated
    with the BLEU-4 score and its relative drop from the clean baseline.
    Diagonal cells (h == t) are outlined in black so they can be cross-
    referenced with the main 8×5 heatmap (Output 1).
    '''
    sev_labels = [f'{s}%' for s in SEVERITY_LEVELS]
    cmap = sns.color_palette('RdYlGn', as_cmap=True)
    vmin = max(0, clean_bleu * 0.1)
    vmax = clean_bleu

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    any_data = False

    for ax_idx, cond in enumerate(COMPOUND_CONDITIONS):
        ax = axes.flat[ax_idx]
        mat = compound_2d_matrix(metrics, cond, 'bleu4')

        if np.all(np.isnan(mat)):
            ax.set_title(PRETTY[cond], fontsize=11)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=12, color='grey')
            continue
        
        any_data = True
        annot = np.empty_like(mat, dtype=object)
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                v = mat[r, c]
                if np.isnan(v): annot[r, c] = ''
                else:
                    drop = (clean_bleu - v) / clean_bleu * 100 if clean_bleu > 0 else 0
                    annot[r, c] = f'{v:.1f}\n({drop:+.0f}%)'

        sns.heatmap(
            mat, annot=annot, fmt='', cmap=cmap, 
            xticklabels=sev_labels, yticklabels=sev_labels,
            vmin=vmin, vmax=vmax, annot_kws={'fontsize': 8},
            linewidths=0.5, ax=ax, cbar_kws={'label': 'BLEU-4'},
        )
        for d in range(len(SEVERITY_LEVELS)): # Black border on diagonal cells (symmetric h == t)
            ax.add_patch(plt.Rectangle((d, d), 1, 1, fill=False, edgecolor='black', lw=2.5, zorder=5))

        ax.set_title(PRETTY[cond], fontsize=11)
        ax.set_xlabel('Tail Severity', fontsize=10)
        ax.set_ylabel('Head Severity', fontsize=10)

    if not any_data:
        plt.close(fig)
        print('  [Skipping — no compound condition data]')
        return

    fig.suptitle(
        f'Compound Condition BLEU-4: Head × Tail Severity Grid\n'
        f'(Clean = {clean_bleu:.2f} | Black border = symmetric h == t)',
        fontsize=14, y=1.01)
    
    fig.tight_layout()
    path = os.path.join(out_dir, 'compound_severity_grid.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# OUTPUT 16: Compound Interaction Analysis
def plot_compound_interaction(metrics, clean_bleu, out_dir):
    '''Consolidated 2×2 analysis of compound-condition interaction effects.

    (a) Additivity — predicted (sum of individual drops) vs actual compound
        drop.  Points above y = x are "superadditive" (compound is worse
        than the sum of its parts); below are *subadditive*.
    (b) Head ↔ Tail asymmetry — for symmetric-mechanism types (HT+TT,
        HC+TC) compares score(h=low, t=high) vs score(h=high, t=low).
    (c) Mean interaction-strength heatmap (actual_drop − predicted_drop),
        averaged across all four compound types.
    (d) Mirror-type comparison — HT+TC vs HC+TT at the same (h, t) pair.
        Tests whether it matters which end is truncated vs contaminated.

    Also writes compound_analysis.csv with per-entry additivity data.
    '''
    # ---- cache single-condition drops --------------------------------
    single_drops = {}
    for single in ('head_trunc', 'tail_trunc', 'head_contam', 'tail_contam'):
        single_drops[single] = {}
        for sev in SEVERITY_LEVELS:
            bleu = metrics.get(single, {}).get(str(sev), {}).get('bleu4')
            if bleu is not None: single_drops[single][sev] = clean_bleu - bleu

    # ---- accumulate compound data ------------------------------------
    csv_rows, additivity_pts = [], []
    interaction_sum = np.zeros((len(SEVERITY_LEVELS), len(SEVERITY_LEVELS)))
    interaction_cnt = np.zeros((len(SEVERITY_LEVELS), len(SEVERITY_LEVELS)))

    for cond in COMPOUND_CONDITIONS:
        head_type, tail_type = COMPOUND_TO_SINGLES[cond]
        cond_data = metrics.get(cond, {})

        for ri, head_sev in enumerate(SEVERITY_LEVELS):
            for ci, tail_sev in enumerate(SEVERITY_LEVELS):
                entry = cond_data.get(f'h{head_sev}_t{tail_sev}', {})
                compound_bleu = entry.get('bleu4')
                if compound_bleu is None: continue

                actual_drop = clean_bleu - compound_bleu
                drop_h = single_drops.get(head_type, {}).get(head_sev)
                drop_t = single_drops.get(tail_type, {}).get(tail_sev)
                predicted = interaction = None
                
                if drop_h is not None and drop_t is not None:
                    predicted = drop_h + drop_t
                    interaction = actual_drop - predicted
                    additivity_pts.append((predicted, actual_drop, cond))
                    interaction_sum[ri, ci] += interaction
                    interaction_cnt[ri, ci] += 1

                csv_rows.append({
                    'compound_type': cond, 'head_sev': head_sev, 'tail_sev': tail_sev,
                    'bleu4': round(compound_bleu, 2),
                    'drop_from_clean': round(actual_drop, 2),
                    'single_head_bleu4': (round(clean_bleu - drop_h, 2) if drop_h is not None else None),
                    'single_tail_bleu4': (round(clean_bleu - drop_t, 2) if drop_t is not None else None),
                    'predicted_additive_drop': (round(predicted, 2) if predicted is not None else None),
                    'interaction_term': (round(interaction, 2) if interaction is not None else None),
                    'rouge_l': entry.get('rouge_l'), 'meteor': entry.get('meteor'), 'ter': entry.get('ter'),
                    'bertscore_f1': entry.get('bertscore_f1'),
                })

    valid_i = interaction_cnt > 0
    interaction_avg = np.full_like(interaction_sum, np.nan)
    if valid_i.any():
        interaction_avg[valid_i] = interaction_sum[valid_i] / interaction_cnt[valid_i]

    # ---- asymmetry data ----------------------------------------------
    asym_pts = []
    for cond in ('head_trunc_tail_trunc', 'head_contam_tail_contam'):
        cond_data = metrics.get(cond, {})
        for i, a in enumerate(SEVERITY_LEVELS):
            for b in SEVERITY_LEVELS[i + 1:]:
                bleu_lt = cond_data.get(f'h{a}_t{b}', {}).get('bleu4')
                bleu_hl = cond_data.get(f'h{b}_t{a}', {}).get('bleu4')
                if bleu_lt is not None and bleu_hl is not None:
                    asym_pts.append((bleu_lt, bleu_hl, cond))

    # ---- mirror-type data --------------------------------------------
    mirror_pts = []
    d_a = metrics.get('head_trunc_tail_contam', {})
    d_b = metrics.get('head_contam_tail_trunc', {})
    for head_sev in SEVERITY_LEVELS:
        for tail_sev in SEVERITY_LEVELS:
            sa = d_a.get(f'h{head_sev}_t{tail_sev}', {}).get('bleu4')
            sb = d_b.get(f'h{head_sev}_t{tail_sev}', {}).get('bleu4')
            if sa is not None and sb is not None:
                mirror_pts.append((sa, sb))

    has_data = additivity_pts or asym_pts or mirror_pts or valid_i.any()
    if not has_data:
        print('  [Skipping — no compound data for interaction analysis]')
        _save_compound_csv(csv_rows, out_dir)
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    cond_palette = {
        'head_trunc_tail_trunc':   '#1f77b4', 'head_trunc_tail_contam':  '#ff7f0e',
        'head_contam_tail_trunc':  '#2ca02c', 'head_contam_tail_contam': '#d62728',
    }

    # (a) Additivity scatter
    ax = axes[0, 0]
    if additivity_pts:
        for pred, actual, cond in additivity_pts:
            ax.scatter(pred, actual, c=cond_palette[cond], s=25, alpha=0.55, label=PRETTY[cond])

        all_v = [v for p, a, _ in additivity_pts for v in (p, a)]
        lo, hi = min(all_v) - 0.5, max(all_v) + 0.5
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5)
        ax.fill_between([lo, hi], [lo, hi], hi, alpha=0.06, color='red')
        ax.fill_between([lo, hi], lo, [lo, hi], alpha=0.06, color='blue')
        ax.text(0.97, 0.03, 'Subadditive', transform=ax.transAxes, ha='right',
                va='bottom', fontsize=8, color='#1565C0', style='italic')
        ax.text(0.03, 0.97, 'Superadditive', transform=ax.transAxes, ha='left',
                va='top', fontsize=8, color='#C62828', style='italic')

        preds = np.array([p for p, a, _ in additivity_pts])
        actuals = np.array([a for p, a, _ in additivity_pts])
        mean_int = np.mean(actuals - preds)
        r, p = scipy_stats.pearsonr(preds, actuals)
        info = f'Mean interaction = {mean_int:+.2f}\nr = {r:.3f}, p = {p:.2e}'
        ax.text(0.97, 0.15, info, transform=ax.transAxes, fontsize=7,
                ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

        ax.set_xlabel('Predicted Additive Drop', fontsize=10)
        ax.set_ylabel('Actual Compound Drop', fontsize=10)
        handles, labels_leg = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_leg, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=6, loc='lower right')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=12, color='grey')
    ax.set_title('(a) Additivity Analysis', fontsize=11)
    ax.grid(True, alpha=0.3)

    # (b) Head ↔ Tail asymmetry scatter
    ax = axes[0, 1]
    if asym_pts:
        for bleu_lt, bleu_hl, cond in asym_pts:
            ax.scatter(bleu_lt, bleu_hl, c=cond_palette[cond], s=50, alpha=0.7, label=PRETTY[cond])

        all_v = [v for a, b, _ in asym_pts for v in (a, b)]
        lo, hi = min(all_v) - 0.5, max(all_v) + 0.5
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_xlabel('BLEU-4 (head = low sev, tail = high sev)', fontsize=9)
        ax.set_ylabel('BLEU-4 (head = high sev, tail = low sev)', fontsize=9)

        n_above = sum(1 for a, b, _ in asym_pts if b > a)
        n_below = sum(1 for a, b, _ in asym_pts if b < a)
        direction = ('Tail more sensitive' if n_above > n_below
                     else 'Head more sensitive' if n_below > n_above
                     else 'Symmetric')
        ax.text(0.03, 0.97,
                f'{direction}\n({n_above} above / {n_below} below y = x)',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

        handles, labels_leg = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_leg, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc='lower right')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=12, color='grey')
    ax.set_title('(b) Head ↔ Tail Asymmetry (Symmetric Types)', fontsize=11)
    ax.grid(True, alpha=0.3)

    # (c) Interaction strength heatmap
    ax = axes[1, 0]
    if valid_i.any():
        sev_labels = [f'{s}%' for s in SEVERITY_LEVELS]
        vabs = max(np.nanmax(np.abs(interaction_avg)), 0.1)
        sns.heatmap(
            interaction_avg, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-vabs, vmax=vabs,
            xticklabels=sev_labels, yticklabels=sev_labels,
            linewidths=0.5, ax=ax, annot_kws={'fontsize': 9},
            cbar_kws={'label': 'Interaction (+ = superadditive)'},
        )
        ax.set_xlabel('Tail Severity', fontsize=10)
        ax.set_ylabel('Head Severity', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=12, color='grey')
    ax.set_title('(c) Mean Interaction Strength\n(Actual Drop − Predicted Additive Drop)', fontsize=11)

    # (d) Mirror-type comparison
    ax = axes[1, 1]
    if mirror_pts:
        xs_m = np.array([m[0] for m in mirror_pts])
        ys_m = np.array([m[1] for m in mirror_pts])
        abs_d = np.abs(xs_m - ys_m)
        sc = ax.scatter(xs_m, ys_m, c=abs_d, cmap='plasma', s=45, alpha=0.7, zorder=5, vmin=0, vmax=max(abs_d.max(), 0.1))
        plt.colorbar(sc, ax=ax, label='|Δ BLEU-4|')

        lo = min(xs_m.min(), ys_m.min()) - 0.5
        hi = max(xs_m.max(), ys_m.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_xlabel(f"BLEU-4 ({PRETTY['head_trunc_tail_contam']})", fontsize=9)
        ax.set_ylabel(f"BLEU-4 ({PRETTY['head_contam_tail_trunc']})", fontsize=9)

        n_a = int(np.sum(ys_m > xs_m))
        n_b = int(np.sum(xs_m > ys_m))
        dir_label = ('HC + TT less harmful' if n_a > n_b
                     else 'HT + TC less harmful' if n_b > n_a
                     else 'Equivalent')
        ax.text(0.03, 0.97, dir_label, transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='grey')
    ax.set_title('(d) Mirror-Type Comparison\n(HT + TC vs HC + TT)', fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Compound Condition Interaction Analysis', fontsize=14, y=1.01)
    fig.tight_layout()
    path_png = os.path.join(out_dir, 'compound_interaction.png')
    fig.savefig(path_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path_png}')
    _save_compound_csv(csv_rows, out_dir)


def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('--results', default='results/benchmark_results.json')
    parser.add_argument('--outdir', default='results')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    results_path = os.path.join(SCRIPT_DIR, args.results)
    out_dir = os.path.join(SCRIPT_DIR, args.outdir)
    os.makedirs(out_dir, exist_ok=True)

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        metrics = data['metrics']
        clean_bleu = metrics['clean']['bleu4']

    print(f'Clean BLEU-4: {clean_bleu:.2f}')
    print(f'Generating analysis outputs in {out_dir}/\n')

    print("[1/16] Degradation curves")
    plot_degradation_curves(metrics, clean_bleu, out_dir)

    print("[2/16] Knee-point analysis")
    compute_knee_points(metrics, clean_bleu, out_dir)

    print("[3/16] Per-condition scores CSV")
    generate_scores_csv(metrics, clean_bleu, out_dir)

    print("[4/16] Sample translation comparison")
    generate_sample_translations(data, out_dir)

    print("[5/16] Violin plots")
    plot_violin_plots(data, out_dir)

    print("[6/16] Length vs drop scatter")
    plot_length_vs_drop(data, out_dir)

    print("[7/16] Multi-metric heatmap grid")
    plot_multi_metric_heatmap(metrics, clean_bleu, out_dir)

    print("[8/16] Radar chart")
    plot_radar_chart(metrics, out_dir)

    print("[9/16] Failure transitions")
    compute_failure_transitions(data, out_dir)

    print("[10/16] Statistical significance tests")
    compute_significance_tests(data, out_dir)

    print("[11/16] Truncation vs contamination comparison")
    plot_trunc_vs_contam(metrics, out_dir)

    print("[12/16] Sample robustness analysis")
    compute_sample_robustness(data, out_dir)

    print("[13/16] Sensitivity ranking")
    compute_sensitivity_ranking(metrics, out_dir)

    print("[14/16] Failure distribution (detailed small multiples)")
    plot_failure_distribution(data, out_dir)

    print("[15/16] Compound severity grid")
    plot_compound_severity_grid(metrics, clean_bleu, out_dir)

    print("[16/16] Compound interaction analysis")
    plot_compound_interaction(metrics, clean_bleu, out_dir)


if __name__ == '__main__':
    main()