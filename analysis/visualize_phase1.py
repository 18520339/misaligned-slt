'''Visualization for misalignment analysis.
• Every figure covers BOTH basic (HT, TT, HC, TC) AND compound conditions
  where applicable (compound means are added as dashed overlay curves).
• Two core metrics are shown consistently in every figure:
    – BLEU-4  (↑ better)    – ROUGE-L (↑ better)
• Combined / multi-metric views: metrics shown as *relative degradation %*
  from clean baseline (positive = more degraded).
  Formula:  ΔM% = (clean−val)/|clean|×100  for BLEU/ROUGE
• Figures 6 and 9 are completely redesigned per reviewer feedback:
    – Fig 6: per-sample failure-mode transition matrices (clean → each severity).
    – Fig 9: scatter of predicted-additive vs actual compound drop, revealing
              super-/sub-additive interaction for every compound condition and severity in the dataset.
• Fig 10 ranks ALL conditions (basic + compound) by BLEU drop with multi-metric bars.
'''
import os, json
import numpy as np
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis.failure_modes import (
    classify_all_predictions, compute_transition_matrix, failure_mode_distribution,
    FAILURE_COLORS, FAILURE_MODES,
)
from analysis.tables import _canonical_pair, _parse_compound_name
from analysis.knee_point import compute_degradation_rate, detect_all_knee_points

# Constants
BASIC_ORDER = ['HT', 'TT', 'HC', 'TC']
CONDITION_COLORS = {'HT': '#e74c3c', 'TT': '#3498db', 'HC': '#e67e22', 'TC': '#2ecc71'}
CONDITION_MARKERS = {'HT': 'o', 'TT': 's', 'HC': '^', 'TC': 'D'}
CONDITION_LABELS = {
    'HT': 'Head Truncation (HT)', 'TT': 'Tail Truncation (TT)',
    'HC': 'Head Contamination (HC)', 'TC': 'Tail Contamination (TC)',
}
COMPOUND_PAIR_COLORS = { # Canonical compound-pair naming: lower basic-order index first
    'HT+TT': '#8e44ad', 'HT+HC': '#1abc9c', 'HT+TC': '#d35400', 'TT+HC': '#2980b9', 'TT+TC': '#c0392b', 'HC+TC': '#16a085',
    # reverse aliases
    'TT+HT': '#8e44ad', 'HC+HT': '#1abc9c', 'TC+HT': '#d35400', 'HC+TT': '#2980b9', 'TC+TT': '#c0392b', 'TC+HC': '#16a085',
}
COMPOUND_PAIR_LABELS = {
    'HT+TT': 'HT + TT', 'HT+HC': 'HT + HC', 'HT+TC': 'HT + TC',
    'TT+HC': 'TT + HC', 'TT+TC': 'TT + TC', 'HC+TC': 'HC + TC',
}
MODE_ABBREV = {
    'Acceptable': 'Acc', 'Partial match': 'Part', 'Under-generation': 'Under',
    'Hallucination': 'Hall', 'Repetition': 'Rep', 'Incoherent': 'Incoh',
}
# Four-mode collapse used in transition matrices:
#   Partial match  → Hallucination   (both reflect wrong content)
#   Repetition     → Incoherent      (both reflect output quality failure)
FOUR_MODES = ['Acceptable', 'Under-generation', 'Hallucination', 'Incoherent']
FOUR_MODE_ABBREV = {
    'Acceptable': 'Acc', 'Under-generation': 'Under',
    'Hallucination': 'Hall', 'Incoherent': 'Incoh',
}
METRIC_LABELS = {'bleu4': 'BLEU-4', 'rouge_l': 'ROUGE-L'}
METRIC_HIGHER_IS_BETTER = {'bleu4': True, 'rouge_l': True}

matplotlib.use('Agg')
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})

def _save_fig(fig, output_dir, name):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f'{name}.png'))
    plt.close(fig)

def _extract_basic_curves(results, severity_levels, metric='bleu4'):
    # Dict ctype -> [metric value or None] aligned to severity_levels
    curves = {}
    for ctype in BASIC_ORDER:
        vals = []
        for sev in severity_levels:
            cond = f'{ctype}_{int(sev * 100):02d}'
            m = results.get(cond, {}).get('metrics', {})
            vals.append(m.get(metric))
        curves[ctype] = vals
    return curves

def _relative_degradation(clean_val, val, metric): # Relative degradation from clean (%). Positive always = more degraded
    if not clean_val: return 0.0
    if METRIC_HIGHER_IS_BETTER.get(metric, True): return (clean_val - val) / abs(clean_val) * 100
    return (val - clean_val) / abs(clean_val) * 100

def _pick_severities(severity_levels, n=3): # Pick n evenly-spaced representative severities from the available list
    if len(severity_levels) <= n: return list(severity_levels)
    idx = np.linspace(0, len(severity_levels) - 1, n, dtype=int)
    return [severity_levels[i] for i in idx]

def _remap_to_4_modes(mode):
    '''Collapse 6 failure modes to 4.

    Partial match  → Hallucination  (wrong content, regardless of fluency)
    Repetition     → Incoherent     (output-quality failure)
    Acceptable / Under-generation / Hallucination / Incoherent → unchanged
    '''
    if mode == 'Partial match': return 'Hallucination'
    if mode == 'Repetition': return 'Incoherent'
    return mode

def _extract_compound_mean_curves(results, metric='bleu4'):
    # Mean metric per (canonical_pair, total_severity) across matching compound conditions
    grouped = defaultdict(lambda: defaultdict(list))
    for cond_name, cond_data in results.items():
        if '+' not in cond_name or cond_name in ('clean', 'meta'): continue
        parsed = _parse_compound_name(cond_name)
        if parsed is None: continue
        a_type, a_sev, b_type, b_sev = parsed
        pair = _canonical_pair(a_type, b_type)
        total = round(a_sev + b_sev, 4)
        val = cond_data.get('metrics', {}).get(metric)
        if val is not None: grouped[pair][total].append(val)

    curves = {}
    for pair, sev_map in grouped.items():
        items = sorted(sev_map.items())
        curves[pair] = {'totals_pct': [x * 100 for x, _ in items], 'means': [float(np.mean(v)) for _, v in items]}
    return curves # {pair_key: {'totals_pct': [...], 'means': [...]}}


def _get_condition_samples(results, cond_name):
    samples = results.get(cond_name, {}).get('predictions', {})
    return samples if isinstance(samples, dict) else {}


# Figure 1: BLEU-4 Degradation — basic conditions + compound mean bands
def fig01_bleu_degradation(results, severity_levels, output_dir, clean_bleu=None):
    '''BLEU-4 degradation curves.

    Basic conditions (4 solid lines with markers) annotated with knee points.
    Compound conditions shown as mean lines grouped by pair type (dashed, lighter),
    plotted at total severity budget sa + sb on the same x-axis.
    '''
    curves        = _extract_basic_curves(results, severity_levels, 'bleu4')
    compound_crvs = _extract_compound_mean_curves(results, 'bleu4')
    knees         = detect_all_knee_points(results, severity_levels)

    if clean_bleu is None and 'clean' in results:
        clean_bleu = results['clean']['metrics'].get('bleu4', 0)

    fig, ax = plt.subplots(figsize=(11, 6))
    sevs_pct = [s * 100 for s in severity_levels]

    # ── basic conditions ──────────────────────────────────────────────────────
    for ctype in BASIC_ORDER:
        vals  = curves[ctype]
        valid = [(s, v) for s, v in zip(sevs_pct, vals) if v is not None]
        if not valid: continue
        xs, ys = zip(*valid)
        ax.plot(
            xs, ys, color=CONDITION_COLORS[ctype], marker=CONDITION_MARKERS[ctype],
            label=CONDITION_LABELS[ctype], linewidth=2.5, markersize=8, zorder=3
        )

        # knee point star
        k = (knees.get(ctype, {}) or {}).get('bleu4')
        if k and k.get('knee_severity'):
            kx, ky = k['knee_severity'] * 100, k['knee_value']
            ax.axvline(kx, color=CONDITION_COLORS[ctype], linestyle=':', alpha=0.45, linewidth=1.2)
            ax.plot(
                kx, ky, '*', color=CONDITION_COLORS[ctype], markersize=18, 
                zorder=6, markeredgecolor='white', markeredgewidth=0.7
            )
            ax.annotate(
                f'knee ({kx:.0f}, {ky:.1f})', xy=(kx, ky), xytext=(7, -10),
                textcoords='offset points', fontsize=10, color=CONDITION_COLORS[ctype],
                arrowprops=dict(arrowstyle='->', lw=0.8, color=CONDITION_COLORS[ctype])
            )

    # ── clean baseline ────────────────────────────────────────────────────────
    if clean_bleu:
        ax.axhline(clean_bleu, color='#555555', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Clean baseline ({clean_bleu:.1f})')

    ax.set_xlabel('Severity (%)')
    ax.set_ylabel('BLEU-4')
    ax.set_title('BLEU-4 Degradation Under Temporal Misalignment\n(★ = first point <= 70% of clean baseline  |  dotted vertical = knee threshold)')
    ax.legend(loc='upper right', ncol=1, fontsize=10)
    ax.set_xlim(left=0)

    # Zoom y-axis to the actual data range so differences between conditions
    # are visible (default ylim(0) flattens all curves when BLEU > 10).
    all_y = [v for vals in curves.values() for v in vals if v is not None]
    if all_y:
        y_lo = max(0, min(all_y) * 0.88)
        y_hi = (clean_bleu if clean_bleu else max(all_y)) * 1.06
        ax.set_ylim(bottom=y_lo, top=y_hi)
    _save_fig(fig, output_dir, 'fig01_bleu_degradation')


# Figure 2: Unified Degradation Dashboard (top) + Heatmap of Absolute Point Changes (bottom) 
def fig02_unified_dashboard_heatmap(results, severity_levels, output_dir, clean_metrics=None):
    # Unified Figure: top dashboard + bottom heatmap
    if clean_metrics is None: clean_metrics = results.get('clean', {}).get('metrics', {})
    line_metrics = [('bleu4', 'BLEU-4'), ('rouge_l', 'ROUGE-L')]
    heat_metrics = [
        ('bleu4', 'BLEU-4 Drop (pp)', 'YlOrRd'),
        ('rouge_l', 'ROUGE-L Drop (pp)', 'YlOrRd'),
    ]
    sevs_pct = [s * 100 for s in severity_levels]

    # Build heatmap matrices first.
    n_types = len(BASIC_ORDER)
    n_sevs = len(severity_levels)
    sev_labels = [f'{int(s * 100)}' for s in severity_levels]
    clean_vals = {m: clean_metrics.get(m, 0) for m in ('bleu4', 'rouge_l')}
    matrices = {}
    for metric, _, _ in heat_metrics:
        mat = np.full((n_types, n_sevs), np.nan)
        cval = clean_vals.get(metric, 0)
        if not cval:
            matrices[metric] = mat
            continue
        for r_i, ctype in enumerate(BASIC_ORDER):
            for c_i, sev in enumerate(severity_levels):
                cond = f'{ctype}_{int(sev * 100):02d}'
                val = results.get(cond, {}).get('metrics', {}).get(metric)
                if val is None: continue
                if METRIC_HIGHER_IS_BETTER.get(metric, True):
                    mat[r_i, c_i] = cval - val
                else:
                    mat[r_i, c_i] = val - cval
        matrices[metric] = mat

    fig_w = max(12, n_sevs * 1.3 + 6)
    fig, axes = plt.subplots(
        2, 2, figsize=(fig_w, 14),
        gridspec_kw={'height_ratios': [1.0, 1.08], 'hspace': 0.2, 'wspace': 0.22},
    )

    # Top row: Dashboard.
    legend_drawn = False
    for ax_idx, (metric, metric_lbl) in enumerate(line_metrics):
        ax = axes[0][ax_idx]
        clean_v = clean_metrics.get(metric, 0)

        if not clean_v:
            ax.set_title(f'{metric_lbl}\n(not available in results)', fontsize=11)
            ax.axis('off')
            continue

        has_data = False
        all_y = []
        for ctype in BASIC_ORDER:
            rel_drops, valid_sevs = [], []
            for sev, sev_pct in zip(severity_levels, sevs_pct):
                cond = f'{ctype}_{int(sev * 100):02d}'
                val = results.get(cond, {}).get('metrics', {}).get(metric)
                if val is None: continue
                rel_v = _relative_degradation(clean_v, val, metric)
                rel_drops.append(rel_v)
                valid_sevs.append(sev_pct)
                all_y.append(rel_v)
            if valid_sevs:
                ax.plot(
                    valid_sevs, rel_drops, linewidth=2.2, markersize=7,
                    color=CONDITION_COLORS[ctype], marker=CONDITION_MARKERS[ctype], label=CONDITION_LABELS[ctype],
                )
                has_data = True

        if not has_data:
            ax.set_title(f'{metric_lbl}\n(no data)', fontsize=11)
            ax.axis('off')
            continue

        ax.axhline(0, color='gray', linestyle='--', linewidth=1.1, alpha=0.55, label='Clean baseline (0%)')
        if all_y:
            y_lo = min(0.0, min(all_y))
            y_hi = max(all_y)
            y_pad = max(1.0, (y_hi - y_lo) * 0.12)
            ax.set_ylim(bottom=y_lo - 0.2 * y_pad, top=y_hi + y_pad)

        ylabel = (
            'Relative Degradation from Clean (%)\n'
            f"{metric_lbl}: {'drop/increase normalized as worse' if METRIC_HIGHER_IS_BETTER.get(metric, True) else 'increase normalized as worse'}"
        )
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel('Severity (%)', fontsize=10)
        ax.set_title(metric_lbl, fontsize=12, fontweight='bold')
        if not legend_drawn:
            ax.legend(loc='upper left', fontsize=9)
            legend_drawn = True

    # Bottom row: Heatmap.
    for ax, (metric, title, cmap) in zip(axes[1], heat_metrics):
        mat = matrices[metric]
        valid = mat[~np.isnan(mat)]
        vmax = float(np.percentile(valid, 95)) if len(valid) else 1.0
        vmax = max(vmax, 0.1)

        im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=0, vmax=vmax, interpolation='nearest')
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        ax.grid(False)

        ax.set_xticks(range(n_sevs))
        ax.set_xticklabels(sev_labels, fontsize=9)
        ax.set_yticks(range(n_types))
        ax.set_yticklabels(BASIC_ORDER, fontsize=13, fontweight='bold')
        ax.set_xlabel('Severity (%)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)

        for r_i in range(n_types):
            for c_i in range(n_sevs):
                v = mat[r_i, c_i]
                if np.isnan(v): continue
                brightness = v / vmax if vmax else 0
                txt_color = 'white' if brightness > 0.62 else 'black'
                ax.text(c_i, r_i, f'{v:.1f}', ha='center', va='center', fontsize=9, color=txt_color, fontweight='bold')

    clean_str = (
        f"BLEU-4={clean_vals.get('bleu4', 0):.1f}  "
        f"ROUGE-L={clean_vals.get('rouge_l', 0):.1f}"
    )
    fig.suptitle(
        'Unified Degradation View\n'
        'Top: Relative Degradation Dashboard   '
        'Bottom: Absolute Point-Change Heatmap\n'
        f'Clean baseline: {clean_str}', fontsize=12,
    )
    # tight_layout is incompatible with this figure's per-axis colorbars.
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.88, hspace=0.36, wspace=0.24)
    _save_fig(fig, output_dir, 'fig02_unified_dashboard_heatmap')


# Figure 3: BLEU vs ROUGE Degradation Diagnostic
def fig03_bleu_vs_rouge(results, severity_levels, output_dir):
    '''BLEU-4 vs ROUGE-L degradation diagnostics.

    Top row (4 panels):
      Relative degradation curves for BLEU-4 and ROUGE-L on a common axis.

    Bottom row (4 panels):
      Raw ROUGE-L vs BLEU-4 scatter across severities for each condition,
      with a fitted slope line to reveal metric coupling.
    '''
    clean_metrics = results.get('clean', {}).get('metrics', {})
    clean_bleu = clean_metrics.get('bleu4')
    clean_rouge = clean_metrics.get('rouge_l')
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), gridspec_kw={'hspace': 0.15, 'wspace': 0.15})
    sevs_pct = [s * 100 for s in severity_levels]

    # Precompute per-condition series for both top (relative) and bottom (raw) rows.
    by_type = {}
    global_rel_vals = []
    all_raw_rouge, all_raw_bleu = [], []
    for ctype in BASIC_ORDER:
        xs, rouge_deg, bleu_deg = [], [], []
        raw_rouge, raw_bleu, raw_sev = [], [], []

        for sev, sev_pct in zip(severity_levels, sevs_pct):
            cond = f'{ctype}_{int(sev * 100):02d}'
            m = results.get(cond, {}).get('metrics', {})
            r = m.get('rouge_l')
            b = m.get('bleu4')
            if r is None or b is None: continue

            raw_rouge.append(r)
            raw_bleu.append(b)
            raw_sev.append(sev_pct)
            all_raw_rouge.append(r)
            all_raw_bleu.append(b)

            if clean_rouge and clean_bleu:
                xs.append(sev_pct)
                rouge_rel = _relative_degradation(clean_rouge, r, 'rouge_l')
                bleu_rel = _relative_degradation(clean_bleu, b, 'bleu4')
                rouge_deg.append(rouge_rel)
                bleu_deg.append(bleu_rel)
                global_rel_vals.extend([rouge_rel, bleu_rel])

        by_type[ctype] = {
            'xs': xs, 'rouge_deg': rouge_deg, 'bleu_deg': bleu_deg,
            'raw_rouge': raw_rouge, 'raw_bleu': raw_bleu, 'raw_sev': raw_sev,
        }

    # Shared y-limits for top-row relative-degradation panels.
    if global_rel_vals:
        y_lo = min(0.0, min(global_rel_vals))
        y_hi = max(global_rel_vals)
        y_pad = max(1.0, (y_hi - y_lo) * 0.12)
        rel_limits = (y_lo - 0.2 * y_pad, y_hi + y_pad)
    else:
        rel_limits = (-1.0, 1.0)

    # Shared axes for bottom-row raw scatter panels.
    raw_xlim = raw_ylim = None
    if all_raw_rouge and all_raw_bleu:
        x_lo, x_hi = min(all_raw_rouge), max(all_raw_rouge)
        y_lo, y_hi = min(all_raw_bleu), max(all_raw_bleu)
        x_pad = max(0.5, (x_hi - x_lo) * 0.12)
        y_pad = max(0.5, (y_hi - y_lo) * 0.12)
        raw_xlim = (max(0.0, x_lo - x_pad), x_hi + x_pad)
        raw_ylim = (max(0.0, y_lo - y_pad), y_hi + y_pad)

    # ── Top row: relative-degradation lines. ──
    for idx, ctype in enumerate(BASIC_ORDER):
        ax = axes[0][idx]
        series = by_type.get(ctype, {})
        xs = series.get('xs', [])
        rouge_deg = series.get('rouge_deg', [])
        bleu_deg = series.get('bleu_deg', [])

        if len(xs) < 2:
            ax.set_title(f'{CONDITION_LABELS[ctype]}\n(insufficient data)')
            ax.axhline(0, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)
            ax.set_ylim(*rel_limits)
            ax.set_xlim(left=0)
            continue

        l1, = ax.plot(xs, bleu_deg, color='#2980b9', marker='s', label='BLEU-4 degradation (%)', linewidth=2.2, markersize=7)
        l2, = ax.plot(xs, rouge_deg, color='#27ae60', marker='^', label='ROUGE-L degradation (%)', linewidth=2.2, markersize=7)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.1, alpha=0.6)
        ax.set_ylim(*rel_limits)
        ax.set_xlim(left=0)

        if len(bleu_deg) >= 3:
            corr = np.corrcoef(bleu_deg, rouge_deg)[0, 1]
            if not np.isfinite(corr): coupling = 'Uncorrelated'
            elif corr >= 0.9: coupling = 'Strongly coupled'
            elif corr >= 0.7: coupling = 'Moderately coupled'
            else: coupling = 'Weakly coupled'
            title_detail = f'r = {corr:.3f}  ({coupling})'
        else:
            title_detail = ''

        ax.set_title(f'{CONDITION_LABELS[ctype]}\n{title_detail}', fontsize=11)
        ax.set_xlabel('Severity (%)')
        ax.legend([l1, l2], ['BLEU-4 degradation (%)', 'ROUGE-L degradation (%)'], loc='upper left', fontsize=8.6)
    axes[0][0].set_ylabel('Relative Degradation from Clean (%)\n(higher = worse)')

    # ── Bottom row: raw ROUGE-L vs BLEU-4 scatter with slope. ─────────────────────
    cbar_vmin = min(sevs_pct) if sevs_pct else 0
    cbar_vmax = max(sevs_pct) if sevs_pct else 1
    for idx, ctype in enumerate(BASIC_ORDER):
        ax = axes[1][idx]
        series = by_type.get(ctype, {})
        raw_rouge = series.get('raw_rouge', [])
        raw_bleu = series.get('raw_bleu', [])
        raw_sev = series.get('raw_sev', [])

        if len(raw_rouge) < 2:
            ax.set_title(f'{CONDITION_LABELS[ctype]}\n(raw scatter: insufficient data)', fontsize=10.8)
            ax.set_xlabel('Raw ROUGE-L')
            if idx == 0: ax.set_ylabel('Raw BLEU-4')
            if raw_xlim: ax.set_xlim(*raw_xlim)
            if raw_ylim: ax.set_ylim(*raw_ylim)
            continue

        scatter = ax.scatter(
            raw_rouge, raw_bleu, c=raw_sev, cmap='viridis_r', vmin=cbar_vmin, vmax=cbar_vmax,
            s=60, alpha=0.86, edgecolor='white', linewidth=0.55,
            marker=CONDITION_MARKERS[ctype], zorder=3,
        )
        if clean_rouge is not None and clean_bleu is not None: ax.scatter(
            [clean_rouge], [clean_bleu], marker='*', s=130,
            color='#f1c40f', edgecolor='black', linewidth=0.6, zorder=4,
        )
        slope = None
        corr_raw = np.nan
        if len(raw_rouge) >= 2:
            slope, intercept = np.polyfit(raw_rouge, raw_bleu, 1)
            x_fit = np.linspace(min(raw_rouge), max(raw_rouge), 60)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, color=CONDITION_COLORS[ctype], linewidth=2.1, alpha=0.95, zorder=2)
            if len(raw_rouge) >= 3: corr_raw = np.corrcoef(raw_rouge, raw_bleu)[0, 1]

        if np.isfinite(corr_raw):
            if corr_raw >= 0.9: pattern = 'strong coupling'
            elif corr_raw >= 0.7: pattern = 'moderate coupling'
            else: pattern = 'weak/nonlinear'
            diag_text = f'slope={slope:.2f}\nr={corr_raw:.3f} ({pattern})'
        else:
            diag_text = f'slope={slope:.2f}' if slope is not None else 'slope=n/a'

        ax.text(
            0.5, 0.96, diag_text, transform=ax.transAxes, va='top', ha='left', fontsize=8.8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.86),
        )
        for r, b, s in zip(raw_rouge, raw_bleu, raw_sev):
            ax.annotate(f'{int(round(s))}%', (r, b), textcoords='offset points', xytext=(4, 3), fontsize=7.2, alpha=0.80)

        ax.set_xlabel('Raw ROUGE-L')
        if idx == 0: ax.set_ylabel('Raw BLEU-4')
        if raw_xlim: ax.set_xlim(*raw_xlim)
        if raw_ylim: ax.set_ylim(*raw_ylim)

    fig.suptitle(
        'BLEU-4 vs ROUGE-L Degradation Diagnostics\n'
        'Top: Relative degradation from clean   '
        'Bottom: Raw ROUGE-L vs BLEU-4 scatter with fitted slope',
        fontsize=12.8, y=0.99,
    )
    _save_fig(fig, output_dir, 'fig03_bleu_vs_rouge')


# Figure 4: Train vs Dev Structural Vulnerability
def fig04_train_vs_dev(dev_results, train_results, severity_levels, output_dir):
    """Train-vs-dev BLEU structural vulnerability in two complementary views.

    Top row (4 panels):
      Existing BLEU-4-vs-severity curves for dev (solid) and train subset (dashed).

    Bottom row (4 panels):
      Raw Train BLEU-4 vs Dev BLEU-4 scatter across severities, with fitted slope to expose train-dev propagation pattern.
    """
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), gridspec_kw={'hspace': 0.15, 'wspace': 0.15})
    sevs_pct = [s * 100 for s in severity_levels]

    # Precompute per-condition curves (top) and paired raw points (bottom).
    by_type = {}
    all_train_bleu, all_dev_bleu = [], []
    for ctype in BASIC_ORDER:
        dev_vals, dev_xs = [], []
        train_vals, train_xs = [], []
        paired_train, paired_dev, paired_sev = [], [], []

        for sev, sev_pct in zip(severity_levels, sevs_pct):
            cond = f'{ctype}_{int(sev * 100):02d}'
            t_val = dev_results.get(cond, {}).get('metrics', {}).get('bleu4')
            tr_val = train_results.get(cond, {}).get('metrics', {}).get('bleu4')

            if t_val is not None:
                dev_vals.append(t_val)
                dev_xs.append(sev_pct)
            if tr_val is not None:
                train_vals.append(tr_val)
                train_xs.append(sev_pct)

            if t_val is not None and tr_val is not None:
                paired_train.append(tr_val)
                paired_dev.append(t_val)
                paired_sev.append(sev_pct)
                all_train_bleu.append(tr_val)
                all_dev_bleu.append(t_val)

        by_type[ctype] = {
            'dev_vals': dev_vals, 'dev_xs': dev_xs, 'train_vals': train_vals, 'train_xs': train_xs,
            'paired_train': paired_train, 'paired_dev': paired_dev, 'paired_sev': paired_sev,
        }

    raw_xlim = raw_ylim = None
    if all_train_bleu and all_dev_bleu:
        x_lo, x_hi = min(all_train_bleu), max(all_train_bleu)
        y_lo, y_hi = min(all_dev_bleu), max(all_dev_bleu)
        x_pad = max(0.5, (x_hi - x_lo) * 0.12)
        y_pad = max(0.5, (y_hi - y_lo) * 0.12)
        raw_xlim = (max(0.0, x_lo - x_pad), x_hi + x_pad)
        raw_ylim = (max(0.0, y_lo - y_pad), y_hi + y_pad)

    # ── Top row: existing train-vs-dev BLEU curves. ───────────────────────
    for idx, ctype in enumerate(BASIC_ORDER):
        ax = axes[0][idx]
        series = by_type.get(ctype, {})

        t_xs = series.get('dev_xs', [])
        t_vals = series.get('dev_vals', [])
        if t_xs: ax.plot(
            t_xs, t_vals, '-', color=CONDITION_COLORS[ctype], marker='o',
            linewidth=2.8, alpha=0.95, markersize=7.5, markeredgecolor='white', markeredgewidth=0.8,
        )
        tr_xs = series.get('train_xs', [])
        tr_vals = series.get('train_vals', [])
        if tr_xs: ax.plot(
            tr_xs, tr_vals, '--', color=CONDITION_COLORS[ctype], marker='X',
            linewidth=2.2, alpha=0.90, markersize=7.5, markeredgecolor='white', markeredgewidth=0.8,
        )
        # Clean baselines
        for res, lbl, lstyle in [(dev_results, 'Clean (dev)', ':'), (train_results, 'Clean (train)', '-.')]:
            cb = res.get('clean', {}).get('metrics', {}).get('bleu4')
            if cb is not None: ax.axhline(cb, color='gray', linestyle=lstyle, linewidth=1.3, alpha=0.75, label=f'{lbl} ({cb:.1f})')

        ax.set_xlim(left=0)
        ax.set_xlabel('Severity (%)')
        ax.set_title(CONDITION_LABELS[ctype])
        style_handles = [
            Line2D([0], [0], color='black', linestyle='-', marker='o', markersize=7, linewidth=2.8, label='Dev'),
            Line2D([0], [0], color='black', linestyle='--', marker='X', markersize=7, linewidth=2.2, label='Train (subset)'),
            Line2D([0], [0], color='gray', linestyle=':', linewidth=1.3, label='Clean (Dev)'),
            Line2D([0], [0], color='gray', linestyle='-.', linewidth=1.3, label='Clean (Train)'),
        ]
        ax.legend(handles=style_handles, fontsize=8.8, framealpha=0.9, loc='upper right')
    axes[0][0].set_ylabel('BLEU-4')

    # ── Bottom row: raw Train BLEU vs Dev BLEU scatter. ─────────────────────
    cbar_vmin = min(sevs_pct) if sevs_pct else 0
    cbar_vmax = max(sevs_pct) if sevs_pct else 1
    clean_train_bleu = train_results.get('clean', {}).get('metrics', {}).get('bleu4')
    clean_dev_bleu = dev_results.get('clean', {}).get('metrics', {}).get('bleu4')

    for idx, ctype in enumerate(BASIC_ORDER):
        ax = axes[1][idx]
        series = by_type.get(ctype, {})
        x_train = series.get('paired_train', [])
        y_dev = series.get('paired_dev', [])
        sev_pts = series.get('paired_sev', [])

        if len(x_train) < 2:
            ax.set_title(f'{CONDITION_LABELS[ctype]}\n(raw scatter: insufficient data)', fontsize=10.8)
            ax.set_xlabel('Train BLEU-4')
            if idx == 0: ax.set_ylabel('Test BLEU-4')
            if raw_xlim: ax.set_xlim(*raw_xlim)
            if raw_ylim: ax.set_ylim(*raw_ylim)
            continue

        scatter = ax.scatter(
            x_train, y_dev, c=sev_pts, cmap='viridis_r', vmin=cbar_vmin, vmax=cbar_vmax,
            s=60, alpha=0.86, edgecolor='white', linewidth=0.55,
            marker=CONDITION_MARKERS[ctype], zorder=3,
        )
        if clean_train_bleu is not None and clean_dev_bleu is not None: ax.scatter(
            [clean_train_bleu], [clean_dev_bleu], marker='*', s=130,
            color='#f1c40f', edgecolor='black', linewidth=0.6, zorder=4,
        )
        slope = None
        corr_raw = np.nan
        if len(x_train) >= 2:
            slope, intercept = np.polyfit(x_train, y_dev, 1)
            x_fit = np.linspace(min(x_train), max(x_train), 60)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, color=CONDITION_COLORS[ctype], linewidth=2.1, alpha=0.95, zorder=2)
            if len(x_train) >= 3: corr_raw = np.corrcoef(x_train, y_dev)[0, 1]

        if np.isfinite(corr_raw):
            if slope is not None and slope > 1.05: pattern = 'Dev steeper than Train'
            elif slope is not None and slope < 0.95: pattern = 'Train steeper than Dev'
            else: pattern = 'near 1:1 coupling'
            diag_text = f'slope={slope:.2f}\nr={corr_raw:.3f} ({pattern})'
        else:
            diag_text = f'slope={slope:.2f}' if slope is not None else 'slope=n/a'

        ax.text(
            0.04, 0.96, diag_text, transform=ax.transAxes, va='top', ha='left', fontsize=8.8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.86),
        )
        for xv, yv, s in zip(x_train, y_dev, sev_pts):
            ax.annotate(f'{int(round(s))}%', (xv, yv), textcoords='offset points', xytext=(4, 3), fontsize=7.2, alpha=0.80)

        # ax.set_title(f'{CONDITION_LABELS[ctype]}\nRaw Train vs Dev BLEU', fontsize=10.8)
        ax.set_xlabel('Train BLEU-4')
        if idx == 0: ax.set_ylabel('Dev BLEU-4')
        if raw_xlim: ax.set_xlim(*raw_xlim)
        if raw_ylim: ax.set_ylim(*raw_ylim)

    fig.suptitle(
        'Structural Vulnerability: Train vs Dev BLEU-4\n'
        'Top: BLEU-4 degradation curves by severity   '
        'Bottom: Raw Train BLEU vs Dev BLEU scatter with fitted slope',
        fontsize=12.6, y=0.99,
    )
    _save_fig(fig, output_dir, 'fig04_train_vs_dev')


# Figure 5: Failure Mode Distribution — basic (top) + compound (bottom)
def fig05_failure_distribution(results, severity_levels, output_dir, compound_results=None):
    '''2×4 stacked-bar grid.

    Top row: one panel per basic condition type (x-axis = severity levels).
    Bottom row: one panel per compound pair family (x-axis = individual compound conditions sorted by total severity).
    All bars are 100% stacked; colors = FAILURE_COLORS.

    Args:
        results:          Results dict used for the top-row basic panels. Typically knee_point.json (10 severity levels).
        severity_levels:  Severity levels present in `results`.
        compound_results: Optional separate results dict that contains compound conditions (e.g. benchmark.json).
                          When provided, the bottom row is drawn from this dict. When None, `results`
                          is used for both rows (bottom row will be empty if it has no compound keys).
    '''
    if compound_results is None: compound_results = results
    classifications          = classify_all_predictions(results)
    compound_classifications = classify_all_predictions(compound_results)
    
    # Merge so failure_mode_distribution sees all conditions
    merged_class = {**classifications, **compound_classifications}
    distributions = failure_mode_distribution(merged_class)
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), sharey=True)

    def _stacked_bar_panel(ax, x_labels, cond_names, title, x_title='Severity (%)'): #Draw a 100%-stacked bar chart on ax
        if not x_labels:
            ax.set_title(title)
            return
        
        x = np.arange(len(x_labels))
        bottom = np.zeros(len(x_labels))
        for mode in FAILURE_MODES:
            vals = np.array([distributions.get(c, {}).get(mode, 0) for c in cond_names])
            ax.bar(x, vals, bottom=bottom, label=mode, color=FAILURE_COLORS[mode], width=0.72, edgecolor='white', lw=0.4)
            bottom += vals
            
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
        ax.set_xlabel(x_title, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(False)

    # ── Top row: basic conditions ─────────────────────────────────────────────
    for col, ctype in enumerate(BASIC_ORDER):
        ax = axes[0][col]
        xlabels, cond_names = [], []
        
        for sev in severity_levels:
            cond = f'{ctype}_{int(sev * 100):02d}'
            if cond in distributions:
                xlabels.append(f'{int(sev * 100)}%')
                cond_names.append(cond)
                
        _stacked_bar_panel(ax, xlabels, cond_names, title=CONDITION_LABELS[ctype])
        if col == 0: ax.set_ylabel('Samples (%)')

    # ── Bottom row: compound pair families (from compound_results) ────────────
    # Detect which compound pairs ACTUALLY exist in the data — avoids empty panels caused by 
    # canonical pairs (e.g. HC+TT → TT+HC) that don't correspond to any condition name stored in the results.
    actual_pairs_data = defaultdict(list)
    for cond_name in compound_results:
        if '+' not in cond_name or cond_name in ('clean', 'meta'): continue
        p = _parse_compound_name(cond_name)
        if p is None: continue
        pair = _canonical_pair(p[0], p[2])
        actual_pairs_data[pair].append((round(p[1] + p[3], 4), p[1], p[3], cond_name))
    
    pair_order = [ # Order by canonical BASIC_ORDER (lower-index type first)
        _canonical_pair(a, b) for a in BASIC_ORDER for b in BASIC_ORDER
        if BASIC_ORDER.index(a) < BASIC_ORDER.index(b)
    ]
    pair_order = [p for p in pair_order if p in actual_pairs_data]

    for col, pair in enumerate(pair_order[:4]):
        ax         = axes[1][col]
        pair_conds = sorted(actual_pairs_data[pair])
        xlabels, cond_names = [], []
        
        for _, sa, sb, cond_name in pair_conds:
            pi = _parse_compound_name(cond_name)
            if pi: xlabels.append(f'{pi[0]} {int(pi[1]*100)}%\n+ {pi[2]} {int(pi[3]*100)}%')
            else: xlabels.append(cond_name)
            cond_names.append(cond_name)
            
        # keep only those with computed distributions
        xl_f = [xl for xl, cn in zip(xlabels, cond_names) if cn in distributions]
        cn_f = [cn for cn in cond_names if cn in distributions]
        _stacked_bar_panel(ax, xl_f, cn_f, title=COMPOUND_PAIR_LABELS.get(pair, pair), x_title='Component severities')
        if col == 0: ax.set_ylabel('Samples (%)')

    # Shared legend below figure
    legend_handles = [Patch(color=FAILURE_COLORS[m], label=m) for m in FAILURE_MODES]
    fig.legend(handles=legend_handles, loc='lower center', ncol=6, fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Failure Mode Distribution — Basic Conditions (top) & Compound Pairs (bottom)', fontsize=13, y=1.01)
    fig.tight_layout()
    _save_fig(fig, output_dir, 'fig05_failure_distribution')


# Figure 6: Failure Mode Transition Matrices (per-sample cross-tabulation)
def fig06_transition_matrices(results, severity_levels, output_dir):
    '''Failure-mode transition matrices with consecutive severity steps.

    Layout: 4 condition types (rows) × 5 consecutive transition columns.
    Transitions track what happens BETWEEN adjacent severity steps:
        clean → 10%,  10% → 20%,  20% → 30%,  30% → 40%,  40% → 50%
    (uses the nearest available severity level when targets are not exact).

    Failure modes collapsed from 6 → 4:
        Partial match  → Hallucination   (wrong content in any form)
        Repetition     → Incoherent      (output-quality failure)

    Each cell is a 4×4 heatmap:
        Rows    = failure mode at FROM condition
        Columns = failure mode at TO condition
        Color   = row-normalised % (Blues gradient; 100 % = always transitions here)
        Number  = raw sample count
    Diagonal (no mode change between steps) framed in gold.
    '''
    # ── Build 4-mode classifications ─────────────────────────────────────────
    raw_cls = classify_all_predictions(results)
    cls4    = {}
    for cond, samples in raw_cls.items():
        cls4[cond] = {name: _remap_to_4_modes(mode) for name, mode in samples.items()}
    if 'clean' not in cls4: return

    # ── Consecutive transition pairs ─────────────────────────────────────────
    # Each tuple: (from_target_sev, to_target_sev)
    # None means the clean (0 %) condition.
    TRANSITION_TARGETS = [(None,  0.10), (0.10,  0.20), (0.20,  0.30), (0.30,  0.40), (0.40,  0.50)]
    COL_LABELS = ['clean → 10%', '10 → 20%', '20 → 30%', '30 → 40%', '40 → 50%']

    def _nearest_key(ctype, target): # Return the condition key for ctype nearest to target severity
        if target is None: return 'clean'
        best = min(severity_levels, key=lambda s: abs(s - target))
        return f'{ctype}_{int(best * 100):02d}'

    n_modes   = len(FOUR_MODES)
    mode_abbr = [FOUR_MODE_ABBREV[m] for m in FOUR_MODES]
    fig, axes = plt.subplots(4, 5, figsize=(21, 16))

    for row_idx, ctype in enumerate(BASIC_ORDER):
        for col_idx, ((from_t, to_t), col_lbl) in enumerate(zip(TRANSITION_TARGETS, COL_LABELS)):
            ax       = axes[row_idx][col_idx]
            from_key = _nearest_key(ctype, from_t)
            to_key   = _nearest_key(ctype, to_t)
            from_cls = cls4.get(from_key, {})
            to_cls   = cls4.get(to_key,   {})
            common   = set(from_cls.keys()) & set(to_cls.keys())

            # Column header (top row only)
            if row_idx == 0: ax.set_title(col_lbl, fontsize=9, fontweight='bold', pad=5)
            if not common:
                ax.axis('off')
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=9, color='gray')
                continue

            # ── Build 4×4 transition matrix ──────────────────────────────────
            mode_to_idx = {m: i for i, m in enumerate(FOUR_MODES)}
            mat = np.zeros((n_modes, n_modes), dtype=int)
            for sample in common:
                i = mode_to_idx.get(from_cls[sample], n_modes - 1)
                j = mode_to_idx.get(to_cls[sample],   n_modes - 1)
                mat[i, j] += 1

            # Row-normalise to percentages
            row_sums = mat.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            pct_mat  = mat / row_sums * 100

            im = ax.imshow(pct_mat, cmap='Blues', vmin=0, vmax=100, aspect='auto', interpolation='nearest')
            ax.set_xticks(range(n_modes))
            ax.set_yticks(range(n_modes))
            ax.set_xticklabels(mode_abbr, fontsize=10)
            ax.set_yticklabels(mode_abbr, fontsize=10)
            if col_idx == 0: ax.set_ylabel(f'{ctype}\nFrom mode', fontsize=9, fontweight='bold')
            if row_idx == 3: ax.set_xlabel('To mode', fontsize=10)

            # Cell text: raw count
            for i in range(n_modes):
                for j in range(n_modes):
                    cnt = mat[i, j]
                    if cnt == 0: continue
                    txt_color = 'white' if pct_mat[i, j] > 58 else 'black'
                    ax.text(j, i, str(cnt), ha='center', va='center', fontsize=12, color=txt_color)

            # Gold border around diagonal (stable — no mode change this step)
            for d in range(n_modes):
                rect = plt.Rectangle((d - 0.5, d - 0.5), 1, 1, fill=False, edgecolor='#f39c12', linewidth=2.0, zorder=4)
                ax.add_patch(rect)

            ax.text( # Small n= annotation inside cell
                0.02, 0.02, f'n={len(common)}', transform=ax.transAxes, fontsize=10, va='bottom', 
                color='#444', bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.65)
            )
            ax.grid(False)

    fig.suptitle(
        'Failure Mode Transitions — Consecutive Severity Steps  '
        '(4-mode view: Partial match→Hall., Repetition→Incoh.)\n'
        'Color = row-normalised %  |  Number = raw count  |  Gold frame = no mode change', fontsize=11, y=1.01)
    fig.tight_layout(pad=1.2)
    _save_fig(fig, output_dir, 'fig06_transition_matrices')


# Figure 7: Sentence Length vs BLEU Drop
def fig07_length_vs_bleu_drop(results, severity_levels, output_dir, target_severities=(0.10, 0.20, 0.40)):
    '''Scatter: reference length vs BLEU drop per basic condition.

    Multiple severity levels overlaid with alpha encoding so severity trend is visible within each panel.
    '''
    clean_ps = _get_condition_samples(results, 'clean')
    if not clean_ps: return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sev_colors = ['#95a5a6', '#3498db', '#e74c3c']  # low / mid / high severity
    sev_show = []
    for t in target_severities:
        best = min(severity_levels, key=lambda s: abs(s - t))
        if best not in sev_show: sev_show.append(best)

    for idx, ctype in enumerate(BASIC_ORDER):
        ax = axes[idx // 2][idx % 2]
        corr_rows = []
        for sev, sev_color in zip(sev_show, sev_colors):
            cond = f'{ctype}_{int(sev * 100):02d}'
            mis_ps = _get_condition_samples(results, cond)
            if not mis_ps: continue
            
            lengths, drops = [], []
            for name, cm in clean_ps.items():
                if name in mis_ps:
                    lengths.append(cm['ref_length'])
                    drops.append(cm['sentence_bleu'] - mis_ps[name]['sentence_bleu'])

            if not lengths: continue
            ax.scatter(lengths, drops, alpha=0.42, s=12, color=sev_color, label=f'{int(sev * 100)}%')
            if len(lengths) > 5:
                z = np.polyfit(lengths, drops, 1)
                p = np.poly1d(z)
                xl = np.linspace(min(lengths), max(lengths), 80)
                ax.plot(xl, p(xl), color=sev_color, linewidth=2.2, alpha=0.95)
                r = np.corrcoef(lengths, drops)[0, 1]
                corr_rows.append(f'{int(sev*100)}%: r={r:.2f}')

        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Reference length (tokens)')
        ax.set_ylabel('BLEU drop  (clean − misaligned)')
        ax.set_title(CONDITION_LABELS[ctype])
        if corr_rows: ax.text(
            0.98, 0.02, '\n'.join(corr_rows), transform=ax.transAxes, ha='right', va='bottom', 
            fontsize=8.8, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.82)
        )

    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='None', markersize=6, color=sev_color, alpha=0.75, label=f'{int(sev*100)}% points')
        for sev, sev_color in zip(sev_show, sev_colors)
    ] + [
        Line2D([0], [0], linestyle='-', linewidth=2.2, color=sev_color, label=f'{int(sev*100)}% trend')
        for sev, sev_color in zip(sev_show, sev_colors)
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, framealpha=0.92)
    title = (
        'Sentence Length vs BLEU-4 Drop (multi-severity)\n'
        'Color = severity; lines = trend; in-panel text reports correlation'
    )
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.subplots_adjust(top=0.89, bottom=0.16, wspace=0.22, hspace=0.28)
    _save_fig(fig, output_dir, 'fig07_length_vs_bleu_drop')


# Figure 8: Per-Sample Vulnerability Profile
def fig08_vulnerability_profile(results, severity_levels, output_dir):
    '''Per-sample vulnerability profile using histogram + actionable annotations.

    x-axis = BLEU slope (more negative = more vulnerable).
    The plot is optimized for quick audience takeaway: vulnerable share, robust share, center tendency, and spread.
    '''
    clean_ps = _get_condition_samples(results, 'clean')
    if not clean_ps: return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, ctype in enumerate(BASIC_ORDER):
        ax = axes[idx // 2][idx % 2]
        sample_rates = []
        for name in clean_ps:
            xs, ys = [], []
            for sev in severity_levels:
                cond = f'{ctype}_{int(sev * 100):02d}'
                ps   = _get_condition_samples(results, cond)
                if name in ps:
                    xs.append(sev)
                    ys.append(ps[name]['sentence_bleu'])
            if len(xs) >= 3: sample_rates.append(compute_degradation_rate(xs, ys) / 100)

        if not sample_rates: continue
        rates = np.array(sample_rates, dtype=float)
        q25, q50, q75 = np.percentile(rates, [25, 50, 75])
        n_total = len(rates)
        n_vuln = int(np.sum(rates < -0.5))
        p_vuln = 100 * n_vuln / max(1, n_total)
        
        # Histogram with common readable bins and distribution shape
        bins = np.linspace(-1.5, 0.5, 22)
        ax.hist(rates, bins=bins, color=CONDITION_COLORS[ctype], alpha=0.78, edgecolor='white', linewidth=0.45)

        # Decision thresholds and summary markers
        ax.axvline(-0.5, color='#c0392b', linestyle='--', linewidth=1.8, alpha=0.95)
        ax.axvline(q50, color='#2c3e50', linestyle='-', linewidth=2.0, alpha=0.95)
        ax.axvspan(q25, q75, color='#2c3e50', alpha=0.10)

        ax.text(-0.45, ax.get_ylim()[1] * 0.95, 'vulnerability\ncutoff (-0.5)', color='#c0392b', fontsize=8.2, ha='left', va='top')
        ax.text(
            0.05, 0.98, f'Vulnerable: {n_vuln}/{n_total} ({p_vuln:.1f}%)\nMedian slope: {q50:.3f}\nIQR: [{q25:.3f}, {q75:.3f}]',
            transform=ax.transAxes, va='top', ha='left', fontsize=8.8,
            bbox=dict(boxstyle='round,pad=0.35', fc='white', alpha=0.88)
        )
        ax.set_xlabel('BLEU slope (BLEU / severity unit)')
        ax.set_ylabel('Sample count')
        ax.set_title(CONDITION_LABELS[ctype], fontsize=10)

    fig.suptitle('Per-Sample Vulnerability Profiles (histogram view)\n'
                 'Panels with more mass left of -0.5 are more vulnerable', fontsize=12, y=1.01)
    fig.subplots_adjust(top=0.90, bottom=0.10, wspace=0.24, hspace=0.28)
    _save_fig(fig, output_dir, 'fig08_vulnerability_profile')


# Figure 9: Interaction Analysis — Predicted-Additive vs Actual Compound Drop
def fig09_interaction_scatter(results, output_dir, clean_metrics=None):
    '''Deviation-from-additivity scatter for all compound conditions.

    For each compound condition A_{sa} + B_{sb}:
        predicted_drop = drop_A + drop_B  (linear/independent assumption)
        actual_drop    = clean − compound
        deviation      = actual − predicted

    Y-axis = deviation (pp):
        > 0 → superadditive  (compound is WORSE than sum of parts)
        = 0 → purely additive
        < 0 → subadditive    (compound is MILDER than sum of parts)
    X-axis = predicted additive drop (how severe the combined effect was expected).

    This layout spreads points vertically around y=0 so super/sub-additive structure is immediately visible 
    unlike a (pred, actual) scatter where all points cluster near the y=x diagonal.

    BLEU-4 drop and ROUGE-L drop shown in side-by-side panels.
    Colour = compound pair family; marker = severity combo (sa, sb).
    '''
    if clean_metrics is None: clean_metrics = results.get('clean', {}).get('metrics', {})
    clean_bleu  = clean_metrics.get('bleu4', 0)
    clean_rouge = clean_metrics.get('rouge_l', 0)
    if not clean_bleu: return

    fig, (ax_rouge, ax_bleu) = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    all_sev_combos = {}
    rouge_data = defaultdict(list)
    bleu_data  = defaultdict(list)  # pair → [(pred, deviation, a_sev, b_sev, lbl)]

    for cond_name, cond_data in results.items():
        if '+' not in cond_name or cond_name in ('clean', 'meta'): continue
        p = _parse_compound_name(cond_name)
        if p is None: continue

        a_type, a_sev, b_type, b_sev = p
        pair  = _canonical_pair(a_type, b_type)
        cond_a = f'{a_type}_{int(a_sev * 100):02d}'
        cond_b = f'{b_type}_{int(b_sev * 100):02d}'
        if cond_a not in results or cond_b not in results: continue

        ma = results[cond_a]['metrics']
        mb = results[cond_b]['metrics']
        mc = cond_data['metrics']

        sev_key = (int(a_sev * 100), int(b_sev * 100))
        if sev_key not in all_sev_combos: all_sev_combos[sev_key] = len(all_sev_combos)
        short_lbl = f'{a_type}{int(a_sev*100)}+{b_type}{int(b_sev*100)}'

        # BLEU deviation
        bleu_c = mc.get('bleu4')
        if bleu_c is not None:
            drop_a  = clean_bleu - ma.get('bleu4', clean_bleu)
            drop_b  = clean_bleu - mb.get('bleu4', clean_bleu)
            pred    = drop_a + drop_b
            actual  = clean_bleu - bleu_c
            dev     = actual - pred          # >0 superadditive
            bleu_data[pair].append((pred, dev, a_sev, b_sev, short_lbl))

        # ROUGE-L deviation
        if clean_rouge:
            rouge_c = mc.get('rouge_l')
            if rouge_c is not None:
                drop_a_r = clean_rouge - ma.get('rouge_l', clean_rouge)
                drop_b_r = clean_rouge - mb.get('rouge_l', clean_rouge)
                pred_r   = drop_a_r + drop_b_r
                act_r    = clean_rouge - rouge_c
                dev_r    = act_r - pred_r
                rouge_data[pair].append((pred_r, dev_r, a_sev, b_sev, short_lbl))

    # Ensure unique marker per severity pair (no reuse ambiguity).
    unique_pairs = sorted(all_sev_combos.keys())
    marker_catalog = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P', '<', '>', 'd', 'H']
    marker_map = {}
    for i, sev_key in enumerate(unique_pairs):
        if i < len(marker_catalog): marker_map[sev_key] = marker_catalog[i]
        else: marker_map[sev_key] = marker_catalog[i % len(marker_catalog)]

    # Share one y-scale so ROUGE/BLEU interaction strength is directly comparable.
    all_dev_vals = [dev for pts in bleu_data.values() for _, dev, *_ in pts]
    all_dev_vals += [dev for pts in rouge_data.values() for _, dev, *_ in pts]
    shared_ylim = None
    if all_dev_vals:
        y_abs_global = max(abs(v) for v in all_dev_vals)
        y_span_global = max(1.0, y_abs_global * 1.35)
        shared_ylim = (-y_span_global, y_span_global - 1.0)

    def _draw_deviation_scatter(ax, data_dict, title, xlabel, ylabel, shared_ylim=None):
        all_preds, all_devs, flat = [], [], []
        for pair, pts in data_dict.items():
            color = COMPOUND_PAIR_COLORS.get(pair, '#7f8c8d')
            label = COMPOUND_PAIR_LABELS.get(pair, pair)
            for pred, dev, a_sev, b_sev, lbl in pts:
                all_preds.append(pred)
                all_devs.append(dev)
                flat.append((pred, dev, lbl, pair))
                sev_key = (int(a_sev * 100), int(b_sev * 100))
                mk = marker_map.get(sev_key, 'o')
                ax.scatter(pred, dev, color=color, marker=mk, s=62, edgecolor='white', linewidth=0.35, alpha=0.78, zorder=4)
            ax.scatter([], [], color=color, marker='o', label=label, s=52)

        if not all_preds:
            ax.set_title(f'{title}\n(no compound data)')
            if shared_ylim is not None: ax.set_ylim(*shared_ylim)
            return

        # y = 0 horizontal baseline (additive)
        x_lo, x_hi = min(all_preds), max(all_preds)
        x_margin = max(0.4, (x_hi - x_lo) * 0.08)
        ax.axhline(0, color='black', linestyle='--', linewidth=1.6, alpha=0.55, label='Additive baseline')

        # Shaded zones
        if shared_ylim is not None: y_lo, y_hi = shared_ylim
        else:
            y_abs = max(abs(v) for v in all_devs) if all_devs else 1
            y_span = max(1.0, y_abs * 1.35)
            y_lo, y_hi = -y_span, y_span
        ax.fill_between([x_lo - x_margin, x_hi + x_margin], [0, 0], [y_hi, y_hi],
                        alpha=0.07, color='red', label='Superadditive zone\n(worse than expected)')
        ax.fill_between([x_lo - x_margin, x_hi + x_margin], [y_lo, y_lo], [0, 0],
                        alpha=0.07, color='green', label='Subadditive zone\n(milder than expected)')

        n_super = sum(1 for d in all_devs if d > 0.5)
        n_sub   = sum(1 for d in all_devs if d < -0.5)
        n_add   = len(all_devs) - n_super - n_sub
        ax.text(
            0.03, 0.97, f'Superadditive (> 0.5): {n_super}\n≈ Additive (0.5 to -0.5): {n_add}\nSubadditive (< -0.5): {n_sub}',
            transform=ax.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85)
        )
        if flat: # Label only the strongest interactions to reduce clutter.
            top_idx = np.argsort(np.abs(np.array(all_devs)))[-3:]
            for i in top_idx:
                pred, dev, lbl, pair = flat[int(i)]
                ax.annotate(
                    lbl, (pred, dev), xytext=(5, 5), textcoords='offset points', fontsize=7.3,
                    color=COMPOUND_PAIR_COLORS.get(pair, '#7f8c8d'), alpha=0.9
                )
        ax.set_xlim(x_lo - x_margin, x_hi + x_margin)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(loc='lower left', fontsize=8.8, ncol=2, framealpha=0.9)

    _draw_deviation_scatter(
        ax_rouge, rouge_data, 'ROUGE-L: Deviation from Additivity',
        'Predicted additive ROUGE-L drop (pp)\n= drop_A + drop_B',
        'Deviation (actual − predicted)  pp\n> 0 = superadditive  |  < 0 = subadditive',
        shared_ylim=shared_ylim
    )
    _draw_deviation_scatter(
        ax_bleu, bleu_data, 'BLEU-4: Deviation from Additivity',
        'Predicted additive BLEU drop (pp)\n= drop_A + drop_B',
        'Deviation (actual − predicted)  pp\n> 0 = superadditive  |  < 0 = subadditive',
        shared_ylim=shared_ylim
    )
    # Explicit marker-shape legend: shape maps severity combination (a%, b%).
    shape_handles = []
    for sev_key in unique_pairs:
        mk = marker_map.get(sev_key, 'o')
        shape_handles.append(Line2D(
            [0], [0], marker=mk, linestyle='None', color='#34495e', 
            markersize=6, label=f'{sev_key[0]}% + {sev_key[1]}%'
        ))
    if shape_handles: fig.legend(
        handles=shape_handles, loc='lower center', ncol=3, fontsize=8.6, 
        framealpha=0.9, title='Marker shape = severity pair (a + b)'
    )
    fig.suptitle(
        'Compound Interaction Effects — Deviation from Independent Additivity\n'
        'x-axis: expected (additive) degradation  |  y-axis: how much worse/better the compound is',
        fontsize=12, y=1.03)
    fig.subplots_adjust(top=0.9, bottom=0.18, wspace=0.24)
    _save_fig(fig, output_dir, 'fig09_interaction_scatter')


# Figure 10: Sensitivity Ranking — ALL conditions, multiple metrics
def fig10_sensitivity_ranking(results, output_dir, clean_metrics=None):
    '''Compact sensitivity ranking bar chart.

    Rows:
        • 1 row per basic condition type (mean across selected benchmark severities present in data).
        • 1 row per compound pair (mean across all sa/sb combos for that pair) — typically 4-6 rows, marked with diagonal hatching.

    Two-metric encoding:
        ▬  BLEU-4 drop   (pp, colored bar, bottom axis)
        ◆  ROUGE-L drop  (pp, dark dots, same bottom axis)
    '''
    if clean_metrics is None: clean_metrics = results.get('clean', {}).get('metrics', {})
    clean_bleu  = clean_metrics.get('bleu4', 0)
    clean_rouge = clean_metrics.get('rouge_l', 0)
    if not clean_bleu: return

    # ── Identify up to 3 representative basic severity levels ─────────────────
    basic_sevs_present = sorted(set(
        int(k.split('_')[1]) for k in results
        if '_' in k and '+' not in k and k not in ('clean', 'meta') and k.split('_')[0] in BASIC_ORDER
    ))
    # prefer 5, 10, 20 (benchmark levels); fallback to first 3 present
    preferred = [5, 10, 20]
    show_sevs = [s for s in preferred if s in basic_sevs_present]
    if len(show_sevs) < 3: show_sevs = basic_sevs_present[:3]

    # ── Build rows: aggregated basic + aggregated compound ────────────────────
    rankings = []
    for ctype in BASIC_ORDER: # Basic: one averaged row per basic condition type across selected severities
        basic_bleu, basic_rouge = [], []
        for sev_pct in show_sevs:
            cond = f'{ctype}_{sev_pct:02d}'
            m    = results.get(cond, {}).get('metrics', {})
            bleu = m.get('bleu4')
            if bleu is None: continue

            basic_bleu.append(clean_bleu - bleu)
            if clean_rouge and m.get('rouge_l') is not None: basic_rouge.append(clean_rouge - m['rouge_l'])

        if not basic_bleu: continue
        rankings.append({
            'label':      f"{ctype} (avg)",
            'color':      CONDITION_COLORS.get(ctype, '#7f8c8d'),
            'bleu_drop':  float(np.mean(basic_bleu)),
            'rouge_drop': float(np.mean(basic_rouge)) if basic_rouge else None,
            'hatch':      '',
        })

    pair_buckets = defaultdict(lambda: {'bleu': [], 'rouge_l': []})
    for cond_name, cond_data in results.items(): # Compound: one averaged row per canonical pair
        if '+' not in cond_name or cond_name in ('clean', 'meta'): continue
        p = _parse_compound_name(cond_name)
        if p is None: continue

        pair = _canonical_pair(p[0], p[2])
        m    = cond_data.get('metrics', {})
        bleu = m.get('bleu4')
        if bleu is None: continue

        pair_buckets[pair]['bleu'].append(clean_bleu - bleu)
        if clean_rouge and m.get('rouge_l') is not None: pair_buckets[pair]['rouge_l'].append(clean_rouge - m['rouge_l'])

    for pair, buckets in pair_buckets.items():
        if not buckets['bleu']: continue
        mean_bleu  = float(np.mean(buckets['bleu']))
        mean_rouge = float(np.mean(buckets['rouge_l'])) if buckets['rouge_l'] else None
        rankings.append({
            'label': f"{COMPOUND_PAIR_LABELS.get(pair, pair)} (avg)",
            'color': COMPOUND_PAIR_COLORS.get(pair, '#7f8c8d'),
            'bleu_drop': mean_bleu, 'rouge_drop': mean_rouge, 'hatch': '///',
        })

    rankings.sort(key=lambda x: x['bleu_drop'], reverse=True)
    if not rankings: return
    N       = len(rankings)
    fig_h   = max(7, N * 0.52 + 2.0)
    fig, ax = plt.subplots(figsize=(13.2, fig_h))
    y_pos   = np.arange(N)

    # ── BLEU-4 drop bars ──────────────────────────────────────────────────────
    bars = ax.barh(
        y_pos, [r['bleu_drop'] for r in rankings],
        height=0.70, alpha=0.82, edgecolor='white', linewidth=0.5, zorder=3,
        color=[r['color'] for r in rankings],
        hatch=[r['hatch'] for r in rankings],
    )

    # ── ROUGE drop ◆ (same bottom axis) ──────────────────────────────────────
    rouge_vals = [r['rouge_drop'] for r in rankings]
    rouge_ys   = list(y_pos)
    rv_valid   = [(rv, ry) for rv, ry in zip(rouge_vals, rouge_ys) if rv is not None]
    if rv_valid: ax.scatter(
        *zip(*[(rv, ry) for rv, ry in rv_valid]), marker='D', s=45, color='#2c3e50', 
        zorder=5, alpha=0.85, label='ROUGE-L drop (pp, bottom axis)'
    )
    ax.set_xlabel('BLEU-4 drop (pp)  ▬     ROUGE-L drop (pp)  ◆\n'
                  '(absolute point-change from clean baseline)', fontsize=10)
    ax.set_xlim(left=0)

    # ── Y-axis labels ─────────────────────────────────────────────────────────
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r['label'] for r in rankings], fontsize=9)
    ax.invert_yaxis()
    ax.grid(False)

    # Light reference lines only within actual data range to avoid axis expansion.
    max_bleu_drop = max(r['bleu_drop'] for r in rankings)
    thr_guides = [thr for thr in [2, 5, 10, 15] if thr <= max_bleu_drop + 0.5]
    for thr in thr_guides: ax.axvline(thr, color='gray', linestyle=':', linewidth=0.9, alpha=0.4)
    ax.set_title(
        f'Sensitivity Ranking — {N} entries  (basic means + compound pair means, worst first)\n'
        f'▬ BLEU-4 drop   ◆ ROUGE-L drop   (/// = compound mean, solid = basic mean)',
        fontsize=11, pad=10
    )
    compound_pairs_present = {
        _canonical_pair(p[0], p[2]) for r in rankings if '(avg)' in r['label']
        for p in [_parse_compound_name(r['label'].split(' (')[0].replace(' + ', '+'))] if p
    }
    # Simpler: just gather from pair_buckets
    compound_pairs_present = set(pair_buckets.keys())
    legend_handles = [
        *[Patch(color=CONDITION_COLORS[c], label=f'{c} avg') for c in BASIC_ORDER],
        *[Patch(color=COMPOUND_PAIR_COLORS.get(pair, '#7f8c8d'), hatch='///', 
                label=f'{COMPOUND_PAIR_LABELS.get(pair, pair)} avg') for pair in sorted(compound_pairs_present)],
        Line2D([0], [0], marker='D', color='#2c3e50', linestyle='None', markersize=6, label='ROUGE-L drop (◆)'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8.8, ncol=2, framealpha=0.92)
    fig.subplots_adjust(left=0.28, right=0.98, top=0.90, bottom=0.12)
    _save_fig(fig, output_dir, 'fig10_sensitivity_ranking')


# Figure 11: Output Length Ratio
def fig11_output_length_ratio(results, severity_levels, output_dir):
    '''Mean output/reference length ratio per condition.

    Solid lines = single-noise basic conditions.
    Ratio = 1.0 is ideal; ratio < 1 = under-generation; ratio > 1 = over-generation.
    '''
    fig, ax = plt.subplots(figsize=(11, 6))
    sevs_pct = [s * 100 for s in severity_levels]

    # Basic conditions
    for ctype in BASIC_ORDER:
        ratios, xs = [], []
        for sev, sev_pct in zip(severity_levels, sevs_pct):
            cond = f'{ctype}_{int(sev * 100):02d}'
            ps = _get_condition_samples(results, cond)
            if ps:
                r = np.mean([v['output_length_ratio'] for v in ps.values()])
                ratios.append(r)
                xs.append(sev_pct)
        if xs: ax.plot(
            xs, ratios, linewidth=2.2, markersize=7, color=CONDITION_COLORS[ctype], 
            marker=CONDITION_MARKERS[ctype], label=CONDITION_LABELS[ctype]
        )

    # Clean baseline
    clean_ps = _get_condition_samples(results, 'clean')
    if clean_ps:
        clean_ratio = np.mean([v['output_length_ratio'] for v in clean_ps.values()])
        ax.axhline(clean_ratio, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Clean mean ratio ({clean_ratio:.3f})')

    ax.axhline(1.0, color='black', linestyle=':', linewidth=1.0, alpha=0.4)
    ax.text(sevs_pct[-1] * 0.02, 1.01, 'ideal (1.0)', fontsize=10, color='gray')
    ax.set_xlabel('Severity (%)')
    ax.set_ylabel('Mean Output / Reference Length Ratio')
    ax.set_title('Output Length Distortion Under Temporal Misalignment\n< 1.0 = under-generation;  > 1.0 = over-generation')
    ax.legend(ncol=2, fontsize=10, loc='best')
    ax.set_xlim(left=0)
    _save_fig(fig, output_dir, 'fig11_output_length_ratio')


# Master function
def generate_all_figures(results_dir: str, output_dir: str):
    '''Load result JSON files and generate all analysis figures.

    Expected files under results_dir/raw/:
      knee_point.json  – 41 conditions (clean + 4×10 basic), severities 5–50%
      benchmark.json   – 49 conditions (clean + 12 basic + 36 compound), 3 severities
      train_eval.json  – train subset, same structure as benchmark

    Figures are saved as .png under output_dir/.
    '''
    results_dir = Path(results_dir)
    output_dir  = Path(output_dir)
    knee_path  = results_dir / 'raw' / 'knee_point.json'
    bench_path = results_dir / 'raw' / 'benchmark.json'
    train_path = results_dir / 'raw' / 'train_eval.json'
    knee_sevs  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    benchmark_results = None

    # ── Knee-point figures (rely on the 10-level dense sweep) ─────────────────
    if knee_path.exists():
        print('▶ Knee-point figures …')
        with open(knee_path) as f:
            knee_results = json.load(f)
        cm = knee_results.get('clean', {}).get('metrics', {})

        fig01_bleu_degradation(knee_results, knee_sevs, output_dir, clean_bleu=cm.get('bleu4'))
        fig02_unified_dashboard_heatmap(knee_results, knee_sevs, output_dir, clean_metrics=cm)
        fig03_bleu_vs_rouge(knee_results, knee_sevs, output_dir)
        
        # fig5: compound bottom row deferred until after bench block.
        fig06_transition_matrices(knee_results, knee_sevs, output_dir)
        fig07_length_vs_bleu_drop(knee_results, knee_sevs, output_dir)
        fig08_vulnerability_profile(knee_results, knee_sevs, output_dir)
        fig11_output_length_ratio(knee_results, knee_sevs, output_dir)

    # ── Benchmark figures (all 49 conditions; compound interactions) ──────────
    if bench_path.exists():
        print('▶ Benchmark figures …')
        with open(bench_path) as f:
            benchmark_results = json.load(f)
            
        cm = benchmark_results.get('clean', {}).get('metrics', {})
        fig09_interaction_scatter(benchmark_results, output_dir, clean_metrics=cm)
        fig10_sensitivity_ranking(benchmark_results, output_dir, clean_metrics=cm)

    # ── Fig 5: basic panels from knee (10 levels) + compound from bench ───────
    # Deferred here so benchmark_results is always defined before use.
    if knee_path.exists(): fig05_failure_distribution(
        knee_results, knee_sevs, output_dir, 
        compound_results=benchmark_results if benchmark_results is not None else knee_results
    )

    # ── Train-vs-dev (needs both knee_point and train_eval) ──────────────────
    if knee_path.exists() and train_path.exists():
        print('▶ Train-vs-dev figure …')
        with open(train_path) as f:
            train_results = json.load(f)
        fig04_train_vs_dev(knee_results, train_results, knee_sevs, output_dir)
    print(f'✓ All figures saved to {output_dir}')