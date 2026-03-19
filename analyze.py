'''Analyze benchmark results and produce all visualizations.

Reads results/benchmark_results.json and generates:
  1. Vulnerability heatmap (heatmap.png)
  2. Degradation curves   (degradation_curves.png)
  3. Knee-point analysis   (knee_points.csv)
  4. Per-condition scores  (scores.csv)
  5. Sample translations   (sample_translations.md)
  6. Failure-type distribution (failure_distribution.png)

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
PRETTY = {
    'head_trunc': 'Head Truncation', 'tail_trunc': 'Tail Truncation',
    'head_contam': 'Head Contamination', 'tail_contam': 'Tail Contamination',
    'head_trunc_tail_trunc': 'Head Trunc + Tail Trunc',
    'head_trunc_tail_contam': 'Head Trunc + Tail Contam',
    'head_contam_tail_trunc': 'Head Contam + Tail Trunc',
    'head_contam_tail_contam': 'Head Contam + Tail Contam',
}
SEVERITY_LEVELS = [10, 20, 30, 40, 50]


def bleu4_matrix(metrics): # Return (8 x 5) numpy array of BLEU-4 scores, rows=conditions, cols=severity
    mat = np.full((len(MISALIGN_ORDER), len(SEVERITY_LEVELS)), np.nan)
    for ri, cond in enumerate(MISALIGN_ORDER):
        cond_data = metrics.get(cond, {})
        for ci, sev in enumerate(SEVERITY_LEVELS):
            entry = cond_data.get(str(sev), {})
            if 'bleu4' in entry: mat[ri, ci] = entry['bleu4']
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


# OUTPUT 1: Vulnerability Heatmap
def plot_heatmap(metrics, clean_bleu, out_dir):
    mat = bleu4_matrix(metrics)
    knee_col = []
    
    for ri, cond in enumerate(MISALIGN_ORDER):
        row = mat[ri]
        valid = ~np.isnan(row)
        if valid.any():
            x = np.array(SEVERITY_LEVELS)[valid]
            y = row[valid]
            kx, _ = find_knee(x, y)
            knee_col.append(f'{int(kx)}%')
        else:
            knee_col.append('—')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        mat, ax=ax, annot=True, fmt='.1f', linewidths=0.5,
        vmin=max(0, clean_bleu * 0.1), vmax=clean_bleu,
        cmap=sns.color_palette('RdYlGn', as_cmap=True),
        xticklabels=[f'{s}%' for s in SEVERITY_LEVELS], 
        yticklabels=[PRETTY[c] for c in MISALIGN_ORDER],
        cbar_kws={'label': 'BLEU-4'}
    )

    # Knee column on the right
    for ri, txt in enumerate(knee_col):
        ax.text(len(SEVERITY_LEVELS) + 0.5, ri + 0.5, txt, ha='center', va='center', fontsize=9, fontweight='bold')
        
    ax.text(len(SEVERITY_LEVELS) + 0.5, -0.3, 'Knee', ha='center', fontsize=9, fontweight='bold')
    ax.set_title(f'MSKA-SLT Vulnerability Heatmap on Phoenix-2014T\n'
                 f'(Clean baseline BLEU-4 = {clean_bleu:.2f})', fontsize=12)
    
    ax.set_xlabel('Severity')
    ax.set_ylabel('')
    fig.tight_layout()
    path = os.path.join(out_dir, 'heatmap.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f'  Saved {path}')


# OUTPUT 2: Degradation Curves
def plot_degradation_curves(metrics, clean_bleu, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = [0] + SEVERITY_LEVELS
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    colors = plt.cm.tab10.colors
    ax.axhline(clean_bleu, color='grey', linestyle='--', linewidth=1, label=f'Clean ({clean_bleu:.1f})')

    for i, cond in enumerate(MISALIGN_ORDER):
        cond_data = metrics.get(cond, {})
        y = [clean_bleu]
        for sev in SEVERITY_LEVELS:
            v = cond_data.get(str(sev), {}).get('bleu4')
            y.append(v if v is not None else np.nan)
            
        y = np.array(y, dtype=float)
        ax.plot(
            x, y, marker=markers[i % len(markers)], color=colors[i % len(colors)], 
            label=PRETTY[cond], linewidth=1.5, markersize=6
        )

        valid = ~np.isnan(y[1:])
        if valid.any(): # mark knee
            kx, ky = find_knee(np.array(SEVERITY_LEVELS)[valid], y[1:][valid])
            ax.plot(kx, ky, marker='|', color=colors[i % len(colors)], markersize=18, markeredgewidth=2)

    ax.set_xlabel('Severity (%)', fontsize=11)
    ax.set_ylabel('BLEU-4', fontsize=11)
    ax.set_title('BLEU-4 Degradation Under Temporal Misalignment', fontsize=12)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'{v}%' for v in x])
    ax.legend(fontsize=8, loc='lower left', ncol=2)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    path = os.path.join(out_dir, 'degradation_curves.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f'  Saved {path}')


# OUTPUT 3: Knee-Point Analysis CSV
def compute_knee_points(metrics, clean_bleu, out_dir):
    rows = []
    for cond in MISALIGN_ORDER:
        cond_data = metrics.get(cond, {})
        sevs, vals = [], []
        for sev in SEVERITY_LEVELS:
            v = cond_data.get(str(sev), {}).get('bleu4')
            if v is not None:
                sevs.append(sev)
                vals.append(v)
                
        if not sevs: continue
        kx, ky = find_knee(np.array(sevs), np.array(vals))
        drop = (clean_bleu - ky) / clean_bleu * 100 if clean_bleu > 0 else 0
        rows.append({
            'misalignment_type': cond,
            'knee_severity': int(kx),
            'bleu_at_knee': round(ky, 2),
            'bleu_at_clean': round(clean_bleu, 2),
            'relative_drop_pct': round(drop, 1),
        })
    rows.sort(key=lambda r: r['knee_severity'])

    path = os.path.join(out_dir, 'knee_points.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f'  Saved {path}')


# OUTPUT 4: Per-Condition Summary Table CSV
def generate_scores_csv(metrics, clean_bleu, out_dir):
    rows = []
    c = metrics['clean']
    rows.append({ # Clean row
        'misalignment_type': 'clean',
        'severity': 0,
        'bleu4': c['bleu4'],
        'bleu4_drop': '—',
        'rouge_l': c.get('rouge_l'),
        'meteor': c.get('meteor'),
    })
    for cond in MISALIGN_ORDER:
        cond_data = metrics.get(cond, {})
        for sev in SEVERITY_LEVELS:
            entry = cond_data.get(str(sev))
            if entry is None: continue
            b = entry['bleu4']
            drop = (clean_bleu - b) / clean_bleu * 100 if clean_bleu else 0
            rows.append({
                'misalignment_type': cond,
                'severity': sev,
                'bleu4': b,
                'bleu4_drop': f'{drop:+.1f}%',
                'rouge_l': entry.get('rouge_l'),
                'meteor': entry.get('meteor'),
            })

    path = os.path.join(out_dir, 'scores.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f'  Saved {path}')


# OUTPUT 5: Sample Translation Comparison (Markdown)
def generate_sample_translations(data, out_dir, n_samples=5):
    translations = data.get('translations', {})
    clean_trans = {t['name']: t for t in translations.get('clean', [])}
    lines = ['# Sample Translation Comparison\n', 'Showing random samples for each misalignment type at **20% severity**.\n']

    for cond in MISALIGN_ORDER:
        samples = translations.get(f'{cond}_20', [])
        if not samples: continue
        lines.append(f'\n## {PRETTY[cond]} (20%)\n')
        chosen = random.sample(samples, min(n_samples, len(samples)))

        for s in chosen:
            name = s['name']
            ref = s['ref']
            hyp_mis = s['hyp']
            hyp_clean = clean_trans.get(name, {}).get('hyp', '—')
            ftype = classify_failure(ref, hyp_mis)

            lines.append(f'### Sample: `{name}`\n')
            lines.append(f'| | Text |')
            lines.append(f'|---|---|')
            lines.append(f'| **Reference** | {ref} |')
            lines.append(f'| **Clean output** | {hyp_clean} |')
            lines.append(f'| **Misaligned output** | {hyp_mis} |')
            lines.append(f'| **Failure type** | {ftype} |')
            lines.append('')

    path = os.path.join(out_dir, 'sample_translations.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  Saved {path}')


# OUTPUT 6: Failure-Type Distribution (Stacked Bar Chart)
def plot_failure_distribution(data, out_dir):
    translations = data.get('translations', {})
    clean_trans = {t['name']: t for t in translations.get('clean', [])}
    failure_types = ['acceptable', 'under-generation', 'hallucination', 'incoherent']
    colours = {'acceptable': '#4CAF50', 'under-generation': '#FF9800',
               'hallucination': '#F44336', 'incoherent': '#9C27B0'}

    labels = []
    counts = {ft: [] for ft in failure_types}
    for cond in MISALIGN_ORDER: # Gather failure distributions for every condition at every severity
        for sev in SEVERITY_LEVELS:
            key = f'{cond}_{sev}'
            samples = translations.get(key, [])
            if not samples: continue
            
            dist = {ft: 0 for ft in failure_types}
            for s in samples:
                ft = classify_failure(s['ref'], s['hyp'])
                dist[ft] += 1
                
            total = len(samples)
            labels.append(f'{PRETTY[cond]}\n{sev}%')
            for ft in failure_types:
                counts[ft].append(dist[ft] / total * 100)

    if not labels:
        print('  [Skipping failure distribution — no per-sample translations]')
        return

    fig, ax = plt.subplots(figsize=(min(20, max(14, len(labels) * 0.45)), 7))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))

    for ft in failure_types:
        vals = np.array(counts[ft])
        ax.bar(x, vals, bottom=bottom, label=ft.capitalize(), color=colours[ft], width=0.7)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Percentage of samples (%)')
    ax.set_title('Failure-Type Distribution Across Misalignment Conditions')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 105)
    
    fig.tight_layout()
    path = os.path.join(out_dir, 'failure_distribution.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f'  Saved {path}')


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

    print('[1/6] Vulnerability heatmap')
    plot_heatmap(metrics, clean_bleu, out_dir)

    print('[2/6] Degradation curves')
    plot_degradation_curves(metrics, clean_bleu, out_dir)

    print('[3/6] Knee-point analysis')
    compute_knee_points(metrics, clean_bleu, out_dir)

    print('[4/6] Per-condition scores CSV')
    generate_scores_csv(metrics, clean_bleu, out_dir)

    print('[5/6] Sample translation comparison')
    generate_sample_translations(data, out_dir)

    print('[6/6] Failure-type distribution')
    plot_failure_distribution(data, out_dir)


if __name__ == '__main__':
    main()