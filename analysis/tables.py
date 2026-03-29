import json, os, csv
from pathlib import Path
from collections import OrderedDict
import numpy as np
from analysis.knee_point import detect_all_knee_points, compute_degradation_rate
from analysis.failure_modes import classify_all_predictions, failure_mode_distribution

BASIC_ORDER = ['HT', 'TT', 'HC', 'TC']

def _parse_compound_name(cond_name): # 'HT_05+TT_10' → ('HT', 0.05, 'TT', 0.10) or None on failure
    parts = cond_name.split('+')
    if len(parts) != 2: return None
    try:
        a_type, a_pct = parts[0].rsplit('_', 1)
        b_type, b_pct = parts[1].rsplit('_', 1)
        return a_type.strip(), int(a_pct) / 100, b_type.strip(), int(b_pct) / 100
    except (ValueError, AttributeError):
        return None

def _canonical_pair(a_type, b_type): # Return canonical pair key with lower BASIC_ORDER index first
    if BASIC_ORDER.index(a_type) <= BASIC_ORDER.index(b_type):
        return f'{a_type}+{b_type}'
    return f'{b_type}+{a_type}'

def _load_results(results_dir, filename):
    path = Path(results_dir) / 'raw' / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# Table 1: Knee Point Summary
def table1_knee_points(results: dict, severity_levels: list) -> str:
    knees = detect_all_knee_points(results, severity_levels)
    clean_bleu = results.get('clean', {}).get('metrics', {}).get('bleu4', 0)
    classifications = classify_all_predictions(results)
    distributions = failure_mode_distribution(classifications)

    lines = ['# Table 1: Knee Point Summary\n']
    lines.append('| Condition | Knee Severity | BLEU at Knee | Clean BLEU | '
                 'Abs Drop | Rel Drop (%) | WER at Knee | Dominant Failure Mode |')
    lines.append('|-----------|--------------|-------------|-----------|'
                 '---------|-------------|------------|----------------------|')

    for ctype in ['HT', 'TT', 'HC', 'TC']:
        k = knees.get(ctype, {}).get('bleu4')
        if not k:
            lines.append(f'| {ctype} | N/A | N/A | {clean_bleu:.1f} | - | - | - | - |')
            continue

        knee_sev = k['knee_severity']
        knee_bleu = k['knee_value']
        abs_drop = clean_bleu - knee_bleu
        rel_drop = abs_drop / max(clean_bleu, 0.01) * 100

        # WER at knee
        wer_k = knees.get(ctype, {}).get('wer')
        wer_at_knee = wer_k['knee_value'] if wer_k else 'N/A'

        # Dominant failure mode at knee severity
        cond_at_knee = f'{ctype}_{int(knee_sev * 100):02d}'
        dom_mode = 'N/A'
        if cond_at_knee in distributions:
            dist = distributions[cond_at_knee]
            dom_mode = max([m for m in dist if m != '_total'], key=lambda m: dist[m], default='N/A')

        lines.append(
            f"| {ctype} | {knee_sev*100:.0f}% | {knee_bleu:.1f} | {clean_bleu:.1f} | "
            f"{abs_drop:.1f} | {rel_drop:.1f}% | "
            f"{wer_at_knee if isinstance(wer_at_knee, str) else f'{wer_at_knee:.1f}'} | "
            f"{dom_mode} |")
    return '\n'.join(lines)


# Table 2: Cross-Metric Robustness Summary (aggregated; non-duplicate of CSV)
def table2_executive_summary(results: dict, severity_levels: list) -> str:
    clean_m = results.get('clean', {}).get('metrics', {})
    clean_wer = clean_m.get('wer', 0.0)
    clean_bleu = clean_m.get('bleu4', 0.0)
    clean_rouge = clean_m.get('rouge_l', 0.0)

    lines = ['# Table 2: Cross-Metric Robustness Summary\n']
    lines.append(
        'This table is intentionally aggregated (mean ± std across selected severities), '
        'so it complements the CSV instead of duplicating row-level metrics.\n'
    )
    lines.append('| Group | Mean ΔBLEU (pp) | Mean ΔWER (pp) | Mean ΔROUGE (pp)  | Worst cond (ΔBLEU) |')
    lines.append('|-------|----------------:|---------------:|------------------:|--------------------|')

    groups = OrderedDict({
        'HT': [], 'TT': [], 'HC': [], 'TC': [],
        'HT+TT': [], 'HT+HC': [], 'HT+TC': [], 'TT+HC': [], 'TT+TC': [], 'HC+TC': [],
        'All (basic+compound)': [],
    })

    def _row_delta(cond_name):
        m = results.get(cond_name, {}).get('metrics', {})
        bleu = m.get('bleu4')
        if bleu is None: return None
        return {
            'cond': cond_name,
            'd_bleu': clean_bleu - bleu,
            'd_wer': (m.get('wer', clean_wer) - clean_wer) if clean_wer else 0.0,
            'd_rouge': (clean_rouge - m.get('rouge_l', clean_rouge)) if clean_rouge else 0.0,
        }

    # Basic conditions (selected severities)
    for ctype in ['HT', 'TT', 'HC', 'TC']:
        for sev in severity_levels:
            cond = f'{ctype}_{int(sev * 100):02d}'
            d = _row_delta(cond)
            if d:
                groups[ctype].append(d)
                groups['All (basic+compound)'].append(d)

    # Compound conditions
    for cond in sorted(k for k in results if '+' in k and k != 'meta'):
        d = _row_delta(cond)
        p = _parse_compound_name(cond)
        if d and p:
            pair = _canonical_pair(p[0], p[2])
            if pair in groups: groups[pair].append(d)
            groups['All (basic+compound)'].append(d)

    for gname, rows in groups.items():
        if not rows: continue
        d_bleu = [r['d_bleu'] for r in rows]
        d_wer = [r['d_wer'] for r in rows]
        d_rouge = [r['d_rouge'] for r in rows]
        worst = max(rows, key=lambda r: r['d_bleu'])
        lines.append(
            f"| {gname} | "
            f"{sum(d_bleu)/len(d_bleu):.2f} ± {np.std(d_bleu):.2f} | "
            f"{sum(d_wer)/len(d_wer):.2f} ± {np.std(d_wer):.2f} | "
            f"{sum(d_rouge)/len(d_rouge):.2f} ± {np.std(d_rouge):.2f} | "
            f"{worst['cond']} ({worst['d_bleu']:.2f}) |"
        )
    return '\n'.join(lines)


# Table 3: Rank conditions by average BLEU-4 drop at matched severities
def table3_severity_ranking(results: dict, severity_levels: list) -> str:
    clean_bleu = results.get('clean', {}).get('metrics', {}).get('bleu4', 0)
    all_conds = sorted([k for k in results if k not in ('clean', 'meta')])
    rankings = []
    
    for cond in all_conds:
        m = results[cond].get('metrics', {})
        bleu, wer = m.get('bleu4'), m.get('wer')
        if bleu is not None: rankings.append({
            'condition': cond, 'bleu4': bleu, 'bleu_drop': clean_bleu - bleu,
            'wer': wer if wer is not None else 0, 'rouge_l': m.get('rouge_l', 0),
        })

    rankings.sort(key=lambda x: x['bleu_drop'], reverse=True)
    lines = ['# Table 3: Condition Severity Ranking (by BLEU-4 drop)\n']
    lines.append('| Rank | Condition | BLEU-4 | BLEU Drop | WER | ROUGE |')
    lines.append('|------|-----------|--------|-----------|-----|-------|')
    for i, r in enumerate(rankings, 1):
        lines.append(
            f"| {i} | {r['condition']} | {r['bleu4']:.1f} | "
            f"-{r['bleu_drop']:.1f} | {r['wer']:.1f} | {r['rouge_l']:.1f} |")

    # Key finding
    basic_avgs = {}
    for ctype in ['HT', 'TT', 'HC', 'TC']:
        drops = [r['bleu_drop'] for r in rankings if r['condition'].startswith(ctype + '_')]
        if drops: basic_avgs[ctype] = sum(drops) / len(drops)

    if len(basic_avgs) >= 2:
        sorted_avgs = sorted(basic_avgs.items(), key=lambda x: x[1], reverse=True)
        worst, best = sorted_avgs[0], sorted_avgs[-1]
        ratio = worst[1] / max(best[1], 0.01)
        lines.append(f'\n**Key finding:** {worst[0]} is on average {ratio:.1f}x '
                     f'more damaging than {best[0]} at matched severity levels.')
    return '\n'.join(lines)


# Export all condition metrics to CSV
def export_metrics_csv(results: dict, output_path: str):
    rows = []
    for cond_name, cond_data in results.items():
        if cond_name == 'meta': continue
        m = cond_data.get('metrics', {})
        row = {
            'condition': cond_name, 
            'delta_s': cond_data.get('delta_s', 0), 'delta_e': cond_data.get('delta_e', 0),
            'num_evaluated': cond_data.get('num_evaluated', 0),
            'wer': m.get('wer', ''), 'bleu4': m.get('bleu4', ''), 'rouge_l': m.get('rouge_l', ''), 
        }
        rows.append(row)

    if rows:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

# Generate all tables from saved JSON results
def generate_all_tables(results_dir: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    knee_results = _load_results(results_dir, 'knee_point.json')
    bench_results = _load_results(results_dir, 'benchmark.json')
    knee_sevs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    bench_sevs = [0.05, 0.10, 0.20]
    
    if knee_results:
        print('▶ Knee point tables...')
        t1 = table1_knee_points(knee_results, knee_sevs)
        (output_dir / 'table1_knee_points.md').write_text(t1, encoding='utf-8')
        export_metrics_csv(knee_results, str(output_dir / 'knee_point_metrics.csv'))

    if bench_results:
        print('▶ Benchmark tables...')
        t2 = table2_executive_summary(bench_results, bench_sevs)
        (output_dir / 'table2_executive_summary.md').write_text(t2, encoding='utf-8')

        t3 = table3_severity_ranking(bench_results, bench_sevs)
        (output_dir / 'table3_severity_ranking.md').write_text(t3, encoding='utf-8')
        export_metrics_csv(bench_results, str(output_dir / 'benchmark_metrics.csv'))
    print('✓ All tables saved to', output_dir)