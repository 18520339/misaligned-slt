'''Qualitative translation example selection and formatting.

Selects representative samples showing different failure modes and formats
them for inclusion in the paper.
'''
import json
from pathlib import Path
from analysis.failure_modes import classify_failure_mode

# Failure mode abbreviations for compact display
MODE_ABBREV = {
    'Acceptable': 'ACCEPT',
    'Under-generation': 'UNDER',
    'Hallucination': 'HALL',
    'Partial match': 'PART',
    'Repetition': 'REPET',
    'Incoherent': 'INCOH',
}

def select_representative_samples(results_json: dict, n_per_category: int = 2) -> list:
    '''Select representative samples for qualitative analysis.

    Selection criteria:
        1. 2 samples where HT causes hallucination
        2. 2 samples where TT causes under-generation
        3. 1 sample where contamination causes sentence mixing
        4. 1 sample with repetition under misalignment
        5. 1 surprisingly robust sample
        6. 1 sample showing compound condition effects

    Returns:
        List of dicts with sample info and selection reason.
    '''
    selections = []
    clean_samples = results_json.get('clean', {}).get('predictions', {})

    # Helper: get merged per-sample entries for a condition
    def get_samples(cond_name):
        samples = results_json.get(cond_name, {}).get('predictions', {})
        return samples if isinstance(samples, dict) else {}

    # 1. HT causing hallucination (novel_token_rate > 0.5)
    ht_hall = []
    for sev in [20, 30, 40]:
        cond = f'HT_{sev:02d}'
        samples = get_samples(cond)
        for name, m in samples.items():
            if m.get('novel_token_rate', 0) > 0.5 and m.get('sentence_bleu', 0) < 2:
                clean_bleu = clean_samples.get(name, {}).get('sentence_bleu', 0)
                if clean_bleu > 0.3:  # was decent originally
                    ht_hall.append((name, cond, clean_bleu, m.get('sentence_bleu', 0)))
                    
    ht_hall.sort(key=lambda x: x[2] - x[3], reverse=True)  # sort by drop
    for item in ht_hall[:n_per_category]:
        selections.append({
            'name': item[0], 'reason': f'HT hallucination ({item[1]})',
            'highlight_condition': item[1], 'category': 'hallucination'})

    # 2. TT causing under-generation (output_length_ratio < 0.5)
    tt_under = []
    for sev in [20, 30, 40]:
        cond = f'TT_{sev:02d}'
        samples = get_samples(cond)
        for name, m in samples.items():
            if m.get('output_length_ratio', 1) < 0.5 and m.get('sentence_bleu', 0) < 4:
                clean_bleu = clean_samples.get(name, {}).get('sentence_bleu', 0)
                if clean_bleu > 0.3:
                    tt_under.append((name, cond, clean_bleu, m.get('output_length_ratio', 1)))
                    
    tt_under.sort(key=lambda x: x[3])  # sort by shortest ratio
    for item in tt_under[:n_per_category]:
        selections.append({
            'name': item[0], 'reason': f'TT under-generation ({item[1]})',
            'highlight_condition': item[1], 'category': 'under-generation'})

    # 3. Contamination causing sentence mixing
    for sev in [20, 30]:
        for ctype in ['HC', 'TC']:
            cond = f'{ctype}_{sev:02d}'
            samples = get_samples(cond)
            for name, sample in samples.items():
                # Check if output contains tokens from adjacent sentence reference
                if sample.get('novel_token_rate', 0) > 0.3:
                    if len(selections) < 5 or not any(s['category'] == 'mixing' for s in selections):
                        selections.append({
                            'name': name, 'reason': f'Contamination mixing ({cond})',
                            'highlight_condition': cond, 'category': 'mixing'})
                        break
            else: continue
            break

    # 4. Repetition under misalignment
    for sev in [20, 30, 40, 50]:
        for ctype in ['HT', 'TT', 'HC', 'TC']:
            cond = f'{ctype}_{sev:02d}'
            samples = get_samples(cond)
            for name, m in samples.items():
                if m.get('has_repetition', False):
                    if not any(s['category'] == 'repetition' for s in selections):
                        selections.append({
                            'name': name, 'reason': f'Repetition ({cond})',
                            'highlight_condition': cond, 'category': 'repetition'})
                        break
            else: continue
            break

    # 5. Surprisingly robust sample (high BLEU even at 30% truncation)
    for ctype in ['HT', 'TT']:
        cond = f'{ctype}_30'
        samples = get_samples(cond)
        robust = [(n, m.get('sentence_bleu', 0)) for n, m in samples.items() if m.get('sentence_bleu', 0) > 3]
        robust.sort(key=lambda x: x[1], reverse=True)
        if robust and not any(s['category'] == 'robust' for s in selections):
            selections.append({
                'name': robust[0][0], 'reason': f'Robust under {cond}',
                'highlight_condition': cond, 'category': 'robust'})

    # 6. Compound condition effect
    compound_conds = [k for k in results_json if '+' in k and k != 'meta']
    for cond in compound_conds[:5]:
        samples = get_samples(cond)
        for name, m in samples.items():
            if m.get('sentence_bleu', 0) < 1.5 and name in clean_samples:
                if clean_samples[name].get('sentence_bleu', 0) > 4:
                    if not any(s['category'] == 'compound' for s in selections):
                        selections.append({
                            'name': name, 'reason': f'Compound degradation ({cond})',
                            'highlight_condition': cond, 'category': 'compound'})
                        break
    return selections[:8]  # Cap at 8


def format_example_table(sample_name: str, results_json: dict, conditions_to_show: list = None) -> str:
    # Format a single sample's predictions across conditions as a markdown table
    clean_sample = results_json.get('clean', {}).get('predictions', {}).get(sample_name, {})
    ref_text = clean_sample.get('txt_ref', 'N/A')

    if conditions_to_show is None:
        conditions_to_show = ['clean']
        for ctype in ['HT', 'TT', 'HC', 'TC']:
            for sev in [10, 20, 30]:
                conditions_to_show.append(f'{ctype}_{sev:02d}')

    lines = [
        f"### Sample: {sample_name}\n",
        f"**Reference:** {ref_text}\n",
        "| Condition | Translation | sBLEU | Mode |",
        "|-----------|-------------|-------|------|",
    ]
    for cond in conditions_to_show:
        if cond not in results_json: continue
        sample = results_json.get(cond, {}).get('predictions', {}).get(sample_name, {})
        if not sample: continue

        txt = sample.get('txt_hyp', '')[:80]
        sbleu = sample.get('sentence_bleu', 0)
        mode = classify_failure_mode(
            sample.get('sentence_bleu', 0),
            sample.get('novel_token_rate', 0),
            sample.get('output_length_ratio', 1),
            sample.get('has_repetition', False))
        mode_abbrev = MODE_ABBREV.get(mode, mode[:5])
        lines.append(f"| {cond} | {txt} | {sbleu:.2f} | {mode_abbrev} |")
    return '\n'.join(lines)


def generate_qualitative_report(results_dir: str, output_dir: str):
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try loading benchmark results first, then knee_point
    for fname in ['benchmark.json', 'knee_point.json']:
        fpath = results_dir / 'raw' / fname
        if fpath.exists():
            with open(fpath) as f:
                results = json.load(f)
            break
    else:
        print("No results files found for qualitative analysis.")
        return

    selections = select_representative_samples(results)
    if not selections:
        print("Could not find suitable samples for qualitative analysis.")
        return

    report_lines = ["# Qualitative Translation Examples\n"]
    for sel in selections:
        report_lines.append(f"---\n**Category:** {sel['category']} | **Reason:** {sel['reason']}\n")

        # Show relevant conditions for this sample
        conds = ['clean']
        highlight = sel.get('highlight_condition', '')
        ctype = highlight.split('_')[0] if '_' in highlight else ''
        if ctype:
            for sev in [10, 20, 30]:
                conds.append(f'{ctype}_{sev:02d}')
                
        # Also add some cross-condition views
        for ct in ['HT', 'TT', 'HC', 'TC']:
            if ct != ctype:
                conds.append(f'{ct}_20')

        table = format_example_table(sel['name'], results, conds)
        report_lines.append(table + '\n')

    report_path = output_dir / 'qualitative_examples.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"Qualitative report saved to {report_path}")
    return selections