'''Evaluation for GFSLT-VLP sign language translation models.

Runs inference under various misalignment conditions and collects results
(text predictions, metrics) into a structured JSON file.

Metrics: BLEU-4 (sacrebleu), ROUGE-L, per-sample sentence BLEU.
No gloss/WER/CTC — this is a gloss-free pipeline.
'''
import os, json, time
from collections import defaultdict, OrderedDict
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Use sacrebleu for corpus BLEU (standard in MT research)
import sacrebleu
from rouge_score import rouge_scorer


def compute_bleu(references, hypotheses):
    '''Compute corpus-level BLEU-4 using sacrebleu.'''
    refs = [references]  # sacrebleu expects list of ref lists
    result = sacrebleu.corpus_bleu(hypotheses, refs)
    return {
        'bleu1': result.precisions[0],
        'bleu2': result.precisions[1],
        'bleu3': result.precisions[2],
        'bleu4': result.score,
    }


def compute_rouge(references, hypotheses):
    '''Compute corpus-level ROUGE-L.'''
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    scores = [scorer.score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    rouge_l = np.mean([s['rougeL'].fmeasure for s in scores]) * 100
    return rouge_l


def collect_batch_predictions(
    model, names, texts, pixel_values, pixel_mask,
    sample_results, tokenizer, generate_cfg=None,
):
    '''Generate translations for one batch and collect into sample_results.

    Args:
        model: SLTModel instance (AR or BD decoder).
        names: Tuple of sample names from batch.
        texts: Tuple of reference texts from batch.
        pixel_values: (B, T, 77, 3) on device.
        pixel_mask: (B, T) on device.
        sample_results: Dict to accumulate {name: {txt_hyp, txt_ref}}.
        tokenizer: HuggingFace mBART tokenizer for decoding.
        generate_cfg: Generation config dict.
    '''
    gen_cfg = generate_cfg or {}
    decoder_type = getattr(model, 'decoder_type', None)
    if decoder_type is None:
        # Heuristic: BD decoder wraps a BlockDiffusionDecoder; otherwise AR.
        decoder_type = 'bd' if hasattr(model, 'bd_decoder') else 'ar'

    # Generate token IDs
    if decoder_type == 'bd':
        generated_ids = model.generate(
            pixel_values=pixel_values, pixel_mask=pixel_mask,
            max_length=gen_cfg.get('max_length', 100),
            diffusion_steps=gen_cfg.get('diffusion_steps', 128),
        )
    else:
        tgt_lang = getattr(tokenizer, 'tgt_lang', None) or 'de_DE'
        decoder_start_token_id = tokenizer.lang_code_to_id.get(
            tgt_lang, tokenizer.bos_token_id,
        )
        generated_ids = model.generate(
            pixel_values=pixel_values, pixel_mask=pixel_mask,
            max_new_tokens=gen_cfg.get('max_new_tokens', gen_cfg.get('max_length', 150)),
            num_beams=gen_cfg.get('num_beams', 4),
            length_penalty=gen_cfg.get('length_penalty', 1.0),
            decoder_start_token_id=decoder_start_token_id,
        )

    # Decode generated tokens
    pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    for name, txt_hyp, txt_ref in zip(names, pred_texts, texts):
        sample_results[name] = {
            'txt_hyp': txt_hyp.strip(),
            'txt_ref': txt_ref.strip(),
        }


def compute_metrics(results):
    '''Compute corpus-level and per-sample metrics from collected predictions.

    Args:
        results: Dict of {name: {txt_hyp: str, txt_ref: str}}.
    Returns:
        Dict with bleu4, rouge_l, and per_sample metrics.
    '''
    names = list(results.keys())
    txt_ref = [results[n]['txt_ref'] for n in names]
    txt_hyp = [results[n]['txt_hyp'] for n in names]

    metrics = {}

    # BLEU
    bleu_dict = compute_bleu(references=txt_ref, hypotheses=txt_hyp)
    for k, v in bleu_dict.items(): metrics[k] = v

    # ROUGE
    metrics['rouge_l'] = compute_rouge(references=txt_ref, hypotheses=txt_hyp)

    # Per-sample metrics
    per_sample = {}
    smooth = SmoothingFunction().method1
    for n in names:
        hyp, ref = results[n]['txt_hyp'], results[n]['txt_ref']
        hyp_tokens = hyp.split() if hyp else []
        ref_tokens = ref.split() if ref else ['']

        s_bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth) * 100
        ref_set = set(ref_tokens)
        novel_tokens = [t for t in hyp_tokens if t not in ref_set]
        novel_rate = len(novel_tokens) / max(len(hyp_tokens), 1)
        len_ratio = len(hyp_tokens) / max(len(ref_tokens), 1)

        has_repetition = False
        if len(hyp_tokens) >= 3:
            for i in range(len(hyp_tokens) - 2):
                if hyp_tokens[i] == hyp_tokens[i + 1] == hyp_tokens[i + 2]:
                    has_repetition = True
                    break

        per_sample[n] = {
            'sentence_bleu': s_bleu,
            'novel_token_rate': novel_rate,
            'output_length_ratio': len_ratio,
            'has_repetition': has_repetition,
            'hyp_length': len(hyp_tokens),
            'ref_length': len(ref_tokens),
        }

    metrics['per_sample'] = per_sample
    return metrics


@torch.no_grad()
def run_evaluation(
    model, dataset, conditions, output_path,
    tokenizer, batch_size=8, num_workers=4,
    generate_cfg=None,
):
    '''Run model inference across all misalignment conditions.

    Args:
        model: SLTModel (already on device, eval mode).
        dataset: EvalDataset instance (with set_condition method).
        conditions: List of (name, delta_s, delta_e) tuples.
        output_path: Path to save results JSON.
        tokenizer: HuggingFace mBART tokenizer.
        batch_size: Batch size for DataLoader.
        num_workers: Number of DataLoader workers.
        generate_cfg: Translation generation config.

    Returns:
        Dict with all results.
    '''
    from torch.utils.data import DataLoader
    import sys as _sys, os as _os
    _gfslt_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'gfslt-pose-trainer')
    if _gfslt_dir not in _sys.path:
        _sys.path.insert(0, _gfslt_dir)
    from loader import collate_fn

    model.eval()
    device = next(model.parameters()).device

    all_results = OrderedDict()
    all_results['meta'] = {
        'num_samples': len(dataset), 'num_conditions': len(conditions),
        'generate_cfg': generate_cfg,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for cond_name, delta_s, delta_e in tqdm(conditions):
        t_start = time.time()
        dataset.set_condition(cond_name, delta_s, delta_e)
        loader = DataLoader(
            dataset, batch_size=batch_size,
            num_workers=num_workers, collate_fn=collate_fn,
            shuffle=False, pin_memory=True, drop_last=False,
        )
        skipped_names = set()
        for idx in dataset._skipped_indices:
            skipped_names.add(dataset.list[idx])

        sample_results = OrderedDict()
        for batch in loader:
            pixel_values = batch['pixel_values'].to(device)
            pixel_mask = batch['pixel_mask'].to(device)
            names = batch['names']
            texts = batch['texts']

            collect_batch_predictions(
                model, names, texts, pixel_values, pixel_mask,
                sample_results, tokenizer, generate_cfg=generate_cfg,
            )

        # Compute metrics (excluding skipped samples)
        metrics = {}
        per_sample_metrics = {}
        eval_results = {k: v for k, v in sample_results.items()
                        if k not in skipped_names and 'txt_hyp' in v}
        if eval_results:
            metrics = compute_metrics(eval_results)
            per_sample_metrics = metrics.pop('per_sample', {})

        merged_predictions = OrderedDict()
        for n, v in sample_results.items():
            merged_entry = {
                'txt_hyp': v.get('txt_hyp', ''),
                'txt_ref': v.get('txt_ref', ''),
            }
            merged_entry.update(per_sample_metrics.get(n, {}))
            merged_predictions[n] = merged_entry

        elapsed = time.time() - t_start
        all_results[cond_name] = {
            'delta_s': delta_s, 'delta_e': delta_e,
            'num_evaluated': len(eval_results), 'num_skipped': len(skipped_names),
            'elapsed_seconds': round(elapsed, 1), 'metrics': metrics,
            'predictions': merged_predictions,
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"{cond_name}: BLEU-4={metrics.get('bleu4', 0):.2f}; "
              f"ROUGE-L={metrics.get('rouge_l', 0):.2f} "
              f"({elapsed:.1f}s, {len(eval_results)} samples)")

    # Group-level summaries
    group_summary = _compute_group_summaries(all_results)
    all_results['group_summary'] = group_summary

    print('\n=== Group-Level Summary ===')
    for gname, gdata in group_summary.items():
        print(f"  {gname:12s}: BLEU-4={gdata['bleu4']:.2f}, "
              f"ROUGE-L={gdata['rouge_l']:.2f} "
              f"({gdata['n_conditions']} conditions)")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results


def _compute_group_summaries(all_results):
    '''Compute mean metrics for each misalignment group.'''
    groups = defaultdict(lambda: defaultdict(list))
    for cond_name, cond_data in all_results.items():
        if cond_name in ('meta', 'clean', 'group_summary'): continue
        m = cond_data.get('metrics', {})
        bleu4 = m.get('bleu4')
        if bleu4 is None: continue
        rouge_l = m.get('rouge_l', 0)

        if '+' in cond_name:
            parts = cond_name.split('+')
            a_type = parts[0].rsplit('_', 1)[0]
            b_type = parts[1].rsplit('_', 1)[0]
            group = f'{a_type}+{b_type}'
            groups[group]['bleu4'].append(bleu4)
            groups[group]['rouge_l'].append(rouge_l)
            groups['All_Compound']['bleu4'].append(bleu4)
            groups['All_Compound']['rouge_l'].append(rouge_l)
        else:
            ctype = cond_name.rsplit('_', 1)[0]
            groups[ctype]['bleu4'].append(bleu4)
            groups[ctype]['rouge_l'].append(rouge_l)
            groups['All_Basic']['bleu4'].append(bleu4)
            groups['All_Basic']['rouge_l'].append(rouge_l)

        groups['Overall']['bleu4'].append(bleu4)
        groups['Overall']['rouge_l'].append(rouge_l)

    ordered_keys = ['HT', 'TT', 'HC', 'TC', 'HT+TT', 'HC+TC', 'HT+TC', 'HC+TT',
                    'All_Basic', 'All_Compound', 'Overall']
    summary = OrderedDict()
    for gname in ordered_keys:
        if gname in groups:
            g = groups[gname]
            summary[gname] = {
                'bleu4': float(np.mean(g['bleu4'])),
                'rouge_l': float(np.mean(g['rouge_l'])),
                'n_conditions': len(g['bleu4']),
            }
    return summary


def verify_clean_baseline(
    model, dataset, output_dir, tokenizer,
    batch_size=8, num_workers=4, generate_cfg=None, expected=None,
):
    '''Run clean evaluation and verify against expected numbers.'''
    output_path = os.path.join(output_dir, 'raw', 'verify_clean.json')
    conditions = [('clean', 0.0, 0.0)]
    results = run_evaluation(
        model, dataset, conditions, output_path,
        tokenizer=tokenizer, batch_size=batch_size,
        num_workers=num_workers, generate_cfg=generate_cfg,
    )
    metrics = results['clean']['metrics']

    print("=== Clean Baseline Verification ===")
    if expected:
        tol = expected.get('tolerance', 2.0)
        checks = []
        for metric_name in ['bleu4', 'rouge_l']:
            if metric_name in expected and metric_name in metrics:
                diff = abs(metrics[metric_name] - expected[metric_name])
                ok = diff <= tol
                status = 'OK' if ok else 'MISMATCH'
                print(f'{metric_name}: expected {expected[metric_name]:.2f}, '
                      f'got {metrics[metric_name]:.2f}, diff={diff:.2f} [{status}]')
                checks.append(ok)
        if all(checks): print('\nVerification PASSED - metrics match within tolerance.')
        else: print('WARNING: Some metrics differ from expected values.')
    return results
