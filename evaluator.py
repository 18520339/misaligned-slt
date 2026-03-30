'''
Runs MSKA inference under various misalignment conditions and collects all results
(text predictions, gloss predictions, CTC logits, metrics) into a structured JSON file.
'''
import os, sys, json, time
from collections import defaultdict, OrderedDict
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

# MSKA imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'MSKA'))

from data.misalign import generate_conditions, parse_condition_name
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from metrics import wer_single, wer_list, bleu, rouge
from phoenix_cleanup import clean_phoenix_2014_trans
from utils import MetricLogger


def _select_primary_gls_hyp(sample_row: dict) -> str:
    # Select the canonical gloss hypothesis used in exported predictions.'''
    if sample_row.get('ensemble_last_gls_hyp'): return sample_row['ensemble_last_gls_hyp']
    if sample_row.get('fuse_gls_hyp'): return sample_row['fuse_gls_hyp']

    # Fallback for other recognition-head outputs.
    gls_keys = sorted(k for k in sample_row if 'gls_hyp' in k and sample_row.get(k))
    if gls_keys: return sample_row.get(gls_keys[0], '')
    return ''


def compute_metrics(results: dict, config: dict) -> dict:
    '''Compute corpus-level and per-sample metrics from collected predictions.

    Replicates MSKA's evaluate() exactly for WER: finds all keys that contain
    'gls_hyp' (one per recognition head), computes WER for each, and takes the
    minimum — matching the MSKA paper's reported numbers.
    '''
    names = list(results.keys())
    txt_ref = [results[n]['txt_ref'] for n in names]
    txt_hyp = [results[n]['txt_hyp'] for n in names]

    level = config['data'].get('level', 'word')
    dataset_name = config['data']['dataset_name'].lower()
    metrics = {} # Corpus-level metrics
    
    # WER: replicate MSKA's approach — iterate over every *_gls_hyp key,
    # compute WER per head, keep the minimum (= best head).
    # Collect all gloss-hypothesis key names present in ANY sample
    all_gls_hyp_keys = set()
    for n in names:
        for k in results[n]:
            if 'gls_hyp' in k:
                all_gls_hyp_keys.add(k)

    if all_gls_hyp_keys:
        best_wer = 200.0
        best_wer_details = {}
        per_head_wer = {}

        # Reference is the same regardless of head
        gls_ref_raw = [results[n].get('gls_ref', '') for n in names]
        if dataset_name == 'phoenix-2014t':
            gls_ref_clean = [clean_phoenix_2014_trans(g) for g in gls_ref_raw]
        else:
            gls_ref_clean = gls_ref_raw

        for hyp_key in sorted(all_gls_hyp_keys):
            gls_hyp_raw = [results[n].get(hyp_key, '') for n in names]
            if dataset_name == 'phoenix-2014t':
                gls_hyp_clean = [clean_phoenix_2014_trans(g) for g in gls_hyp_raw]
            else:
                gls_hyp_clean = gls_hyp_raw

            wer_res = wer_list(references=gls_ref_clean, hypotheses=gls_hyp_clean)
            head_name = hyp_key.replace('gls_hyp', '').strip('_') or 'default'
            per_head_wer[head_name] = wer_res

            if wer_res['wer'] < best_wer:
                best_wer = wer_res['wer']
                best_wer_details = wer_res

        metrics['wer'] = best_wer
        metrics['del_rate'] = best_wer_details.get('del_rate', 0)
        metrics['ins_rate'] = best_wer_details.get('ins_rate', 0)
        metrics['sub_rate'] = best_wer_details.get('sub_rate', 0)
        metrics['per_head_wer'] = {k: v['wer'] for k, v in per_head_wer.items()}
        metrics['best_wer_head'] = min(per_head_wer, key=lambda k: per_head_wer[k]['wer'])

    # BLEU
    bleu_dict = bleu(references=txt_ref, hypotheses=txt_hyp, level=level)
    for k, v in bleu_dict.items(): metrics[k] = v

    # ROUGE
    rouge_score = rouge(references=txt_ref, hypotheses=txt_hyp, level=level)
    metrics['rouge_l'] = rouge_score

    # Per-sample metrics
    per_sample = {}
    smooth = SmoothingFunction().method1
    for n in names:
        hyp, ref = results[n]['txt_hyp'], results[n]['txt_ref']
        hyp_tokens = hyp.split() if hyp else []
        ref_tokens = ref.split() if ref else ['']

        s_bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth) * 100
        gls_hyp = _select_primary_gls_hyp(results[n])
        gls_ref = results[n].get('gls_ref', '')
        if dataset_name == 'phoenix-2014t':
            gls_hyp = clean_phoenix_2014_trans(gls_hyp)
            gls_ref = clean_phoenix_2014_trans(gls_ref)

        # sentence_wer is gloss-level WER to match gls_hyp/gls_ref inspection.
        if gls_ref or gls_hyp:
            sent_wer_res = wer_single(r=gls_ref, h=gls_hyp)
        else:
            sent_wer_res = wer_single(r=ref, h=hyp)
        sent_wer = (sent_wer_res['num_err'] / max(sent_wer_res['num_ref'], 1)) * 100
        ref_set = set(ref_tokens)
        novel_tokens = [t for t in hyp_tokens if t not in ref_set]
        novel_rate = len(novel_tokens) / max(len(hyp_tokens), 1)
        len_ratio = len(hyp_tokens) / max(len(ref_tokens), 1)

        # Check for repetition (any token repeated >= 3 times consecutively)
        has_repetition = False
        if len(hyp_tokens) >= 3:
            for i in range(len(hyp_tokens) - 2):
                if hyp_tokens[i] == hyp_tokens[i + 1] == hyp_tokens[i + 2]:
                    has_repetition = True
                    break

        per_sample[n] = {
            'sentence_bleu': s_bleu, 'sentence_wer': sent_wer,
            'novel_token_rate': novel_rate, 'output_length_ratio': len_ratio,
            'has_repetition': has_repetition, 'hyp_length': len(hyp_tokens), 'ref_length': len(ref_tokens),
        }
    metrics['per_sample'] = per_sample
    return metrics


@torch.no_grad()
def run_evaluation(
    model, dataset, conditions: list, config: dict, output_path: str,
    batch_size: int = 8, num_workers: int = 4, beam_size: int = 5,
    generate_cfg: dict = None, collect_logits: bool = False,
):
    '''Run model inference across all misalignment conditions.

    Args:
        model: Loaded MSKA SignLanguageModel (already on device, eval mode).
        dataset: MisalignedDataset instance.
        conditions: List of (name, delta_s, delta_e) tuples.
        config: MSKA config dict.
        output_path: Path to save results JSON.
        batch_size: Batch size for DataLoader.
        num_workers: Number of DataLoader workers.
        beam_size: CTC beam search width.
        generate_cfg: Translation generation config.
        collect_logits: Whether to store CTC log-probabilities.

    Returns:
        Dict with all results.
    '''
    from torch.utils.data import DataLoader
    from Tokenizer import GlossTokenizer_S2G

    if generate_cfg is None: generate_cfg = config.get(
        'testing', {}).get('translation', {
        'length_penalty': 1, 'max_length': 100, 'num_beams': 5
    })

    model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    tokenizer = model.gloss_tokenizer
    do_recognition = config.get('do_recognition', True)
    do_translation = config.get('do_translation', True)

    all_results = OrderedDict()
    all_results['meta'] = {
        'num_samples': len(dataset), 'num_conditions': len(conditions),
        'beam_size': beam_size, 'generate_cfg': generate_cfg,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for cond_name, delta_s, delta_e in tqdm(conditions):
        t_start = time.time()
        dataset.set_condition(cond_name, delta_s, delta_e) # Set misalignment condition on dataset
        loader = DataLoader(
            dataset, batch_size=batch_size,
            num_workers=num_workers, collate_fn=dataset.collate_fn,
            shuffle=False, pin_memory=True, drop_last=False,
        )
        skipped_names = set() # Get the set of skipped sample names
        for idx in dataset._skipped_indices:
            skipped_names.add(dataset.list[idx])

        sample_results = OrderedDict()
        for src_input in metric_logger.log_every(loader, 1, f'{cond_name}'):
            output = model(src_input)
            batch_names = src_input['name']

            # --- Recognition (CTC gloss decoding) ---
            if do_recognition:
                for k in output: # Use ensemble logits for best decoding
                    if 'gloss_logits' not in k: continue
                    ctc_out = model.recognition_network.decode(
                        gloss_logits=output[k], beam_size=beam_size,
                        input_lengths=output['input_lengths']
                    )
                    pred_glosses = tokenizer.convert_ids_to_tokens(ctc_out)
                    for name, gls_hyp, gls_ref in zip(batch_names, pred_glosses, src_input['gloss']):
                        if name not in sample_results: sample_results[name] = {}
                        logits_prefix = k.replace('gloss_logits', '')
                        sample_results[name][f'{logits_prefix}gls_hyp'] = (
                            ' '.join(gls_hyp).upper() 
                            if tokenizer.lower_case else ' '.join(gls_hyp)
                        )
                        sample_results[name]['gls_ref'] = gls_ref.upper() if tokenizer.lower_case else gls_ref

                # Collect CTC logits for confidence analysis
                if collect_logits and 'ensemble_last_gloss_logits' in output:
                    logits = output['ensemble_last_gloss_logits']
                    probs = logits.softmax(dim=-1)
                    max_conf = probs.max(dim=-1).values  # (B, T)
                    for i, name in enumerate(batch_names):
                        valid_len = output['input_lengths'][i].item()
                        mean_conf = max_conf[i, :valid_len].mean().item()
                        sample_results[name]['mean_ctc_confidence'] = mean_conf

            # --- Translation ---
            if do_translation:
                gen_output = model.generate_txt(transformer_inputs=output['transformer_inputs'], generate_cfg=generate_cfg)
                for name, txt_hyp, txt_ref in zip(batch_names, gen_output['decoded_sequences'], src_input['text']):
                    if name not in sample_results: sample_results[name] = {}
                    sample_results[name]['txt_hyp'] = txt_hyp
                    sample_results[name]['txt_ref'] = txt_ref

        # --- Compute metrics for this condition ---
        # Filter out skipped samples
        metrics = {}
        per_sample_metrics = {}
        eval_results = {k: v for k, v in sample_results.items() if k not in skipped_names and 'txt_hyp' in v}
        if eval_results:
            metrics = compute_metrics(eval_results, config)
            per_sample_metrics = metrics.pop('per_sample', {})

        merged_predictions = OrderedDict()
        for n, v in sample_results.items():
            merged_entry = {
                'txt_hyp': v.get('txt_hyp', ''), 'txt_ref': v.get('txt_ref', ''),
                'gls_hyp': _select_primary_gls_hyp(v), 'gls_ref': v.get('gls_ref', ''),
                'mean_ctc_confidence': v.get('mean_ctc_confidence', None),
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
        with open(output_path, 'w', encoding='utf-8') as f: # Save incrementally
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"{cond_name}: WER={metrics.get('wer', 0):.2f}; BLEU-4={metrics.get('bleu4', 0):.2f}; "
              f"ROUGE-L={metrics.get('rouge_l', 0):.2f} ({elapsed:.1f}s, {len(eval_results)} samples)\n")
    return all_results


def verify_clean_baseline(
    model, dataset, config, output_dir, batch_size=8, 
    num_workers=4, beam_size=5, generate_cfg=None, expected=None
): # Run clean evaluation and verify against MSKA's reported numbers
    output_path = os.path.join(output_dir, 'raw', 'verify_clean.json')
    conditions = [('clean', 0.0, 0.0)]
    results = run_evaluation(
        model, dataset, conditions, config, output_path,
        batch_size=batch_size, num_workers=num_workers, beam_size=beam_size, 
        generate_cfg=generate_cfg, collect_logits=True
    )
    metrics = results['clean']['metrics']
    
    print("=== Clean Baseline Verification ===")
    if expected:
        tol = expected.get('tolerance', 2.0)
        checks = []
        for metric_name in ['wer', 'bleu4', 'rouge_l']:
            if metric_name in expected and metric_name in metrics:
                diff = abs(metrics[metric_name] - expected[metric_name])
                ok = diff <= tol
                status = 'OK' if ok else 'MISMATCH'
                print(f'{metric_name}: expected {expected[metric_name]:.2f}, '
                      f'got {metrics[metric_name]:.2f}, diff={diff:.2f} [{status}]')
                checks.append(ok)
        if all(checks): print('\nVerification PASSED - metrics match MSKA paper within tolerance.')
        else: print('WARNING: Some metrics differ from expected values.')
    return results