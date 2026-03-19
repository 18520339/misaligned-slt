'''Run the temporal misalignment benchmark on MSKA-SLT.

Evaluates the pretrained MSKA-SLT model on 121 conditions:
  1 clean
  + 4 single-sided types × 5 severity levels            =  20
  + 4 compound types × 5 head severities × 5 tail sevs  = 100

# Full benchmark (121 conditions)
python run_benchmark.py --config configs/benchmark.yaml

# Clean baseline only (verify reproduced numbers)
python run_benchmark.py --config configs/benchmark.yaml --clean-only

# Override device / batch size
python run_benchmark.py --device cpu --batch-size 4
'''
import os
import sys
import json
import time
import yaml
import random
import argparse

import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset

import evaluate as hf_evaluate
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import logging

logging.set_verbosity_error()
meteor = hf_evaluate.load('meteor')
ter = hf_evaluate.load('ter')

# Path setup – make MSKA modules importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MSKA_DIR = os.path.join(SCRIPT_DIR, 'MSKA')
sys.path.insert(0, MSKA_DIR)

'''MSKA imports. Note that:
- Rename datasets.py to mska_datasets.py to avoid confusion with HF's Datasets library inside evaluate library
- Remove sacrebleu.py since we use HF's sacrebleu, which also contains TER
- Change bleu function at line 2894 of metrics.py to:
def bleu(references, hypotheses, level='word'):
    if level == 'char':
        references = [' '.join(list(r)) for r in references]
        hypotheses = [' '.join(list(r)) for r in hypotheses]
    bleu4 = sacrebleu.raw_corpus_bleu(hypotheses, [references]).score
    return {'bleu4': bleu4}
'''
from mska_datasets import S2T_Dataset 
from Tokenizer import GlossTokenizer_S2G
from model import SignLanguageModel
from metrics import bleu, rouge
from utils import MetricLogger

# Our misalignment module (lives in project root alongside this script)
sys.path.insert(0, SCRIPT_DIR)
from misalign import (
    compute_frame_counts, get_all_conditions, apply_misalignment, 
    CONDITION_TYPES, SEVERITY_LEVELS, SINGLE_CONDITIONS, COMPOUND_CONDITIONS
)
class MisalignedDatasetWrapper(Dataset): # Dataset wrapper — applies misalignment on-the-fly
    '''Thin wrapper around S2T_Dataset that applies temporal misalignment.

    Shares the base dataset's raw data (no data duplication) and delegates collation to the base dataset's collate_fn.
    '''
    def __init__(self, base_dataset: S2T_Dataset, delta_s_pct=0.0, delta_e_pct=0.0):
        self.base = base_dataset
        self.delta_s_pct = delta_s_pct
        self.delta_e_pct = delta_e_pct

    def __len__(self):
        return len(self.base)

    def _get_keypoint(self, index): # helper: raw keypoint (C, T, V) for an index
        key = self.base.list[index]
        sample = self.base.raw_data[key]
        return sample['keypoint'].permute(2, 0, 1).to(torch.float32)

    def __getitem__(self, index):
        key = self.base.list[index]
        sample = self.base.raw_data[key]
        keypoint = sample['keypoint'].permute(2, 0, 1).to(torch.float32)

        gloss = sample['gloss']
        text = sample['text'] if self.base.config['task'] != 'S2G' else None

        if self.delta_s_pct != 0 or self.delta_e_pct != 0:
            prev_kp = self._get_keypoint(index - 1) if self.delta_s_pct < 0 and index > 0 else None
            next_kp = self._get_keypoint(index + 1) if self.delta_e_pct > 0 and index < len(self) - 1 else None
            keypoint = apply_misalignment(keypoint, prev_kp, next_kp, self.delta_s_pct, self.delta_e_pct)

        length = keypoint.shape[1]  # updated temporal length
        return sample['name'], keypoint, gloss, text, length


def load_config(benchmark_config_path): # Load the benchmark YAML and the MSKA YAML it references
    with open(benchmark_config_path, 'r', encoding='utf-8') as f:
        bench_cfg = yaml.load(f, Loader=yaml.FullLoader)

    mska_cfg_path = os.path.join(SCRIPT_DIR, bench_cfg['mska']['config'])
    with open(mska_cfg_path, 'r', encoding='utf-8') as f:
        mska_cfg = yaml.load(f, Loader=yaml.FullLoader)
    return bench_cfg, mska_cfg


def create_model(mska_cfg, checkpoint_path, device):
    # Remove component-level pretrained paths — the full MSKA-SLT checkpoint
    # will overwrite all weights anyway, so we avoid needing extra files.
    cfg = deepcopy(mska_cfg)
    cfg['model']['RecognitionNetwork'].pop('pretrained_path', None)
    cfg['model']['TranslationNetwork'].pop('load_ckpt', None)
    cfg['device'] = device

    # MSKA config paths are relative to MSKA dir
    prev_cwd = os.getcwd()
    os.chdir(MSKA_DIR)
    try:
        args = argparse.Namespace(device=device, distributed=False)
        model = SignLanguageModel(cfg=cfg, args=args)
        model.to(device)
    finally: os.chdir(prev_cwd)

    # Load full SLT checkpoint
    ckpt_path = os.path.join(SCRIPT_DIR, checkpoint_path)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    epoch = checkpoint.get('epoch', '?')
    
    print(f'Loaded SLT checkpoint: {ckpt_path}  (epoch {epoch})')
    model.eval()
    return model, cfg


def create_base_dataset(mska_cfg):
    # Create the clean test-set S2T_Dataset (loaded once, shared by all MisalignedDatasetWrapper instances)
    prev_cwd = os.getcwd()
    os.chdir(MSKA_DIR)
    try:
        args = argparse.Namespace(device='cuda', distributed=False)
        tokenizer = GlossTokenizer_S2G(mska_cfg['gloss'])
        dataset = S2T_Dataset(
            path=mska_cfg['data']['test_label_path'], tokenizer=tokenizer, 
            config=mska_cfg, args=args, phase='test', training_refurbish=True,
        )
    finally: os.chdir(prev_cwd)
    print(f'Test dataset: {len(dataset)} samples')
    return dataset


def run_inference(model, dataloader, generate_cfg, device): # Run SLT inference, return OrderedDict {name: {hyp, ref}}
    model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    results = OrderedDict()

    with torch.no_grad():
        for src_input in metric_logger.log_every(dataloader, 10, 'Test:'):
            output = model(src_input)
            gen = model.generate_txt(transformer_inputs=output['transformer_inputs'], generate_cfg=generate_cfg)
            
            for name, hyp, ref in zip(src_input['name'], gen['decoded_sequences'], src_input['text']):
                results[name] = {'hyp': hyp, 'ref': ref}
    return results


def _sentence_rouge_l(ref_toks, hyp_toks):
    '''Compute LCS-based ROUGE-L F1 * 100 for a single (ref, hyp) pair.

    Args:
        ref_toks: list of str tokens (reference).
        hyp_toks: list of str tokens (hypothesis).

    Returns:
        float, ROUGE-L F1 score scaled to [0, 100].
    '''
    if not ref_toks or not hyp_toks: return 0.0
    m, n = len(ref_toks), len(hyp_toks)
    
    # LCS via dynamic programming
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_toks[i - 1] == hyp_toks[j - 1]: dp[i][j] = dp[i - 1][j - 1] + 1
            else: dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            
    lcs = dp[m][n]
    if lcs == 0: return 0.0
    prec, rec = lcs / n, lcs / m
    f1 = 2 * prec * rec / (prec + rec)
    return f1 * 100


def compute_per_sample_metrics(results):
    '''Compute sentence-level BLEU and ROUGE-L for each sample.

    Args:
        results: OrderedDict {name: {hyp, ref}}.

    Returns:
        dict mapping name -> {sent_bleu, sent_rouge_l}.
    '''
    smooth = SmoothingFunction().method1
    per_sample = {}
    for name in results:
        ref = results[name]['ref']
        hyp = results[name]['hyp']
        ref_toks = ref.split()
        hyp_toks = hyp.split()
        
        try: sb = sentence_bleu([ref_toks], hyp_toks, smoothing_function=smooth) * 100
        except Exception: sb = 0.0
        sr = _sentence_rouge_l(ref_toks, hyp_toks)
        per_sample[name] = {'sent_bleu': round(sb, 2), 'sent_rouge_l': round(sr, 2)}
    return per_sample


def compute_metrics(results, level='word', skip_bertscore=False): # Compute BLEU-1..4, ROUGE-L, METEOR, TER, and BERTScore
    refs = [results[n]['ref'] for n in results]
    hyps = [results[n]['hyp'] for n in results]

    bleu_dict = bleu(references=refs, hypotheses=hyps, level=level)
    rouge_score = rouge(references=refs, hypotheses=hyps, level=level)
    meteor_score = meteor.compute(predictions=hyps, references=refs)['meteor']
    ter_score = ter.compute(predictions=hyps, references=refs)['score']
    
    metrics = {k: round(v, 2) for k, v in bleu_dict.items()}
    metrics['rouge_l'] = round(rouge_score, 2)
    metrics['meteor'] = round(meteor_score * 100, 2)
    metrics['ter'] = round(ter_score, 2)
        
    if not skip_bertscore:
        P, R, F1 = bert_score_fn(
            hyps, refs, lang='de', device='cuda' if torch.cuda.is_available() else 'cpu', 
            rescale_with_baseline=True, use_fast_tokenizer=True, verbose=False
        )
        metrics['bertscore_f1'] = round(F1.mean().item() * 100, 2)
    else: metrics['bertscore_f1'] = None
    return metrics


def _run_single_condition(
    ds, de, base_dataset, model,  generate_cfg, 
    batch_size, num_workers, level, device, _name_to_idx, skip_bertscore):
    '''Run inference for one (ds, de) condition.

    Returns:
        (metrics_dict, enhanced_translations_list)
    '''
    t0 = time.time()
    wrapped = MisalignedDatasetWrapper(base_dataset, ds, de)
    loader = DataLoader(
        wrapped, batch_size=batch_size, num_workers=num_workers,
        collate_fn=base_dataset.collate_fn, pin_memory=True, shuffle=False
    )
    results = run_inference(model, loader, generate_cfg, device)
    metrics = compute_metrics(results, level, skip_bertscore=skip_bertscore)
    per_sample_metrics = compute_per_sample_metrics(results)

    dt = time.time() - t0
    for k in ['bleu4', 'rouge_l', 'meteor', 'ter', 'bertscore_f1']:
        v = metrics.get(k, 'N/A')
        if v is not None: print(f'  {k.upper()}: {v:.2f}', end='  ')
    print(f'({dt:.1f}s)')
    
    enhanced_translations = []
    for n, r in results.items():
        key_idx = _name_to_idx.get(n)
        frame_meta = {}
        cross_session = False
        
        if key_idx is not None:
            key = base_dataset.list[key_idx]
            T_original = base_dataset.raw_data[key]['keypoint'].shape[0]
            frame_meta = compute_frame_counts(T_original, ds, de)

        psm = per_sample_metrics.get(n, {})
        enhanced_translations.append({
            'name': n, 'ref': r['ref'], 'hyp': r['hyp'],
            'sent_bleu': psm.get('sent_bleu'),
            'sent_rouge_l': psm.get('sent_rouge_l'),
            'T_original': frame_meta.get('T_original'),
            'T_after': frame_meta.get('T_after'),
            'head_trunc_frames': frame_meta.get('head_trunc_frames'),
            'tail_trunc_frames': frame_meta.get('tail_trunc_frames'),
            'head_contam_frames': frame_meta.get('head_contam_frames'),
            'tail_contam_frames': frame_meta.get('tail_contam_frames'),
        })
    return metrics, enhanced_translations


def run_benchmark(bench_cfg, mska_cfg, model, base_dataset, device, skip_bertscore=False): 
    # Iterate over all 121 conditions, run inference, collect metrics
    generate_cfg = mska_cfg['testing']['translation']
    batch_size = bench_cfg['inference']['batch_size']
    num_workers = bench_cfg['inference']['num_workers']
    level = mska_cfg['data'].get('level', 'word')

    all_results = {'meta': {
        'dataset': mska_cfg['data']['dataset_name'],
        'num_test_samples': len(base_dataset),
        'severity_levels': SEVERITY_LEVELS,
        'conditions': list(CONDITION_TYPES.keys()),
    }, 'metrics': {}, 'translations': {}}
    
    # Precompute name → index mapping for efficient lookup
    _name_to_idx = {base_dataset.raw_data[k]['name']: i for i, k in enumerate(base_dataset.list)}
    conditions = get_all_conditions()
    total = len(conditions)
    
    for idx, (label, cond_name, head_sev, tail_sev, ds, de) in enumerate(conditions):
        print(f'\n[{idx+1}/{total}] {label}  (δs={ds:+.2f}  δe={de:+.2f})')
        cond_metrics, enhanced = _run_single_condition(
            ds, de, base_dataset, model, generate_cfg,
            batch_size, num_workers, level, device, _name_to_idx, skip_bertscore
        )
        if cond_name == 'clean': all_results['metrics']['clean'] = cond_metrics
        elif cond_name in COMPOUND_CONDITIONS:
            all_results['metrics'].setdefault(cond_name, {})[f'h{head_sev}_t{tail_sev}'] = cond_metrics
        else:
            sev = head_sev or tail_sev
            all_results['metrics'].setdefault(cond_name, {})[str(sev)] = cond_metrics
        all_results['translations'][label] = enhanced
    return all_results


def check_sample_ordering(dataset): # Print a few sample names to verify sequential ordering
    names = dataset.list[:10]
    print('\nSample ordering check (first 10 names):')
    for i, n in enumerate(names): print(f'  [{i}] {n}')
    print(f'  ... ({len(dataset)} total)')
    print('If names share a video-session prefix and are sequential,')
    print('contamination frames come from the same continuous stream.\n')


def print_summary(all_results): # Print a compact summary table after the full run
    clean = all_results['metrics']['clean']
    clean_bleu4 = clean['bleu4']

    # Check if TER / BERTScore are available
    has_ter = clean.get('ter') is not None
    has_bertscore = clean.get('bertscore_f1') is not None
    
    rows = []
    for cond in CONDITION_TYPES:
        if cond == 'clean': continue
        scores = all_results['metrics'].get(cond, {})
        row = {'condition': cond}
        
        is_compound = cond in COMPOUND_CONDITIONS
        for s in SEVERITY_LEVELS:
            mkey = f'h{s}_t{s}' if is_compound else str(s)
            sev_data = scores.get(mkey, {})
            b = sev_data.get('bleu4')
            
            if b is not None:
                drop = (clean_bleu4 - b) / clean_bleu4 * 100 if clean_bleu4 > 0 else 0
                cell = f'{b:.1f} (-{drop:.0f}%)'
                if has_ter:
                    ter_val = sev_data.get('ter')
                    cell += f'  TER={ter_val}' if ter_val is not None else ''
                if has_bertscore:
                    bs_val = sev_data.get('bertscore_f1')
                    cell += f'  BS={bs_val}' if bs_val is not None else ''
                row[f'{s}%'] = cell
            else:
                row[f'{s}%'] = 'N/A'
        rows.append(row)

    df = pd.DataFrame(rows).set_index('condition')
    print(f'\nClean baseline BLEU-4: {clean_bleu4:.2f}', end='')
    if has_ter: print(f'  TER: {clean.get('ter')}', end='')
    if has_bertscore: print(f'  BERTScore-F1: {clean.get('bertscore_f1')}', end='')
    print()
    print(df.to_string())


def main():
    parser = argparse.ArgumentParser(description='Temporal misalignment benchmark for MSKA-SLT')
    parser.add_argument('--config', default='configs/benchmark.yaml')
    parser.add_argument('--device', default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--clean-only', action='store_true', help='Run only the clean baseline for verification')
    parser.add_argument('--skip-bertscore', action='store_true', help='Skip BERTScore computation for speed')
    args = parser.parse_args()
    
    # Seed setting for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    # Load configs
    cfg_path = os.path.join(SCRIPT_DIR, args.config)
    bench_cfg, mska_cfg = load_config(cfg_path)

    device = args.device or bench_cfg['inference']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA unavailable — falling back to CPU')
        device = 'cpu'
    mska_cfg['device'] = device

    if args.batch_size: bench_cfg['inference']['batch_size'] = args.batch_size
    results_dir = os.path.join(SCRIPT_DIR, bench_cfg['results_dir'])
    os.makedirs(results_dir, exist_ok=True)
    print(f"Device: {device}, Batch Size: {bench_cfg['inference']['batch_size']}")

    # ---- model -----------------------------------------------------------
    print('\n--- Loading MSKA-SLT model ---')
    model, mska_cfg = create_model(mska_cfg, bench_cfg['mska']['checkpoint'], device)

    # ---- dataset ---------------------------------------------------------
    print('\n--- Loading test dataset ---')
    base_dataset = create_base_dataset(mska_cfg)
    check_sample_ordering(base_dataset)

    # ---- run -------------------------------------------------------------
    if args.clean_only:
        print('--- Running clean baseline only ---')
        generate_cfg = mska_cfg['testing']['translation']
        batch_size = bench_cfg['inference']['batch_size']
        num_workers = bench_cfg['inference']['num_workers']
        level = mska_cfg['data'].get('level', 'word')

        loader = DataLoader(
            base_dataset, batch_size=batch_size, num_workers=num_workers,
            collate_fn=base_dataset.collate_fn, pin_memory=True, shuffle=False
        )
        results = run_inference(model, loader, generate_cfg, device)
        metrics = compute_metrics(results, level, skip_bertscore=args.skip_bertscore)

        print(f'\nClean baseline results:')
        for k, v in metrics.items(): print(f'  {k}: {v}')
        print('\nExpected (MSKA paper): BLEU-4 ~ 29.03, ROUGE ~ 53.54')

        all_results = {
            'meta': {
                'dataset': mska_cfg['data']['dataset_name'],
                'num_test_samples': len(base_dataset),
                'seeds': {'random': SEED, 'numpy': SEED, 'torch': SEED},
            },
            'metrics': {'clean': metrics},
            'translations': {'clean': [
                {'name': n, 'ref': r['ref'], 'hyp': r['hyp']} 
                for n, r in results.items()
            ]},
        }
    else:
        print('--- Running full benchmark (121 conditions) ---')
        all_results = run_benchmark(
            bench_cfg, mska_cfg, model, base_dataset, 
            device, skip_bertscore=args.skip_bertscore
        )
        all_results['meta']['seeds'] = {'random': SEED, 'numpy': SEED, 'torch': SEED}
        print_summary(all_results)

    # ---- save ------------------------------------------------------------
    out_path = os.path.join(results_dir, 'benchmark_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()