"""Run the temporal misalignment benchmark on MSKA-SLT.

Evaluates the pretrained MSKA-SLT model on 41 conditions:
  1 clean  +  8 misalignment types  x  5 severity levels.

Usage:
    # Full benchmark (41 conditions)
    python run_benchmark.py --config configs/benchmark.yaml

    # Clean baseline only (verify reproduced numbers)
    python run_benchmark.py --config configs/benchmark.yaml --clean-only

    # Override device / batch size
    python run_benchmark.py --device cpu --batch-size 4
"""

import os
import sys
import json
import time
import copy
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Path setup – make MSKA modules importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MSKA_DIR = os.path.join(SCRIPT_DIR, 'MSKA')
sys.path.insert(0, MSKA_DIR)

# Disable wandb before any transitive import might pull it in
os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# MSKA imports
from model import SignLanguageModel
from datasets import S2T_Dataset
from Tokenizer import GlossTokenizer_S2G

# Our misalignment module (lives in project root alongside this script)
sys.path.insert(0, SCRIPT_DIR)
from misalign import (apply_misalignment, get_all_conditions,
                       CONDITION_TYPES, SEVERITY_LEVELS)


# ===================================================================
# Dataset wrapper — applies misalignment on-the-fly
# ===================================================================
class MisalignedDatasetWrapper(Dataset):
    """Thin wrapper around S2T_Dataset that applies temporal misalignment.

    Shares the base dataset's raw data (no data duplication) and delegates
    collation to the base dataset's collate_fn.
    """

    def __init__(self, base_dataset, delta_s_pct=0.0, delta_e_pct=0.0):
        self.base = base_dataset
        self.delta_s_pct = delta_s_pct
        self.delta_e_pct = delta_e_pct

    def __len__(self):
        return len(self.base)

    # helper: raw keypoint (C, T, V) for an index
    def _get_keypoint(self, index):
        key = self.base.list[index]
        sample = self.base.raw_data[key]
        return sample['keypoint'].permute(2, 0, 1).to(torch.float32)

    def __getitem__(self, index):
        key = self.base.list[index]
        sample = self.base.raw_data[key]

        keypoint = sample['keypoint'].permute(2, 0, 1).to(torch.float32)
        gloss = sample['gloss']
        text = sample['text'] if self.base.config['task'] != 'S2G' else None
        name_sample = sample['name']

        if self.delta_s_pct != 0 or self.delta_e_pct != 0:
            prev_kp = (self._get_keypoint(index - 1)
                       if self.delta_s_pct < 0 and index > 0 else None)
            next_kp = (self._get_keypoint(index + 1)
                       if self.delta_e_pct > 0 and index < len(self) - 1
                       else None)
            keypoint = apply_misalignment(
                keypoint, prev_kp, next_kp,
                self.delta_s_pct, self.delta_e_pct)

        length = keypoint.shape[1]  # updated temporal length
        return name_sample, keypoint, gloss, text, length


# ===================================================================
# Model loading
# ===================================================================
def load_config(benchmark_config_path):
    """Load the benchmark YAML and the MSKA YAML it references."""
    with open(benchmark_config_path, 'r', encoding='utf-8') as f:
        bench_cfg = yaml.load(f, Loader=yaml.FullLoader)

    mska_cfg_path = os.path.join(SCRIPT_DIR, bench_cfg['mska']['config'])
    with open(mska_cfg_path, 'r', encoding='utf-8') as f:
        mska_cfg = yaml.load(f, Loader=yaml.FullLoader)

    return bench_cfg, mska_cfg


def create_model(mska_cfg, checkpoint_path, device):
    """Instantiate MSKA-SLT, skip component-level pretrained loads,
    then load the full SLT checkpoint."""
    cfg = copy.deepcopy(mska_cfg)

    # Remove component-level pretrained paths — the full SLT checkpoint
    # will overwrite all weights anyway, so we avoid needing extra files.
    cfg['model']['RecognitionNetwork'].pop('pretrained_path', None)
    cfg['model']['TranslationNetwork'].pop('load_ckpt', None)
    cfg['device'] = device

    args = argparse.Namespace(device=device, distributed=False)

    # MSKA config paths are relative to MSKA dir
    prev_cwd = os.getcwd()
    os.chdir(MSKA_DIR)
    try:
        model = SignLanguageModel(cfg=cfg, args=args)
        model.to(device)
    finally:
        os.chdir(prev_cwd)

    # Load full SLT checkpoint
    ckpt_path = os.path.join(SCRIPT_DIR, checkpoint_path)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    epoch = checkpoint.get('epoch', '?')
    print(f"Loaded SLT checkpoint: {ckpt_path}  (epoch {epoch})")

    model.eval()
    return model, cfg


def create_base_dataset(mska_cfg):
    """Create the clean test-set S2T_Dataset (loaded once, shared by all
    MisalignedDatasetWrapper instances)."""
    args = argparse.Namespace(device='cuda', distributed=False)

    prev_cwd = os.getcwd()
    os.chdir(MSKA_DIR)
    try:
        tokenizer = GlossTokenizer_S2G(mska_cfg['gloss'])
        dataset = S2T_Dataset(
            path=mska_cfg['data']['test_label_path'],
            tokenizer=tokenizer,
            config=mska_cfg,
            args=args,
            phase='test',
            training_refurbish=True,
        )
    finally:
        os.chdir(prev_cwd)

    print(f"Test dataset: {len(dataset)} samples")
    return dataset


# ===================================================================
# Inference
# ===================================================================
def run_inference(model, dataloader, generate_cfg, device):
    """Run SLT inference, return OrderedDict {name: {hyp, ref}}."""
    model.eval()
    results = OrderedDict()

    with torch.no_grad():
        for src_input in dataloader:
            output = model(src_input)
            gen = model.generate_txt(
                transformer_inputs=output['transformer_inputs'],
                generate_cfg=generate_cfg)

            for name, hyp, ref in zip(
                    src_input['name'],
                    gen['decoded_sequences'],
                    src_input['text']):
                results[name] = {'hyp': hyp, 'ref': ref}

    return results


# ===================================================================
# Metrics
# ===================================================================
def compute_metrics(results, level='word'):
    """Compute BLEU-1..4, ROUGE-L, and METEOR (if available)."""
    from metrics import bleu, rouge

    refs = [results[n]['ref'] for n in results]
    hyps = [results[n]['hyp'] for n in results]

    bleu_dict = bleu(references=refs, hypotheses=hyps, level=level)
    rouge_score = rouge(references=refs, hypotheses=hyps, level=level)

    m = {k: round(v, 2) for k, v in bleu_dict.items()}
    m['rouge_l'] = round(rouge_score, 2)

    # METEOR via HuggingFace evaluate (optional)
    try:
        import evaluate as hf_evaluate
        meteor = hf_evaluate.load('meteor')
        score = meteor.compute(predictions=hyps, references=refs)
        m['meteor'] = round(score['meteor'] * 100, 2)
    except Exception as e:
        print(f"  [METEOR unavailable: {e}]")
        m['meteor'] = None

    return m


# ===================================================================
# Main benchmark loop
# ===================================================================
def run_benchmark(bench_cfg, mska_cfg, model, base_dataset, device):
    """Iterate over all 41 conditions, run inference, collect metrics."""
    generate_cfg = mska_cfg['testing']['translation']
    bs = bench_cfg['inference']['batch_size']
    nw = bench_cfg['inference']['num_workers']
    level = mska_cfg['data'].get('level', 'word')

    all_results = {
        'meta': {
            'dataset': mska_cfg['data']['dataset_name'],
            'num_test_samples': len(base_dataset),
            'severity_levels': SEVERITY_LEVELS,
            'conditions': list(CONDITION_TYPES.keys()),
        },
        'metrics': {},
        'translations': {},
    }

    conditions = get_all_conditions()
    total = len(conditions)

    for idx, (cond, sev, ds, de) in enumerate(conditions):
        label = 'clean' if cond == 'clean' else f"{cond}_{sev}"
        print(f"\n[{idx+1}/{total}] {label}  (ds={ds:+.2f}  de={de:+.2f})")
        t0 = time.time()

        wrapped = MisalignedDatasetWrapper(base_dataset, ds, de)
        loader = DataLoader(
            wrapped, batch_size=bs, num_workers=nw,
            collate_fn=base_dataset.collate_fn,
            pin_memory=True, shuffle=False)

        results = run_inference(model, loader, generate_cfg, device)
        metrics = compute_metrics(results, level)

        dt = time.time() - t0
        print(f"  BLEU-4={metrics['bleu4']:.2f}  ROUGE-L={metrics['rouge_l']:.2f}  "
              f"METEOR={metrics.get('meteor', 'N/A')}  ({dt:.1f}s)")

        # store metrics
        if cond == 'clean':
            all_results['metrics']['clean'] = metrics
        else:
            all_results['metrics'].setdefault(cond, {})[str(sev)] = metrics

        # store per-sample translations
        all_results['translations'][label] = [
            {'name': n, 'ref': r['ref'], 'hyp': r['hyp']}
            for n, r in results.items()
        ]

    return all_results


def check_sample_ordering(dataset):
    """Print a few sample names so the user can verify sequential ordering."""
    names = dataset.list[:10]
    print("\nSample ordering check (first 10 names):")
    for i, n in enumerate(names):
        print(f"  [{i}] {n}")
    print(f"  ... ({len(dataset)} total)")
    print("If names share a video-session prefix and are sequential,")
    print("contamination frames come from the same continuous stream.\n")


def print_summary(all_results):
    """Print a compact summary table using pandas."""
    import pandas as pd

    clean = all_results['metrics']['clean']['bleu4']
    rows = []
    for cond in CONDITION_TYPES:
        if cond == 'clean':
            continue
        scores = all_results['metrics'].get(cond, {})
        row = {'condition': cond}
        for s in SEVERITY_LEVELS:
            b = scores.get(str(s), {}).get('bleu4')
            if b is not None:
                drop = (clean - b) / clean * 100 if clean > 0 else 0
                row[f'{s}%'] = f"{b:.1f} ({drop:+.0f}%)"
            else:
                row[f'{s}%'] = 'N/A'
        rows.append(row)

    df = pd.DataFrame(rows).set_index('condition')
    print(f"\nClean baseline BLEU-4: {clean:.2f}")
    print(df.to_string())


# ===================================================================
# Entry point
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Temporal misalignment benchmark for MSKA-SLT')
    parser.add_argument('--config', default='configs/benchmark.yaml')
    parser.add_argument('--device', default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--clean-only', action='store_true',
                        help='Run only the clean baseline for verification')
    args = parser.parse_args()

    # Load configs
    cfg_path = os.path.join(SCRIPT_DIR, args.config)
    bench_cfg, mska_cfg = load_config(cfg_path)

    device = args.device or bench_cfg['inference']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA unavailable — falling back to CPU")
        device = 'cpu'
    mska_cfg['device'] = device

    if args.batch_size:
        bench_cfg['inference']['batch_size'] = args.batch_size

    results_dir = os.path.join(SCRIPT_DIR, bench_cfg['results_dir'])
    os.makedirs(results_dir, exist_ok=True)

    print(f"Device : {device}")
    print(f"Batch  : {bench_cfg['inference']['batch_size']}")

    # ---- model -----------------------------------------------------------
    print("\n--- Loading MSKA-SLT model ---")
    model, mska_cfg = create_model(
        mska_cfg, bench_cfg['mska']['checkpoint'], device)

    # ---- dataset ---------------------------------------------------------
    print("\n--- Loading test dataset ---")
    base_dataset = create_base_dataset(mska_cfg)
    check_sample_ordering(base_dataset)

    # ---- run -------------------------------------------------------------
    if args.clean_only:
        print("--- Running clean baseline only ---")
        generate_cfg = mska_cfg['testing']['translation']
        bs = bench_cfg['inference']['batch_size']
        nw = bench_cfg['inference']['num_workers']
        level = mska_cfg['data'].get('level', 'word')

        loader = DataLoader(
            base_dataset, batch_size=bs, num_workers=nw,
            collate_fn=base_dataset.collate_fn,
            pin_memory=True, shuffle=False)

        results = run_inference(model, loader, generate_cfg, device)
        metrics = compute_metrics(results, level)

        print(f"\nClean baseline results:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print("\nExpected (MSKA paper): BLEU-4 ~ 29.03, ROUGE ~ 53.54")

        all_results = {
            'meta': {
                'dataset': mska_cfg['data']['dataset_name'],
                'num_test_samples': len(base_dataset),
            },
            'metrics': {'clean': metrics},
            'translations': {
                'clean': [{'name': n, 'ref': r['ref'], 'hyp': r['hyp']}
                          for n, r in results.items()]
            },
        }
    else:
        print("--- Running full benchmark (41 conditions) ---")
        all_results = run_benchmark(
            bench_cfg, mska_cfg, model, base_dataset, device)
        print_summary(all_results)

    # ---- save ------------------------------------------------------------
    out_path = os.path.join(results_dir, 'benchmark_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
