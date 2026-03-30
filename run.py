'''Entry point for temporal misalignment analysis of sign language translation.

Usage:
    python run.py --mode verify        # Verify MSKA works, log clean metrics
    python run.py --mode knee_point    # Run 1: dev set, basic conditions, 10 severities
    python run.py --mode benchmark     # Run 2: test set, all conditions, 3 severities
    python run.py --mode train_eval    # Run 3: train subset, basic conditions
    python run.py --mode analyze       # Generate all figures and tables from saved JSONs
    python run.py --mode qualitative   # Select samples and generate example tables
    python run.py --mode all           # Everything in sequence
'''
import os, sys, time, json, random, argparse
import numpy as np
import torch
import yaml

from pathlib import Path
from transformers import logging
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MSKA_DIR = os.path.join(PROJECT_ROOT, 'MSKA')
sys.path.insert(0, MSKA_DIR)
logging.set_verbosity_error()

def load_configs(args): # Load project config and MSKA config
    with open(args.config, 'r', encoding='utf-8') as f:
        proj_cfg = yaml.safe_load(f)

    mska_cfg_path = os.path.join(PROJECT_ROOT, proj_cfg['paths']['mska_config'])
    with open(mska_cfg_path, 'r', encoding='utf-8') as f:
        mska_cfg = yaml.safe_load(f)
    return proj_cfg, mska_cfg


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(mska_cfg, proj_cfg, device):
    from model import SignLanguageModel

    # MSKA config paths are relative to MSKA dir
    prev_cwd = os.getcwd()
    os.chdir(MSKA_DIR)
    try:
        model_args = argparse.Namespace(device='cuda', distributed=False)
        model = SignLanguageModel(cfg=mska_cfg, args=model_args)
        model.to(device)
    finally: os.chdir(prev_cwd)
    
    ckpt_path = os.path.join(PROJECT_ROOT, proj_cfg['paths']['checkpoint'])
    print('Loading checkpoint from', ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    ret = model.load_state_dict(checkpoint['model'], strict=False)
    if ret.missing_keys: print('Missing keys', ret.missing_keys)
    if ret.unexpected_keys: print('Unexpected keys', ret.unexpected_keys)

    model.to(device)
    model.eval()
    print('Model loaded and set to eval mode.')
    return model


def create_dataset(path, mska_cfg, phase, subsample_indices=None, proj_cfg=None):
    from Tokenizer import GlossTokenizer_S2G
    from data.misaligned_dataset import MisalignedDataset
    
    prev_cwd = os.getcwd()
    os.chdir(MSKA_DIR)
    try:
        tokenizer = GlossTokenizer_S2G(mska_cfg['gloss'])
        min_frames = proj_cfg['misalignment']['min_frames'] if proj_cfg else 8
        max_len = proj_cfg['misalignment']['max_input_length'] if proj_cfg else 400

        dataset = MisalignedDataset(
            path=os.path.join(PROJECT_ROOT, path), tokenizer=tokenizer, config=mska_cfg, 
            args=argparse.Namespace(device='cuda'), phase=phase,
            min_frames=min_frames, max_input_length=max_len,
            subsample_indices=subsample_indices,
        )
    finally: os.chdir(prev_cwd)
    print(f'{dataset} (phase={phase})')
    return dataset


def mode_verify(args, proj_cfg, mska_cfg, device): # Run 0: Verify clean baseline matches reported MSKA numbers
    from evaluator import verify_clean_baseline
    model = load_model(mska_cfg, proj_cfg, device)

    # Run on both dev and test
    for split, path_key, label in [('val', 'dev_label_path', 'Dev'), ('test', 'test_label_path', 'Test')]:
        print(f"\n{'='*50}\nVerifying on {label} set\n{'='*50}")
        dataset = create_dataset(proj_cfg['data'][path_key], mska_cfg, split, proj_cfg=proj_cfg)
        eval_cfg = proj_cfg['evaluation']
        expected = proj_cfg.get('expected_clean') if split == 'test' else None
        verify_clean_baseline(
            model, dataset, mska_cfg,
            output_dir=os.path.join(PROJECT_ROOT, proj_cfg['paths']['results_dir']),
            batch_size=eval_cfg['batch_size'],
            num_workers=eval_cfg['num_workers'],
            beam_size=eval_cfg['beam_size'],
            generate_cfg=eval_cfg['translation'],
            expected=expected,
        )


def mode_knee_point(args, proj_cfg, mska_cfg, device): # Run 1: Knee point analysis on dev set, basic conditions, fine severity
    from evaluator import run_evaluation
    from data.misalign import generate_conditions

    model = load_model(mska_cfg, proj_cfg, device)
    kp_cfg = proj_cfg['misalignment']['knee_point']
    severity_levels = kp_cfg['severity_levels']

    dataset = create_dataset(proj_cfg['data']['dev_label_path'], mska_cfg, 'val', proj_cfg=proj_cfg)
    conditions = generate_conditions(severity_levels, include_compound=False)
    print(f'Knee point analysis: {len(conditions)} conditions, {len(dataset)} samples')

    output_path = os.path.join(PROJECT_ROOT, proj_cfg['paths']['results_dir'], 'raw', 'knee_point.json')
    run_evaluation(
        model, dataset, conditions, mska_cfg, output_path,
        batch_size=proj_cfg['evaluation']['batch_size'],
        num_workers=proj_cfg['evaluation']['num_workers'],
        beam_size=proj_cfg['evaluation']['beam_size'],
        generate_cfg=proj_cfg['evaluation']['translation'],
        collect_logits=True,
    )


def mode_benchmark(args, proj_cfg, mska_cfg, device): # Run 2: Full benchmark on test set, all conditions, coarse severity
    from evaluator import run_evaluation
    from data.misalign import generate_conditions

    model = load_model(mska_cfg, proj_cfg, device)
    bench_cfg = proj_cfg['misalignment']['benchmark']
    severity_levels = bench_cfg['severity_levels']

    dataset = create_dataset(proj_cfg['data']['test_label_path'], mska_cfg, 'test', proj_cfg=proj_cfg)
    conditions = generate_conditions(severity_levels, include_compound=True)
    print(f'Benchmark: {len(conditions)} conditions, {len(dataset)} samples')

    output_path = os.path.join(PROJECT_ROOT, proj_cfg['paths']['results_dir'], 'raw', 'benchmark.json')
    run_evaluation(
        model, dataset, conditions, mska_cfg, output_path,
        batch_size=proj_cfg['evaluation']['batch_size'],
        num_workers=proj_cfg['evaluation']['num_workers'],
        beam_size=proj_cfg['evaluation']['beam_size'],
        generate_cfg=proj_cfg['evaluation']['translation'],
        collect_logits=True,
    )


def mode_train_eval(args, proj_cfg, mska_cfg, device): # Run 3: Train subset evaluation for structural vulnerability diagnostic 
    from evaluator import run_evaluation
    from data.misalign import generate_conditions

    model = load_model(mska_cfg, proj_cfg, device)
    te_cfg = proj_cfg['misalignment']['train_eval']
    severity_levels = te_cfg['severity_levels']

    # Load full train dataset first, then subsample in-place.
    # This avoids a separate load_dataset_file call and ensures subsample indices
    # are always valid positions within dataset.list (which __len__ now reflects).
    dataset = create_dataset(proj_cfg['data']['train_label_path'], mska_cfg, 'val', proj_cfg=proj_cfg)
    n_total = len(dataset)
    
    subsample_size = min(te_cfg['subsample_size'], n_total)
    rng = np.random.RandomState(proj_cfg['seed'])
    subsample_indices = sorted(rng.choice(n_total, subsample_size, replace=False).tolist())
    dataset.list = [dataset.list[i] for i in subsample_indices]
    print(f'Train eval: subsampled to {len(dataset.list)}/{n_total} samples')

    conditions = generate_conditions(severity_levels, include_compound=False)
    print(f'Train eval: {len(conditions)} conditions')

    output_path = os.path.join(PROJECT_ROOT, proj_cfg['paths']['results_dir'], 'raw', 'train_eval.json')
    run_evaluation(
        model, dataset, conditions, mska_cfg, output_path,
        batch_size=proj_cfg['evaluation']['batch_size'],
        num_workers=proj_cfg['evaluation']['num_workers'],
        beam_size=proj_cfg['evaluation']['beam_size'],
        generate_cfg=proj_cfg['evaluation']['translation'],
    )


def mode_analyze(args, proj_cfg, mska_cfg, device): # Generate all figures and tables from saved JSON results
    from analysis.visualize import generate_all_figures
    from analysis.tables import generate_all_tables

    results_dir = os.path.join(PROJECT_ROOT, proj_cfg['paths']['results_dir'])
    fig_dir = os.path.join(results_dir, 'figures')
    table_dir = os.path.join(results_dir, 'tables')
    generate_all_figures(results_dir, fig_dir)
    generate_all_tables(results_dir, table_dir)


def mode_qualitative(args, proj_cfg, mska_cfg, device): # Select representative samples and generate example tables
    from analysis.qualitative import generate_qualitative_report
    results_dir = os.path.join(PROJECT_ROOT, proj_cfg['paths']['results_dir'])
    output_dir = os.path.join(results_dir, 'tables')
    generate_qualitative_report(results_dir, output_dir)


def main():
    parser = argparse.ArgumentParser(description='Temporal Misalignment Analysis for SLT')
    parser.add_argument(
        '--mode', type=str, required=True, help='Evaluation mode to run',
        choices=['verify', 'knee_point', 'benchmark', 'train_eval', 'analyze', 'qualitative', 'all']
    )
    parser.add_argument(
        '--config', type=str, help='Path to project config',
        default=os.path.join(PROJECT_ROOT, 'configs', 'config.yaml')
    )
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    args = parser.parse_args()

    proj_cfg, mska_cfg = load_configs(args)
    if args.batch_size: proj_cfg['evaluation']['batch_size'] = args.batch_size
    seed = proj_cfg.get('seed', 42)
    set_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    t_start = time.time()

    mode_map = {
        'verify': mode_verify, 'knee_point': mode_knee_point, 'benchmark': mode_benchmark,
        'train_eval': mode_train_eval, 'analyze': mode_analyze, 'qualitative': mode_qualitative,
    }
    if args.mode == 'all':
        for mode_name in ['verify', 'knee_point', 'benchmark', 'train_eval', 'analyze', 'qualitative']:
            print(f"\n{'#'*60}\n# Mode: {mode_name}\n{'#'*60}")
            mode_map[mode_name](args, proj_cfg, mska_cfg, device)
    else: mode_map[args.mode](args, proj_cfg, mska_cfg, device)
    elapsed = time.time() - t_start
    print(f'Total elapsed time: {elapsed:.1f}s ({elapsed/60:.1f}m)')


if __name__ == '__main__':
    main()