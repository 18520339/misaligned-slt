'''Entry point for temporal misalignment analysis of sign language translation.

Phase 1 (analysis):
    python run.py --mode verify                      # Verify MSKA works, log clean metrics
    python run.py --mode knee_point                  # Run 1: dev set, basic conditions, 10 severities
    python run.py --mode benchmark                   # Run 2: test set, all conditions, 3 severities
    python run.py --mode train_eval                  # Run 3: train subset, basic conditions
    python run.py --mode analyze_phase1              # Generate all figures and tables from saved JSONs
    python run.py --mode qualitative                 # Select samples and generate example tables

Phase 2 (training & evaluation):
    python run.py --mode train --model ar_aug        # Train Model A (AR + Aug)
    python run.py --mode train --model bd_clean      # Train Model B (BD Clean)
    python run.py --mode train --model bd_aug        # Train Model C (BD + Aug)
    python run.py --mode evaluate --model ar_aug     # Evaluate on all conditions
    python run.py --mode evaluate --model bd_clean
    python run.py --mode evaluate --model bd_aug
    python run.py --mode analyze_phase2              # Phase 2 comparison figures
'''
import os, sys, time, json, random, argparse, yaml
import numpy as np
import wandb
import torch

from pathlib import Path
from transformers import logging
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MSKA_DIR = os.path.join(PROJECT_ROOT, 'MSKA')
sys.path.insert(0, MSKA_DIR)
logging.set_verbosity_error()


def load_base_configs(args): # Load project config and MSKA config
    with open(args.config, 'r', encoding='utf-8') as f:
        proj_cfg = yaml.safe_load(f)

    mska_cfg_path = os.path.join(PROJECT_ROOT, proj_cfg['paths']['mska_config'])
    with open(mska_cfg_path, 'r', encoding='utf-8') as f:
        mska_cfg = yaml.safe_load(f)
    return proj_cfg, mska_cfg


def get_min_frames(mska_cfg, train_label_path=None):
    '''Compute min_frames from DSTA-Net temporal strides AND training data.

    Two constraints:
      1. Architectural floor: DSTA-Net strides determine the minimum input
         length for the encoder to produce >= 2 timesteps.
      2. Data floor: 5th percentile of training sequence lengths — any
         truncation leaving fewer frames than this is statistically extreme.

    Returns max(architectural_floor, data_5th_percentile).
    '''
    from data.misalign import compute_min_frames, MIN_FRAMES_DEFAULT

    # 1. Architectural floor from DSTA-Net temporal strides
    arch_min = MIN_FRAMES_DEFAULT
    try:
        layers = mska_cfg['model']['RecognitionNetwork']['DSTA-Net']['net']
        strides = [layer[4] for layer in layers]  # 5th element is temporal stride
        arch_min = compute_min_frames(strides)
    except (KeyError, IndexError): pass

    # 2. Data floor: 5th percentile of training set sequence lengths
    data_min = 0
    if train_label_path is not None:
        try:
            import gzip, pickle
            full_path = os.path.join(PROJECT_ROOT, train_label_path) if not os.path.isabs(train_label_path) else train_label_path
            if os.path.exists(full_path):
                with gzip.open(full_path, 'rb') as f:
                    data = pickle.load(f)
                lengths = [sample['num_frames'] for sample in data.values() if isinstance(sample, dict) and 'num_frames' in sample]
                if lengths:
                    data_min = int(np.percentile(lengths, 5))
                    print(f'Training data: {len(lengths)} samples, 5th percentile length = {data_min} frames')
        except Exception as e: print(f'  Warning: could not compute 5th percentile: {e}')

    result = max(arch_min, data_min)
    print(f'min_frames = {result} (arch={arch_min}, data_5th={data_min})')
    return result


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def build_model(mska_cfg, model_cfg, device):
    from model_factory import SLTModel
    prev_cwd = os.getcwd()
    os.chdir(MSKA_DIR)
    try:
        model_args = argparse.Namespace(device='cuda', distributed=False)
        decoder_type = model_cfg.get('decoder_type', 'ar')
        model = SLTModel(mska_cfg, model_args, model_cfg, decoder_type=decoder_type)
    finally: os.chdir(prev_cwd)
    return model


def load_phase1_model(mska_cfg, proj_cfg, device):
    '''Load pretrained MSKA model for Phase 1 evaluation.

    Reuses build_model + SLTModel.load_pretrained (same path as Phase 2 AR models) 
    to avoid duplicating model construction and checkpoint loading logic.
    '''
    model_cfg = {'decoder_type': 'ar'}  # minimal config for AR model
    model = build_model(mska_cfg, model_cfg, device)
    ckpt_path = os.path.join(PROJECT_ROOT, proj_cfg['paths']['checkpoint'])
    model.load_pretrained(ckpt_path, strict=False)
    model.to(device)
    model.eval()
    print('Model loaded and set to eval mode.')
    return model


def create_eval_dataset(path, mska_cfg, phase, subsample_indices=None, proj_cfg=None):
    from Tokenizer import GlossTokenizer_S2G
    from data.misaligned_datasets import EvalDataset

    prev_cwd = os.getcwd()
    os.chdir(MSKA_DIR)
    try:
        tokenizer = GlossTokenizer_S2G(mska_cfg['gloss'])
        max_len = proj_cfg['misalignment']['max_input_length'] if proj_cfg else 400
        train_path = proj_cfg['data']['train_label_path'] if proj_cfg else None
        dataset = EvalDataset(
            path=os.path.join(PROJECT_ROOT, path), tokenizer=tokenizer, config=mska_cfg,
            args=argparse.Namespace(device='cuda'), phase=phase,
            min_frames=get_min_frames(mska_cfg, train_label_path=train_path),
            max_input_length=max_len, subsample_indices=subsample_indices,
        )
    finally: os.chdir(prev_cwd)
    print(f'{dataset} (phase={phase})')
    return dataset


def mode_verify(args, proj_cfg, mska_cfg, device): # Run 0: Verify clean baseline matches reported MSKA numbers
    from evaluator import verify_clean_baseline
    model = load_phase1_model(mska_cfg, proj_cfg, device)

    # Run on both dev and test
    for split, path_key, label in [('val', 'dev_label_path', 'Dev'), ('test', 'test_label_path', 'Test')]:
        print(f"\n{'='*50}\nVerifying on {label} set\n{'='*50}")
        dataset = create_eval_dataset(proj_cfg['data'][path_key], mska_cfg, split, proj_cfg=proj_cfg)
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

    model = load_phase1_model(mska_cfg, proj_cfg, device)
    kp_cfg = proj_cfg['misalignment']['knee_point']
    severity_levels = kp_cfg['severity_levels']

    dataset = create_eval_dataset(proj_cfg['data']['dev_label_path'], mska_cfg, 'val', proj_cfg=proj_cfg)
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

    model = load_phase1_model(mska_cfg, proj_cfg, device)
    bench_cfg = proj_cfg['misalignment']['benchmark']
    severity_levels = bench_cfg['severity_levels']

    dataset = create_eval_dataset(proj_cfg['data']['test_label_path'], mska_cfg, 'test', proj_cfg=proj_cfg)
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

    model = load_phase1_model(mska_cfg, proj_cfg, device)
    train_eval_cfg = proj_cfg['misalignment']['train_eval']
    severity_levels = train_eval_cfg['severity_levels']

    # Load full train dataset first, then subsample in-place.
    # This avoids a separate load_dataset_file call and ensures subsample indices
    # are always valid positions within dataset.list (which __len__ now reflects).
    dataset = create_eval_dataset(proj_cfg['data']['train_label_path'], mska_cfg, 'val', proj_cfg=proj_cfg)
    n_total = len(dataset)
    
    subsample_size = min(train_eval_cfg['subsample_size'], n_total)
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


def mode_analyze_phase1(args, proj_cfg, mska_cfg, device): # Generate all figures and tables from saved JSON results
    from analysis.visualize_phase1 import generate_all_figures
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


# ============================= Phase 2 modes =============================
def load_model_cfg(model_name):
    cfg_path = os.path.join(PROJECT_ROOT, 'configs', f'{model_name}.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def mode_train(args, proj_cfg, mska_cfg, device): # Phase 2 training: train AR+Aug, BD Clean, or BD+Aug
    from trainer import train_model
    from data.misaligned_datasets import TrainDataset
    from Tokenizer import GlossTokenizer_S2G

    model_name = args.model
    model_cfg = load_model_cfg(model_name)
    train_cfg = model_cfg['training']
    aug_cfg = model_cfg.get('augmentation', {})
    decoder_type = model_cfg.get('decoder_type', 'ar')

    print(f'\n=== Training {model_name} (decoder={decoder_type}) ===\n')
    model = build_model(mska_cfg, model_cfg, device)

    # Weight initialization
    train_from_scratch = train_cfg.get('train_from_scratch', False)
    if train_from_scratch: print('train_from_scratch=True: all components initialized randomly.')
    else:
        ckpt_path = os.path.join(PROJECT_ROOT, proj_cfg['paths']['checkpoint'])
        model.load_pretrained(ckpt_path, strict=False)

    # Resume from Phase 2 checkpoint (e.g., bd_aug resumes from bd_clean)
    resume_from = train_cfg.get('resume_from')
    if not train_from_scratch and resume_from:
        full_resume = os.path.join(PROJECT_ROOT, resume_from)
        if os.path.exists(full_resume):
            print(f'Loading Phase 2 checkpoint: {full_resume}')
            ckpt = torch.load(full_resume, map_location='cpu')
            model.load_state_dict(ckpt['model'], strict=False)

    # Build training dataset
    prev_cwd = os.getcwd()
    os.chdir(MSKA_DIR)
    try:
        tokenizer = GlossTokenizer_S2G(mska_cfg['gloss'])
        train_path = os.path.join(PROJECT_ROOT, proj_cfg['data']['train_label_path'])
        model_args = argparse.Namespace(device='cuda')
        if aug_cfg.get('enabled', False):
            train_dataset = TrainDataset(
                path=train_path, tokenizer=tokenizer, config=mska_cfg, args=model_args, phase='train',
                p_aug=aug_cfg.get('p_aug', 0.5), knee_thresholds=aug_cfg.get('knee_thresholds'),
                min_severity=aug_cfg.get('min_severity', 0.05),
                min_frames=get_min_frames(mska_cfg, train_label_path=proj_cfg['data']['train_label_path']),
                max_input_length=proj_cfg['misalignment'].get('max_input_length', 400),
            )
            print('Note: Dev dataset is clean for quick validation.')
        else:
            from datasets import S2T_Dataset
            train_dataset = S2T_Dataset(
                path=train_path, tokenizer=tokenizer, config=mska_cfg,
                args=model_args, phase='train', training_refurbish=True
            )

        # Dev dataset (clean, for quick validation)
        dev_path = os.path.join(PROJECT_ROOT, proj_cfg['data']['dev_label_path'])
        from datasets import S2T_Dataset
        dev_dataset = S2T_Dataset(
            path=dev_path, tokenizer=tokenizer, config=mska_cfg,
            args=model_args, phase='val', training_refurbish=True
        )
    finally: os.chdir(prev_cwd)
    print(f'Train: {len(train_dataset)} samples, Dev: {len(dev_dataset)} samples')

    # Param count summary
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {total:,} total, {trainable:,} trainable')

    wandb_run = wandb.init(
        project='misalign-slt', name=model_name,
        config={**model_cfg, 'dataset': proj_cfg['data']['dataset_name']},
    )

    output_dir = os.path.join(PROJECT_ROOT, proj_cfg['paths']['results_dir'], 'checkpoints', model_name)
    train_model(
        model=model, train_dataset=train_dataset, dev_dataset=dev_dataset,
        mska_cfg=mska_cfg, model_cfg=model_cfg, train_cfg=train_cfg,
        output_dir=output_dir, device=device, wandb_run=wandb_run,
    )
    wandb_run.finish()


def mode_evaluate_phase2(args, proj_cfg, mska_cfg, device): # Phase 2 evaluation: run a trained model through the full benchmark
    from evaluator import run_evaluation
    from data.misalign import generate_conditions

    model_name = args.model
    model_cfg = load_model_cfg(model_name)
    decoder_type = model_cfg.get('decoder_type', 'ar')
    print(f'\n=== Evaluating {model_name} (decoder={decoder_type}) ===\n')
    model = build_model(mska_cfg, model_cfg, device)

    # Load best checkpoint
    ckpt_dir = os.path.join(PROJECT_ROOT, proj_cfg['paths']['results_dir'], 'checkpoints', model_name)
    ckpt_path = os.path.join(ckpt_dir, 'best_checkpoint.pth')
    if not os.path.exists(ckpt_path):
        print(f'ERROR: No checkpoint found at {ckpt_path}')
        return
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} (BLEU-4={ckpt.get('best_bleu4', '?')})")

    # Create test dataset and Generate conditions (same as Phase 1 benchmark)
    dataset = create_eval_dataset(proj_cfg['data']['test_label_path'], mska_cfg, 'test', proj_cfg=proj_cfg)
    bench_cfg = proj_cfg['misalignment']['benchmark']
    conditions = generate_conditions(bench_cfg['severity_levels'], include_compound=True)
    print(f'Evaluating {len(conditions)} conditions on {len(dataset)} samples')

    # Generate config
    eval_cfg = proj_cfg['evaluation']
    generate_cfg = dict(eval_cfg['translation'])
    if decoder_type == 'bd':
        bd_cfg = model_cfg.get('block_diffusion', {})
        generate_cfg['diffusion_steps'] = bd_cfg.get('diffusion_steps', 10)

    output_path = os.path.join(PROJECT_ROOT, proj_cfg['paths']['results_dir'], 'raw', f'benchmark_{model_name}.json')
    run_evaluation(
        model, dataset, conditions, mska_cfg, output_path,
        batch_size=eval_cfg['batch_size'],
        num_workers=eval_cfg['num_workers'],
        beam_size=eval_cfg['beam_size'],
        generate_cfg=generate_cfg,
        collect_logits=True,
    )


def mode_analyze_phase2(args, proj_cfg, mska_cfg, device): # Generate Phase 2 comparison figures and tables
    from analysis.visualize_phase2 import generate_phase2_figures
    results_dir = os.path.join(PROJECT_ROOT, proj_cfg['paths']['results_dir'])
    fig_dir = os.path.join(results_dir, 'figures')
    generate_phase2_figures(results_dir, fig_dir)


def main():
    parser = argparse.ArgumentParser(description='Temporal Misalignment Analysis for SLT')
    parser.add_argument('--mode', type=str, required=True, help='Execution mode', choices=[
        'verify', 'knee_point', 'benchmark', 'train_eval', 'analyze_phase1', 'qualitative', 
        'train', 'evaluate', 'analyze_phase2'
    ])
    parser.add_argument(
        '--config', type=str, help='Path to project config',
        default=os.path.join(PROJECT_ROOT, 'configs', 'config.yaml')
    )
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--model', type=str, default=None, help='Phase 2 model name (ar_aug, bd_clean, bd_aug)')
    args = parser.parse_args()

    proj_cfg, mska_cfg = load_base_configs(args)
    if args.batch_size: proj_cfg['evaluation']['batch_size'] = args.batch_size
    seed = proj_cfg.get('seed', 42)
    set_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    t_start = time.time()
    phase1_modes = {
        'verify': mode_verify, 'knee_point': mode_knee_point, 'benchmark': mode_benchmark,
        'train_eval': mode_train_eval, 'analyze_phase1': mode_analyze_phase1, 'qualitative': mode_qualitative,
    }
    phase2_modes = {'train': mode_train, 'evaluate': mode_evaluate_phase2, 'analyze_phase2': mode_analyze_phase2}
    
    if args.mode in phase1_modes: phase1_modes[args.mode](args, proj_cfg, mska_cfg, device)
    elif args.mode in phase2_modes:
        if args.mode in ('train', 'evaluate') and not args.model:
            print('ERROR: --model is required for Phase 2 train/evaluate modes')
            print('  Options: ar_aug, bd_clean, bd_aug')
            sys.exit(1)
        phase2_modes[args.mode](args, proj_cfg, mska_cfg, device)

    elapsed = time.time() - t_start
    print(f'Total elapsed time: {elapsed:.1f}s ({elapsed/60:.1f}m)')


if __name__ == '__main__':
    main()