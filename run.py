'''Entry point for temporal misalignment analysis of sign language translation.

Uses GFSLT-VLP (gloss-free) pipeline with RGB/pose backbone + mBART. Paper defaults
match the original GFSLT-VLP setup (SGD lr=0.01 momentum=0.9, CosineAnnealingLR
eta_min=1e-8, label_smoothing=0.2, max_new_tokens=150, num_beams=4).

Phase 1 (analysis with pretrained model):
    python run.py --mode verify
    python run.py --mode knee_point
    python run.py --mode benchmark
    python run.py --mode train_eval
    python run.py --mode analyze_phase1
    python run.py --mode qualitative

Phase 2 (training & evaluation):
    python run.py --mode vlp --model ar_clean
    python run.py --mode train --model {ar_clean, ar_aug, bd_clean, bd_aug}
    python run.py --mode evaluate --model {ar_clean, ar_aug, bd_clean, bd_aug}
    python run.py --mode analyze_phase2
'''
import os
import sys
import time
import random
import argparse
import yaml
import dataclasses
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from transformers import AutoTokenizer, MBartForConditionalGeneration, logging as hf_logging
hf_logging.set_verbosity_error()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GFSLT_DIR = os.path.join(PROJECT_ROOT, 'gfslt-pose-trainer')
sys.path.insert(0, GFSLT_DIR)

from models import (
    GFSLT, GFSLTConfig, VisualEncoder, build_backbone, load_stage1_weights,
)
from block_diffusion import BlockDiffusionDecoder


# ─── Config loading ──────────────────────────────────────────────────────────

def _load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_config(args):
    '''Load project config and merge the dataset-specific config into cfg["data"].'''
    cfg = _load_yaml(args.config)
    data_cfg = cfg.get('data', {})
    ds_path = data_cfg.get('config_path')
    if ds_path:
        full = os.path.join(PROJECT_ROOT, ds_path) if not os.path.isabs(ds_path) else ds_path
        ds = _load_yaml(full)
        merged = dict(ds)
        merged.update({k: v for k, v in data_cfg.items() if k != 'config_path'})
        cfg['data'] = merged
    return cfg


def load_model_cfg(model_name):
    '''Load model-specific config YAML.'''
    cfg_path = os.path.join(PROJECT_ROOT, 'configs', f'{model_name}.yaml')
    return _load_yaml(cfg_path)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_tokenizer(cfg):
    '''Build mBART tokenizer from config.'''
    model_cfg = cfg.get('model', {})
    mbart_name = model_cfg.get('mbart_name', 'facebook/mbart-large-cc25')
    tgt_lang = cfg.get('data', {}).get('tgt_lang', 'de_DE')
    src_lang = cfg.get('data', {}).get('src_lang', tgt_lang)
    tokenizer = AutoTokenizer.from_pretrained(
        mbart_name, src_lang=src_lang, tgt_lang=tgt_lang,
    )
    return tokenizer


# ─── Model building ──────────────────────────────────────────────────────────

def build_gfslt_config(model_cfg_inner, data_cfg=None):
    '''Filter model-cfg dict to valid GFSLTConfig fields.'''
    valid = {f.name for f in dataclasses.fields(GFSLTConfig)}
    filtered = {k: v for k, v in (model_cfg_inner or {}).items() if k in valid}
    # Fallback: let data_cfg override input_modality if model_cfg didn't set it.
    if data_cfg is not None and 'input_modality' not in filtered:
        mod = data_cfg.get('input_modality')
        if mod is not None:
            filtered['input_modality'] = mod
    return GFSLTConfig(**filtered)


class SLTModel(nn.Module):
    '''Thin wrapper that routes AR (GFSLT) or BD (BlockDiffusionDecoder) forward/generate.

    - AR: owns a full GFSLT (backbone + sign_emb + mbart).
    - BD: owns backbone + sign_emb + bd_decoder (which owns mbart's encoder/decoder).
    Parameters are registered exactly once under each sub-path.
    '''
    def __init__(self, gfslt_config, tokenizer, decoder_type='ar', bd_config=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.decoder_type = decoder_type
        self.config = gfslt_config

        if decoder_type == 'ar':
            self.gfslt = GFSLT(gfslt_config)
            self.bd_decoder = None
        elif decoder_type == 'bd':
            mbart = MBartForConditionalGeneration.from_pretrained(gfslt_config.mbart_name)
            self.backbone = build_backbone(gfslt_config)
            self.sign_emb = VisualEncoder(
                emb_size=mbart.config.d_model, feature_size=gfslt_config.embed_dim,
            )
            self.bd_decoder = BlockDiffusionDecoder(
                mbart=mbart, backbone=self.backbone, sign_emb=self.sign_emb,
                embed_scale=1.0, tokenizer=tokenizer, **(bd_config or {}),
            )
        else:
            raise ValueError(f'Unknown decoder_type: {decoder_type}')

    def forward(self, pixel_values, pixel_mask,
                paragraph_tokens=None, paragraph_attention_mask=None,
                labels=None, labels_mask=None, **kwargs):
        if self.decoder_type == 'bd':
            bd_labels = labels if labels is not None else paragraph_tokens
            return self.bd_decoder(
                pixel_values=pixel_values, pixel_mask=pixel_mask,
                labels=bd_labels,
            )
        return self.gfslt(
            pixel_values=pixel_values, pixel_mask=pixel_mask,
            paragraph_tokens=paragraph_tokens,
            paragraph_attention_mask=paragraph_attention_mask,
        )

    @torch.no_grad()
    def generate(self, pixel_values, pixel_mask, **kwargs):
        if self.decoder_type == 'bd':
            return self.bd_decoder.generate(
                pixel_values=pixel_values, pixel_mask=pixel_mask, **kwargs,
            )
        return self.gfslt.generate(
            pixel_values=pixel_values, pixel_mask=pixel_mask, **kwargs,
        )

    def load_pretrained(self, path, strict=False):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
        return self.load_state_dict(state, strict=strict)


def apply_stage1_weights(model: SLTModel, vlp_path: str, mbart_name: str):
    '''Map SLRCLIP (Stage 1) weights into the SLTModel (AR or BD).'''
    if model.decoder_type == 'ar':
        load_stage1_weights(model.gfslt, vlp_path, mbart_name)
        return

    # BD path: load weights into a throwaway GFSLT, then transplant.
    print('[apply_stage1_weights] BD model: loading via throwaway GFSLT...')
    tmp = GFSLT(model.config)
    load_stage1_weights(tmp, vlp_path, mbart_name)
    model.backbone.load_state_dict(tmp.backbone.state_dict())
    model.sign_emb.load_state_dict(tmp.sign_emb.state_dict())
    model.bd_decoder.mbart_encoder.load_state_dict(tmp.mbart.model.encoder.state_dict())
    # Decoder embed_tokens/positions are already initialised from the trimmed mBART by
    # BlockDiffusionDecoder.__init__, so we only pull in encoder + visual weights here.
    del tmp


def build_model(model_cfg, tokenizer, data_cfg=None):
    '''Build SLTModel (AR or BD) from config.'''
    gfslt_config = build_gfslt_config(model_cfg.get('model', {}), data_cfg=data_cfg)
    decoder_type = model_cfg.get('decoder_type', 'ar')
    bd_config = model_cfg.get('block_diffusion', None)
    return SLTModel(
        gfslt_config=gfslt_config, tokenizer=tokenizer,
        decoder_type=decoder_type, bd_config=bd_config,
    )


# ─── Dataset building ────────────────────────────────────────────────────────

def get_min_frames(cfg):
    from data.misalign import compute_min_frames
    conv_type = cfg.get('model', {}).get('temporal_conv_type', 2)
    return compute_min_frames(temporal_conv_type=conv_type)


def _common_dataset_kwargs(cfg):
    '''Construct the dataset kwargs shared by EvalDataset and TrainDataset.'''
    data_cfg = cfg['data']
    frames_dir = data_cfg.get('frames_dir')
    if frames_dir and not os.path.isabs(frames_dir):
        frames_dir = os.path.join(PROJECT_ROOT, frames_dir)
    frame_dims = data_cfg.get('frame_dims', [210, 260])
    return {
        'input_modality': data_cfg.get('input_modality', 'rgb'),
        'frame_dims': tuple(frame_dims),
        'frames_dir': frames_dir,
        'input_size': data_cfg.get('input_size', 224),
        'resize': data_cfg.get('resize', 256),
        'max_length': data_cfg.get('max_length',
                                  cfg.get('misalignment', {}).get('max_input_length', 300)),
    }


def create_eval_dataset(data_path, tokenizer, phase, cfg, subsample_indices=None):
    '''Create EvalDataset for Phase 1/2 evaluation.'''
    from data.misaligned_datasets import EvalDataset
    full_path = os.path.join(PROJECT_ROOT, data_path) if not os.path.isabs(data_path) else data_path
    kwargs = _common_dataset_kwargs(cfg)
    return EvalDataset(
        data_path=full_path, tokenizer=tokenizer, phase=phase,
        min_frames=get_min_frames(cfg),
        subsample_indices=subsample_indices,
        **kwargs,
    )


def create_train_dataset(data_path, tokenizer, cfg, model_cfg):
    '''Create TrainDataset (with optional misalignment augmentation).'''
    from data.misaligned_datasets import TrainDataset
    full_path = os.path.join(PROJECT_ROOT, data_path) if not os.path.isabs(data_path) else data_path
    aug_cfg = model_cfg.get('augmentation', {})
    train_cfg = model_cfg.get('training', {})
    kwargs = _common_dataset_kwargs(cfg)

    return TrainDataset(
        data_path=full_path, tokenizer=tokenizer, phase='train',
        min_frames=get_min_frames(cfg),
        p_aug=aug_cfg.get('p_aug', 0.0) if aug_cfg.get('enabled', False) else 0.0,
        knee_thresholds=aug_cfg.get('knee_thresholds'),
        min_severity=aug_cfg.get('min_severity', 0.05),
        training_refurbish=train_cfg.get('training_refurbish', False),
        pose_augment=train_cfg.get('pose_augment', False),
        rgb_augment=train_cfg.get('rgb_augment', True),
        **kwargs,
    )


def create_vlp_train_dataset(data_path, tokenizer, cfg, vlp_cfg):
    '''Plain SLTDataset for Stage 1 VLP pretraining (no misalignment aug).'''
    from loader import SLTDataset
    full_path = os.path.join(PROJECT_ROOT, data_path) if not os.path.isabs(data_path) else data_path
    kwargs = _common_dataset_kwargs(cfg)
    return SLTDataset(
        data_path=full_path, tokenizer=tokenizer, phase='train',
        training_refurbish=True,
        noise_rate=vlp_cfg.get('noise_rate', 0.15),
        noise_type=vlp_cfg.get('noise_type', 'omit_last'),
        pose_augment=vlp_cfg.get('pose_augment', False),
        rgb_augment=vlp_cfg.get('rgb_augment', True),
        **kwargs,
    )


# ─── Phase 1 Analysis Modes ──────────────────────────────────────────────────

def load_phase1_model(cfg, tokenizer, device):
    '''Load pretrained model for Phase 1 evaluation.'''
    model_cfg = {'model': cfg.get('model', {}), 'decoder_type': 'ar'}
    model = build_model(model_cfg, tokenizer, data_cfg=cfg.get('data', {}))

    ckpt_path = cfg['paths'].get('checkpoint')
    if ckpt_path:
        full = os.path.join(PROJECT_ROOT, ckpt_path) if not os.path.isabs(ckpt_path) else ckpt_path
        if os.path.exists(full):
            model.load_pretrained(full, strict=False)

    vlp_path = cfg['paths'].get('vlp_checkpoint')
    if vlp_path:
        full = os.path.join(PROJECT_ROOT, vlp_path) if not os.path.isabs(vlp_path) else vlp_path
        if os.path.exists(full):
            apply_stage1_weights(model, full, cfg['model']['mbart_name'])

    model.to(device)
    model.eval()
    print('Model loaded and set to eval mode.')
    return model


def mode_verify(args, cfg, device):
    from evaluator import verify_clean_baseline
    tokenizer = build_tokenizer(cfg)
    model = load_phase1_model(cfg, tokenizer, device)

    for split, path_key, label in [
        ('val', 'dev_label_path', 'Dev'), ('test', 'test_label_path', 'Test'),
    ]:
        print(f"\n{'=' * 50}\nVerifying on {label} set\n{'=' * 50}")
        dataset = create_eval_dataset(cfg['data'][path_key], tokenizer, split, cfg)
        eval_cfg = cfg['evaluation']
        expected = cfg.get('expected_clean') if split == 'test' else None
        verify_clean_baseline(
            model, dataset,
            output_dir=os.path.join(PROJECT_ROOT, cfg['paths']['results_dir']),
            tokenizer=tokenizer,
            batch_size=eval_cfg['batch_size'],
            num_workers=eval_cfg['num_workers'],
            generate_cfg=eval_cfg['translation'],
            expected=expected,
        )


def mode_knee_point(args, cfg, device):
    from evaluator import run_evaluation
    from data.misalign import generate_conditions

    tokenizer = build_tokenizer(cfg)
    model = load_phase1_model(cfg, tokenizer, device)
    kp_cfg = cfg['misalignment']['knee_point']

    dataset = create_eval_dataset(cfg['data']['dev_label_path'], tokenizer, 'val', cfg)
    conditions = generate_conditions(kp_cfg['severity_levels'], include_compound=False)
    print(f'Knee point analysis: {len(conditions)} conditions, {len(dataset)} samples')

    output_path = os.path.join(PROJECT_ROOT, cfg['paths']['results_dir'], 'raw', 'knee_point.json')
    run_evaluation(
        model, dataset, conditions, output_path, tokenizer=tokenizer,
        batch_size=cfg['evaluation']['batch_size'],
        num_workers=cfg['evaluation']['num_workers'],
        generate_cfg=cfg['evaluation']['translation'],
    )


def mode_benchmark(args, cfg, device):
    from evaluator import run_evaluation
    from data.misalign import generate_conditions

    tokenizer = build_tokenizer(cfg)
    model = load_phase1_model(cfg, tokenizer, device)
    bench_cfg = cfg['misalignment']['benchmark']

    dataset = create_eval_dataset(cfg['data']['test_label_path'], tokenizer, 'test', cfg)
    conditions = generate_conditions(bench_cfg['severity_levels'], include_compound=True)
    print(f'Benchmark: {len(conditions)} conditions, {len(dataset)} samples')

    output_path = os.path.join(PROJECT_ROOT, cfg['paths']['results_dir'], 'raw', 'benchmark.json')
    run_evaluation(
        model, dataset, conditions, output_path, tokenizer=tokenizer,
        batch_size=cfg['evaluation']['batch_size'],
        num_workers=cfg['evaluation']['num_workers'],
        generate_cfg=cfg['evaluation']['translation'],
    )


def mode_train_eval(args, cfg, device):
    from evaluator import run_evaluation
    from data.misalign import generate_conditions

    tokenizer = build_tokenizer(cfg)
    model = load_phase1_model(cfg, tokenizer, device)
    train_eval_cfg = cfg['misalignment']['train_eval']

    dataset = create_eval_dataset(cfg['data']['train_label_path'], tokenizer, 'val', cfg)
    n_total = len(dataset)

    subsample_size = min(train_eval_cfg['subsample_size'], n_total)
    rng = np.random.RandomState(cfg.get('seed', 42))
    subsample_indices = sorted(rng.choice(n_total, subsample_size, replace=False).tolist())
    dataset.list = [dataset.list[i] for i in subsample_indices]
    print(f'Train eval: subsampled to {len(dataset.list)}/{n_total} samples')

    conditions = generate_conditions(train_eval_cfg['severity_levels'], include_compound=False)
    output_path = os.path.join(PROJECT_ROOT, cfg['paths']['results_dir'], 'raw', 'train_eval.json')
    run_evaluation(
        model, dataset, conditions, output_path, tokenizer=tokenizer,
        batch_size=cfg['evaluation']['batch_size'],
        num_workers=cfg['evaluation']['num_workers'],
        generate_cfg=cfg['evaluation']['translation'],
    )


def mode_analyze_phase1(args, cfg, device):
    from analysis.visualize_phase1 import generate_all_figures
    from analysis.tables import generate_all_tables
    results_dir = os.path.join(PROJECT_ROOT, cfg['paths']['results_dir'])
    generate_all_figures(results_dir, os.path.join(results_dir, 'figures'))
    generate_all_tables(results_dir, os.path.join(results_dir, 'tables'))


def mode_qualitative(args, cfg, device):
    from analysis.qualitative import generate_qualitative_report
    results_dir = os.path.join(PROJECT_ROOT, cfg['paths']['results_dir'])
    generate_qualitative_report(results_dir, os.path.join(results_dir, 'tables'))


# ─── Phase 2 Training & Evaluation ───────────────────────────────────────────

def mode_vlp(args, cfg, device):
    '''Stage 1: Visual-Language Pretraining.'''
    import wandb
    from trainer import train_vlp

    model_name = args.model
    model_cfg = load_model_cfg(model_name)
    tokenizer = build_tokenizer(cfg)

    gfslt_config = build_gfslt_config(model_cfg.get('model', {}), data_cfg=cfg.get('data', {}))
    vlp_cfg = model_cfg.get('vlp', {})

    train_dataset = create_vlp_train_dataset(
        cfg['data']['train_label_path'], tokenizer, cfg, vlp_cfg,
    )

    output_dir = os.path.join(PROJECT_ROOT, cfg['paths']['results_dir'], 'checkpoints', model_name)
    wandb_run = wandb.init(
        project='misalign-slt', name=f'{model_name}_vlp',
        config={**model_cfg, 'stage': 'vlp'},
    )
    train_vlp(
        gfslt_config=gfslt_config, tokenizer=tokenizer,
        train_dataset=train_dataset, output_dir=output_dir,
        vlp_cfg=vlp_cfg, device=device, wandb_run=wandb_run,
    )
    wandb_run.finish()


def mode_train(args, cfg, device):
    '''Stage 2: Translation training.'''
    import wandb
    from trainer import train_model

    model_name = args.model
    model_cfg = load_model_cfg(model_name)
    train_cfg = model_cfg['training']
    decoder_type = model_cfg.get('decoder_type', 'ar')

    print(f'\n=== Training {model_name} (decoder={decoder_type}) ===\n')

    tokenizer = build_tokenizer(cfg)
    model = build_model(model_cfg, tokenizer, data_cfg=cfg.get('data', {}))

    # VLP weight initialization.
    vlp_ckpt = model_cfg.get('vlp_checkpoint') or train_cfg.get('vlp_checkpoint')
    if vlp_ckpt:
        full = os.path.join(PROJECT_ROOT, vlp_ckpt) if not os.path.isabs(vlp_ckpt) else vlp_ckpt
        if os.path.exists(full):
            apply_stage1_weights(model, full, model_cfg['model']['mbart_name'])

    # Or load a pretrained Stage 2 checkpoint (for fine-tuning, e.g. ar_aug ← ar_clean).
    pretrained_ckpt = train_cfg.get('pretrained_checkpoint')
    if pretrained_ckpt:
        full = (os.path.join(PROJECT_ROOT, pretrained_ckpt)
                if not os.path.isabs(pretrained_ckpt) else pretrained_ckpt)
        if os.path.exists(full):
            model.load_pretrained(full, strict=False)

    # Datasets.
    train_dataset = create_train_dataset(
        cfg['data']['train_label_path'], tokenizer, cfg, model_cfg,
    )
    dev_dataset = create_eval_dataset(
        cfg['data']['dev_label_path'], tokenizer, 'val', cfg,
    )
    print(f'Train: {len(train_dataset)} samples, Dev: {len(dev_dataset)} samples')

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {total:,} total, {trainable:,} trainable\n')

    wandb_run = wandb.init(
        project='misalign-slt', name=model_name,
        config={**model_cfg, 'dataset': cfg['data']['dataset_name']},
    )
    output_dir = os.path.join(PROJECT_ROOT, cfg['paths']['results_dir'], 'checkpoints', model_name)
    train_model(
        model=model, train_dataset=train_dataset, dev_dataset=dev_dataset,
        model_cfg=model_cfg, train_cfg=train_cfg,
        output_dir=output_dir, tokenizer=tokenizer,
        device=device, wandb_run=wandb_run,
    )
    wandb_run.finish()


def mode_evaluate_phase2(args, cfg, device):
    '''Phase 2 evaluation: run trained model through full benchmark.'''
    from evaluator import run_evaluation
    from data.misalign import generate_conditions

    model_name = args.model
    model_cfg = load_model_cfg(model_name)
    decoder_type = model_cfg.get('decoder_type', 'ar')
    print(f'\n=== Evaluating {model_name} (decoder={decoder_type}) ===\n')

    tokenizer = build_tokenizer(cfg)
    model = build_model(model_cfg, tokenizer, data_cfg=cfg.get('data', {}))

    ckpt_dir = os.path.join(PROJECT_ROOT, cfg['paths']['results_dir'], 'checkpoints', model_name)
    ckpt_path = os.path.join(ckpt_dir, 'best_checkpoint.pth')
    if not os.path.exists(ckpt_path):
        print(f'ERROR: No checkpoint found at {ckpt_path}')
        return

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(BLEU-4={ckpt.get('best_bleu4', '?')})")

    dataset = create_eval_dataset(cfg['data']['test_label_path'], tokenizer, 'test', cfg)
    bench_cfg = cfg['misalignment']['benchmark']
    conditions = generate_conditions(bench_cfg['severity_levels'], include_compound=True)
    print(f'Evaluating {len(conditions)} conditions on {len(dataset)} samples')

    eval_cfg = cfg['evaluation']
    generate_cfg = dict(eval_cfg['translation'])
    if decoder_type == 'bd':
        bd_cfg = model_cfg.get('block_diffusion', {})
        generate_cfg.setdefault('diffusion_steps', bd_cfg.get('diffusion_steps', 128))
        generate_cfg.setdefault('max_length', 100)

    output_path = os.path.join(
        PROJECT_ROOT, cfg['paths']['results_dir'], 'raw', f'benchmark_{model_name}.json',
    )
    run_evaluation(
        model, dataset, conditions, output_path, tokenizer=tokenizer,
        batch_size=eval_cfg['batch_size'],
        num_workers=eval_cfg['num_workers'],
        generate_cfg=generate_cfg,
    )


def mode_analyze_phase2(args, cfg, device):
    from analysis.visualize_phase2 import generate_phase2_figures
    results_dir = os.path.join(PROJECT_ROOT, cfg['paths']['results_dir'])
    generate_phase2_figures(results_dir, os.path.join(results_dir, 'figures'))


def main():
    parser = argparse.ArgumentParser(description='Temporal Misalignment Analysis for SLT (GFSLT-VLP)')
    parser.add_argument('--mode', type=str, required=True, choices=[
        'verify', 'knee_point', 'benchmark', 'train_eval', 'analyze_phase1', 'qualitative',
        'vlp', 'train', 'evaluate', 'analyze_phase2',
    ])
    parser.add_argument(
        '--config', type=str,
        default=os.path.join(PROJECT_ROOT, 'configs', 'config.yaml'),
    )
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (ar_clean, ar_aug, bd_clean, bd_aug)')
    args = parser.parse_args()

    cfg = load_config(args)
    if args.batch_size:
        cfg['evaluation']['batch_size'] = args.batch_size
    set_seed(cfg.get('seed', 42))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    t_start = time.time()

    phase1_modes = {
        'verify': mode_verify, 'knee_point': mode_knee_point, 'benchmark': mode_benchmark,
        'train_eval': mode_train_eval, 'analyze_phase1': mode_analyze_phase1,
        'qualitative': mode_qualitative,
    }
    phase2_modes = {
        'vlp': mode_vlp, 'train': mode_train, 'evaluate': mode_evaluate_phase2,
        'analyze_phase2': mode_analyze_phase2,
    }

    if args.mode in phase1_modes:
        phase1_modes[args.mode](args, cfg, device)
    elif args.mode in phase2_modes:
        if args.mode in ('vlp', 'train', 'evaluate') and not args.model:
            print('ERROR: --model is required for vlp/train/evaluate modes')
            print('  Options: ar_clean, ar_aug, bd_clean, bd_aug')
            sys.exit(1)
        phase2_modes[args.mode](args, cfg, device)

    elapsed = time.time() - t_start
    print(f'Total elapsed time: {elapsed:.1f}s ({elapsed / 60:.1f}m)')


if __name__ == '__main__':
    main()
