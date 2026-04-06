'''Unified training loop for Phase 2 models (AR+Aug, BD Clean, BD+Aug):

- Training with optional misalignment augmentation
- Validation on clean dev set (controlled by eval_every_n_epochs)
- Best checkpoint saving by dev BLEU-4
- Early stopping with patience (counted in eval cycles)
- EMA (Exponential Moving Average) for diffusion training stability
- Linear warmup + cosine annealing scheduler
- WandB logging + print
'''
import os, sys, time, json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'MSKA'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'bd3lms'))

from evaluator import compute_metrics, collect_batch_predictions
from models.ema import ExponentialMovingAverage


def build_optimizer(model, train_cfg):
    '''Build optimizer with per-component learning rates.

    Matches MSKA's optimizer.build_optimizer pattern: iterates over
    named_parameters() and assigns LR based on which config key matches
    the parameter name. This correctly handles nested module hierarchies
    (e.g., SLTModel.mska_model.vl_mapper.xxx).
    '''
    lr_cfg = train_cfg['learning_rate']
    base_lr = lr_cfg.get('default', 1e-5)

    # Group parameters by matching LR config keys against parameter names
    param_groups_dict = defaultdict(list)  # group_name -> list of params
    param_group_lrs = {}  # group_name -> lr

    for param_name, param in model.named_parameters():
        if not param.requires_grad: continue
        matched_key, matched_len = 'default', 0
        
        for key in lr_cfg: # Find the best matching LR key (longest match wins)
            if key == 'default': continue
            if key in param_name and len(key) > matched_len:
                matched_key, matched_len = key, len(key)

        lr = lr_cfg[matched_key] if matched_key != 'default' else base_lr
        param_groups_dict[matched_key].append(param)
        param_group_lrs[matched_key] = lr

    # Build param group list
    param_groups = []
    for group_name in sorted(param_groups_dict.keys()):
        params = param_groups_dict[group_name]
        lr = param_group_lrs.get(group_name, base_lr)
        param_groups.append({'params': params, 'lr': lr, 'name': group_name})

    if not param_groups: raise ValueError('No trainable parameters found when building optimizer.')
    optimizer_name = train_cfg.get('optimizer', 'Adam').lower()
    betas = tuple(train_cfg.get('betas', [0.9, 0.998]))
    weight_decay = train_cfg.get('weight_decay', 0.001)

    if optimizer_name == 'adamw': optimizer = optim.AdamW(param_groups, lr=base_lr, betas=betas, weight_decay=weight_decay)
    else: optimizer = optim.Adam(param_groups, lr=base_lr, betas=betas, weight_decay=weight_decay)

    print(f'Optimizer: {optimizer_name}, base_lr={base_lr}')
    for pg in param_groups: print(f"  {pg['name']}: lr={pg['lr']:.1e}, params={sum(p.numel() for p in pg['params']):,}")
    return optimizer


def build_scheduler(optimizer, train_cfg):
    scheduler_name = train_cfg.get('scheduler', 'cosineannealing').lower()
    t_max = train_cfg.get('t_max', 40)
    warmup_epochs = train_cfg.get('warmup_epochs', 0)

    if scheduler_name == 'cosineannealing':
        if warmup_epochs > 0: # Warmup + cosine: use SequentialLR
            warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
            cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(t_max - warmup_epochs, 1), eta_min=0)
            scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
            print(f'Scheduler: {warmup_epochs}-epoch warmup + cosine annealing (T_max={t_max})')
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0)
            print(f'Scheduler: cosine annealing (T_max={t_max})')
    elif scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        print(f'Scheduler: ReduceLROnPlateau')
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0)
        print(f"Warning: unknown scheduler '{scheduler_name}', using cosine annealing")
    return scheduler


def train_one_epoch(model, dataloader, optimizer, device, epoch, train_cfg, clip_norm=None, ema=None):
    model.train()
    num_batches = 0
    total_loss, total_rec_loss, total_trans_loss = 0.0, 0.0, 0.0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for step, src_input in enumerate(pbar):
        optimizer.zero_grad()
        output = model(src_input)
        loss = output['total_loss']

        # NaN/Inf guard
        if not torch.isfinite(loss):
            print(f'  WARNING: non-finite loss at step {step}, skipping batch')
            optimizer.zero_grad()
            continue

        loss.backward()
        if clip_norm: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        if ema is not None: ema.update(model.parameters())
        loss_val = loss.item()
        rec_loss = output.get('recognition_loss', torch.tensor(0.0)).item()
        trans_loss = output.get('translation_loss', torch.tensor(0.0)).item()

        total_loss += loss_val
        total_rec_loss += rec_loss
        total_trans_loss += trans_loss
        num_batches += 1
        pbar.set_postfix(loss=f'{loss_val:.3f}', rec=f'{rec_loss:.3f}', trans=f'{trans_loss:.3f}')

    avg_loss = total_loss / max(num_batches, 1)
    avg_rec = total_rec_loss / max(num_batches, 1)
    avg_trans = total_trans_loss / max(num_batches, 1)
    print(f'Epoch {epoch}: loss={avg_loss:.4f}, rec={avg_rec:.4f}, trans={avg_trans:.4f}')
    return {'loss': avg_loss, 'rec_loss': avg_rec, 'trans_loss': avg_trans}


@torch.no_grad()
def evaluate_one_epoch(model, dataloader, mska_cfg, generate_cfg=None, beam_size=1, desc='Eval'):
    '''Evaluate model: compute dev loss AND generate translations for BLEU.

    Runs the full forward pass to get both loss values and generation metrics.
    With first_hitting=True and block_size=4, diffusion sampling uses early
    stopping (~4 steps per block instead of 5000), so 5K steps is fast.
    '''
    model.eval()
    sample_results = defaultdict(dict)
    total_loss, total_rec_loss, total_trans_loss, num_batches = 0., 0., 0., 0

    for src_input in tqdm(dataloader, desc=desc):
        output = model(src_input) # Run forward pass to get recognition outputs + transformer_inputs + loss
        collect_batch_predictions(model, src_input, output, sample_results, beam_size=beam_size, generate_cfg=generate_cfg)

        # Accumulate dev loss
        total_loss += output['total_loss'].item()
        total_rec_loss += output.get('recognition_loss', torch.tensor(0.0)).item()
        total_trans_loss += output.get('translation_loss', torch.tensor(0.0)).item()
        num_batches += 1

    # Compute metrics using shared compute_metrics (no duplication)
    eval_results = {k: v for k, v in sample_results.items() if 'txt_hyp' in v}
    if not eval_results:
        return {'wer': 200.0, 'bleu4': 0.0, 'rouge_l': 0.0, 'dev_loss': 0.0, 'dev_rec_loss': 0.0, 'dev_trans_loss': 0.0}
    
    metrics = compute_metrics(eval_results, mska_cfg)
    metrics.pop('per_sample', None)
    metrics['dev_loss'] = total_loss / max(num_batches, 1)
    metrics['dev_rec_loss'] = total_rec_loss / max(num_batches, 1)
    metrics['dev_trans_loss'] = total_trans_loss / max(num_batches, 1)
    return metrics


def train_model(model, train_dataset, dev_dataset, mska_cfg, model_cfg, train_cfg, output_dir, device='cuda', wandb_run=None):
    # Full training loop with validation and checkpointing
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = train_cfg.get('epochs', 40)
    batch_size = train_cfg.get('batch_size', 8)
    num_workers = train_cfg.get('num_workers', 4)
    clip_norm = train_cfg.get('gradient_clip_norm', None)
    patience = train_cfg.get('early_stopping_patience', 10)
    save_every = train_cfg.get('save_every_n_epochs', 5)
    eval_every = train_cfg.get('eval_every_n_epochs', 1)

    # Generation config for validation
    generate_cfg = model_cfg.get('validation', {}).get('translation', {'length_penalty': 1, 'max_length': 100, 'num_beams': 5})
    if model_cfg.get('decoder_type') == 'bd':
        bd_cfg = model_cfg.get('block_diffusion', {})
        # bd3lms Section C.5: 5K diffusion steps (early stopping makes this fast)
        generate_cfg['diffusion_steps'] = bd_cfg.get('diffusion_steps', 5000)
    beam_size = model_cfg.get('validation', {}).get('recognition', {}).get('beam_size', 1)

    # Build dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=train_dataset.collate_fn,
        pin_memory=True, drop_last=True)
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=dev_dataset.collate_fn,
        pin_memory=True)

    model.to(device)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    # EMA for diffusion training stability
    ema_decay = train_cfg.get('ema', 0.0)
    ema = None
    if ema_decay > 0:
        ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
        ema.move_shadow_params_to_device(device)
        print(f'EMA enabled: decay={ema_decay}')

    # Resume from checkpoint
    start_epoch, best_bleu4 = 0, 0.0
    resume_from = train_cfg.get('resume_from')
    if resume_from and os.path.exists(resume_from):
        print(f'Resuming from {resume_from}')
        ckpt = torch.load(resume_from, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=False)
        if 'optimizer' in ckpt: optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt and ckpt['scheduler'] is not None: scheduler.load_state_dict(ckpt['scheduler'])
        if 'ema' in ckpt and ckpt['ema'] is not None and ema is not None:
            ema.load_state_dict(ckpt['ema'])
            ema.move_shadow_params_to_device(device)
        start_epoch = ckpt.get('epoch', 0) + 1
        best_bleu4 = ckpt.get('best_bleu4', 0.0)

    # Patience is counted in eval cycles, not epochs
    eval_cycles_without_improvement = 0
    log_file = output_dir / 'training_log.jsonl'

    print(f'\nStarting training: {epochs} epochs, batch_size={batch_size}, '
          f'eval_every={eval_every} epochs, patience={patience} eval cycles')
    print(f'Output: {output_dir}\n')

    for epoch in range(start_epoch, epochs):
        t_start = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, device, epoch, train_cfg, clip_norm=clip_norm, ema=ema)

        # Step scheduler (epoch-level)
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau): pass  # stepped after eval
        else: scheduler.step()

        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt_data = {
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'epoch': epoch, 'best_bleu4': best_bleu4,
            }
            if ema is not None: ckpt_data['ema'] = ema.state_dict()
            torch.save(ckpt_data, output_dir / 'checkpoint.pth')

        log_entry = {
            'epoch': epoch,
            'train_loss': train_stats['loss'],
            'train_rec_loss': train_stats['rec_loss'],
            'train_trans_loss': train_stats['trans_loss'],
            'lr': optimizer.param_groups[0]['lr'],
        }
        do_eval = (epoch + 1) % eval_every == 0 # Eval at epoch eval_every-1, 2*eval_every-1, ...
        if do_eval:
            if ema is not None: # EMA: swap to shadow params for evaluation
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
            
            dev_metrics = evaluate_one_epoch( # Generate translations and compute BLEU (no redundant loss computation)
                model, dev_loader, mska_cfg, generate_cfg=generate_cfg,
                beam_size=beam_size, desc=f'Dev (epoch {epoch})'
            )
            if ema is not None: ema.restore(model.parameters()) # EMA: restore original params
            
            elapsed = time.time() - t_start
            print(f"  Dev loss={dev_metrics.get('dev_loss', 0):.4f} "
                  f"(rec={dev_metrics.get('dev_rec_loss', 0):.4f}, "
                  f"trans={dev_metrics.get('dev_trans_loss', 0):.4f})")
            print(f"  Dev metrics: WER={dev_metrics['wer']:.2f}, "
                  f"BLEU-4={dev_metrics['bleu4']:.2f}, ROUGE-L={dev_metrics['rouge_l']:.2f} ({elapsed:.0f}s)")

            log_entry.update({
                'dev_loss': dev_metrics.get('dev_loss', 0),
                'dev_rec_loss': dev_metrics.get('dev_rec_loss', 0),
                'dev_trans_loss': dev_metrics.get('dev_trans_loss', 0),
                'dev_wer': dev_metrics['wer'],
                'dev_bleu4': dev_metrics['bleu4'],
                'dev_rouge_l': dev_metrics['rouge_l'],
                'elapsed': elapsed,
            })
            wandb_run.log({
                'epoch': epoch,
                'train/loss': train_stats['loss'],
                'train/rec_loss': train_stats['rec_loss'],
                'train/trans_loss': train_stats['trans_loss'],
                'dev/loss': dev_metrics.get('dev_loss', 0),
                'dev/rec_loss': dev_metrics.get('dev_rec_loss', 0),
                'dev/trans_loss': dev_metrics.get('dev_trans_loss', 0),
                'dev/wer': dev_metrics['wer'],
                'dev/bleu4': dev_metrics['bleu4'],
                'dev/rouge_l': dev_metrics['rouge_l'],
                'lr': optimizer.param_groups[0]['lr'],
            })
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau): scheduler.step(dev_metrics['bleu4'])
            if dev_metrics['bleu4'] > best_bleu4: # Best checkpoint
                best_bleu4 = dev_metrics['bleu4']
                eval_cycles_without_improvement = 0
                best_ckpt = {
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'epoch': epoch, 'best_bleu4': best_bleu4,
                }
                if ema is not None:
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    best_ckpt['model'] = model.state_dict()
                    ema.restore(model.parameters())
                    best_ckpt['ema'] = ema.state_dict()
                torch.save(best_ckpt, output_dir / 'best_checkpoint.pth')
                print(f'  ★ New best BLEU-4: {best_bleu4:.2f} (saved)')
            else:
                eval_cycles_without_improvement += 1
                print(f'  No improvement for {eval_cycles_without_improvement}/{patience} eval cycles (best={best_bleu4:.2f})')
        else:
            elapsed = time.time() - t_start
            log_entry['elapsed'] = elapsed
            wandb_run.log({
                'epoch': epoch,
                'train/loss': train_stats['loss'],
                'train/rec_loss': train_stats['rec_loss'],
                'train/trans_loss': train_stats['trans_loss'],
                'lr': optimizer.param_groups[0]['lr'],
            })
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Early stopping (patience counted in eval cycles)
        if do_eval and eval_cycles_without_improvement >= patience:
            print(f'Early stopping at epoch {epoch} ({eval_cycles_without_improvement} eval cycles without improvement)')
            break

    print(f'\nTraining complete. Best dev BLEU-4: {best_bleu4:.2f}')
    return best_bleu4