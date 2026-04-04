'''Unified training loop for Phase 2 models (AR+Aug, BD Clean, BD+Aug):

- Training with optional misalignment augmentation
- Validation on clean dev set (controlled by eval_every_n_epochs)
- Combined dev loss + metrics computation in a single dataloader pass
- Best checkpoint saving by dev BLEU-4
- Early stopping with patience (counted in eval cycles)
- WandB logging + print
'''
import os, sys, time, json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import numpy as np
import wandb
import torch
from torch import optim
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'MSKA'))

# Reuse existing functions (no duplication)
from data.misalign import generate_conditions
from evaluator import compute_metrics

# ── 9 misalignment groups for dev tracking ───────────────────────────────────
TRACKING_SEVERITIES = [0.05, 0.10, 0.20]
BASIC_GROUPS = ['HT', 'TT', 'HC', 'TC']
COMPOUND_GROUPS = ['HT+TT', 'HC+TC', 'HT+TC', 'HC+TT']


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
    if scheduler_name == 'cosineannealing':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0)
    elif scheduler_name == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    print(f"  Warning: unknown scheduler '{scheduler_name}', using cosine annealing")
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0)


def train_one_epoch(model, dataloader, optimizer, device, epoch, train_cfg, clip_norm=None):
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
def evaluate_with_loss(model, dataloader, mska_cfg, generate_cfg=None, beam_size=1, dataset_name='phoenix-2014t', desc='Eval'):
    '''Evaluate model: compute dev loss AND translation metrics in ONE pass.

    Returns:
        metrics: dict with 'wer', 'bleu4', 'rouge_l', etc.
        loss_stats: dict with 'loss', 'rec_loss', 'trans_loss'
    '''
    model.eval()
    num_batches = 0
    total_loss, total_rec_loss, total_trans_loss = 0.0, 0.0, 0.0
    tokenizer = model.gloss_tokenizer
    sample_results = defaultdict(dict)
    
    for src_input in tqdm(dataloader, desc=desc):
        output = model(src_input)
        batch_names = src_input['name']

        # ── Accumulate dev loss ──
        loss = output['total_loss']
        if torch.isfinite(loss):
            total_loss += loss.item()
            total_rec_loss += output.get('recognition_loss', torch.tensor(0.0)).item()
            total_trans_loss += output.get('translation_loss', torch.tensor(0.0)).item()
            num_batches += 1

        # ── Collect predictions for metrics ──
        # Recognition (CTC gloss decoding)
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

        # Translation
        gen_output = model.generate_txt(transformer_inputs=output['transformer_inputs'], generate_cfg=generate_cfg)
        for name, txt_hyp, txt_ref in zip(batch_names, gen_output['decoded_sequences'], src_input['text']):
            if name not in sample_results: sample_results[name] = {}
            sample_results[name]['txt_hyp'] = txt_hyp
            sample_results[name]['txt_ref'] = txt_ref

    # Loss stats
    if num_batches == 0: loss_stats = {'loss': float('inf'), 'rec_loss': 0.0, 'trans_loss': 0.0}
    else: loss_stats = {
        'loss': total_loss / num_batches,
        'rec_loss': total_rec_loss / num_batches,
        'trans_loss': total_trans_loss / num_batches,
    }
    # Compute metrics using shared compute_metrics (no duplication)
    eval_results = {k: v for k, v in sample_results.items() if 'txt_hyp' in v}
    if not eval_results: metrics = {'wer': 200.0, 'bleu4': 0.0, 'rouge_l': 0.0}
    else:
        metrics = compute_metrics(eval_results, mska_cfg)
        metrics.pop('per_sample', None)
    return metrics, loss_stats


def train_model(model, train_dataset, dev_dataset, mska_cfg, model_cfg, train_cfg, output_dir, device='cuda', wandb_run=None):
    # Full training loop with validation and checkpointing
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = train_cfg.get('epochs', 40)
    batch_size = train_cfg.get('batch_size', 8)
    num_workers = 4
    
    clip_norm = train_cfg.get('gradient_clip_norm', None)
    patience = train_cfg.get('early_stopping_patience', 10)
    save_every = train_cfg.get('save_every_n_epochs', 5)
    eval_every = train_cfg.get('eval_every_n_epochs', 1)
    dataset_name = mska_cfg['data'].get('dataset_name', 'Phoenix-2014T')

    generate_cfg = model_cfg.get('validation', {}).get('translation', {'length_penalty': 1, 'max_length': 100, 'num_beams': 5})
    if model_cfg.get('decoder_type') == 'bd':
        bd_cfg = model_cfg.get('block_diffusion', {})
        generate_cfg['diffusion_steps'] = bd_cfg.get('diffusion_steps', 10)
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

    # Resume from checkpoint if specified
    start_epoch, best_bleu4 = 0, 0.0
    if 'resume_from' in train_cfg and train_cfg['resume_from']:
        ckpt_path = train_cfg['resume_from']
        if os.path.exists(ckpt_path):
            print(f'Resuming from {ckpt_path}')
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt['model'], strict=False)
            if 'optimizer' in ckpt: optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt and ckpt['scheduler'] is not None: scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_bleu4 = ckpt.get('best_bleu4', 0.0)

    # Patience is counted in eval cycles, not epochs
    eval_cycles_without_improvement = 0
    log_file = output_dir / 'training_log.jsonl'

    print(f'\nStarting training: {epochs} epochs, batch_size={batch_size}, '
          f'eval_every={eval_every} (eval at epochs: {eval_every}, {2*eval_every+1}, {3*eval_every+2}, ...)')
    print(f'Output: {output_dir}\n')

    for epoch in range(start_epoch, epochs):
        t_start = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, device, epoch, train_cfg, clip_norm=clip_norm)
        scheduler.step()
        if (epoch + 1) % save_every == 0: # Save periodic checkpoint
            ckpt_path = output_dir / 'checkpoint.pth'
            torch.save({
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'epoch': epoch, 'best_bleu4': best_bleu4,
            }, ckpt_path)

        log_entry = { # Logging entry (always)
            'epoch': epoch,
            'train_loss': train_stats['loss'],
            'train_rec_loss': train_stats['rec_loss'],
            'train_trans_loss': train_stats['trans_loss'],
            'lr': optimizer.param_groups[0]['lr'],
        }
        do_eval = epoch % (eval_every + 1) == eval_every # Determine if this is an eval epoch
        if do_eval:
            # ── Combined dev loss + metrics in ONE dataloader pass ──
            dev_metrics, dev_loss_stats = evaluate_with_loss(
                model, dev_loader, mska_cfg, generate_cfg=generate_cfg,
                beam_size=beam_size, dataset_name=dataset_name,
                desc=f'Dev (epoch {epoch})')

            elapsed = time.time() - t_start
            print(f"  Dev loss: total={dev_loss_stats['loss']:.4f}, "
                  f"rec={dev_loss_stats['rec_loss']:.4f}, trans={dev_loss_stats['trans_loss']:.4f}")
            print(f"  Dev metrics: WER={dev_metrics['wer']:.2f}, "
                  f"BLEU-4={dev_metrics['bleu4']:.2f}, ROUGE-L={dev_metrics['rouge_l']:.2f} ({elapsed:.0f}s)")
            log_entry.update({
                'dev_loss': dev_loss_stats['loss'],
                'dev_rec_loss': dev_loss_stats['rec_loss'],
                'dev_trans_loss': dev_loss_stats['trans_loss'],
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
                'dev/loss': dev_loss_stats['loss'],
                'dev/wer': dev_metrics['wer'],
                'dev/bleu4': dev_metrics['bleu4'],
                'dev/rouge_l': dev_metrics['rouge_l'],
                'lr': optimizer.param_groups[0]['lr'],
            })
            if dev_metrics['bleu4'] > best_bleu4: # Best checkpoint
                best_bleu4 = dev_metrics['bleu4']
                eval_cycles_without_improvement = 0
                best_path = output_dir / 'best_checkpoint.pth'
                torch.save({
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'epoch': epoch, 'best_bleu4': best_bleu4,
                }, best_path)
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