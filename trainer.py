'''Training for GFSLT-VLP models, matching the original paper.

Stage 1 (VLP): SLRCLIP contrastive + optional MLM via TextDecoder.
    - Main optimizer: AdamW, lr=5e-4, wd=1e-4, betas=(0.9, 0.98).
    - TextDecoder optimizer (separate, VLP-v2): AdamW, lr=1e-3, wd=0, betas=(0.9, 0.98),
      stepped every 5 iterations.
    - Scheduler: CosineAnnealingLR(T_max=epochs, eta_min=0). No warmup, no grad clip.
    - Label smoothing = 0.2 on MLM CE.

Stage 2 (Translation): External CrossEntropyLoss(label_smoothing=0.2, ignore_index=1)
on GFSLT logits.
    - Optimizer: SGD lr=0.01 momentum=0.9 wd=0 (paper default). Configurable via
      train_cfg.optimizer / learning_rate.{default, ...}.
    - Scheduler: CosineAnnealingLR(T_max=epochs, eta_min=1e-8). No warmup, no grad clip.
    - Per-component learning rates supported via substring matching on parameter names.
    - EMA optional (used for diffusion training stability).
'''
import os
import sys
import json
import time
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GFSLT_DIR = os.path.join(PROJECT_ROOT, 'gfslt-pose-trainer')
if GFSLT_DIR not in sys.path:
    sys.path.insert(0, GFSLT_DIR)

from loader import collate_fn
from models import SLRCLIP, TextDecoder
from evaluator import collect_batch_predictions, compute_metrics


PAD_IDX = 1  # mBART <pad>
LABEL_SMOOTHING = 0.2


# ─── EMA ─────────────────────────────────────────────────────────────────────
class ExponentialMovingAverage:
    '''Maintains exponential moving average of a set of parameters.'''
    def __init__(self, parameters, decay, use_num_updates=True):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

    def move_shadow_params_to_device(self, device):
        self.shadow_params = [i.to(device) for i in self.shadow_params]

    def update(self, parameters):
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates, shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']


# ─── Stage 1: VLP Pretraining ────────────────────────────────────────────────

def train_vlp(
    gfslt_config, tokenizer, train_dataset, output_dir,
    vlp_cfg=None, device='cuda', wandb_run=None,
):
    '''Stage 1: Visual-Language Pretraining (SLRCLIP + optional MLM).

    Matches GFSLT-VLP v2 paper:
      - Main (SLRCLIP): AdamW(lr=5e-4, wd=1e-4, betas=(0.9, 0.98)).
      - TextDecoder: separate AdamW(lr=1e-3, wd=0, betas=(0.9, 0.98)),
                     stepped every 5 iterations.
      - CosineAnnealingLR(T_max=epochs, eta_min=0) on both.
      - No warmup, no grad clip. MLM CE uses label_smoothing=0.2.
    '''
    vlp_cfg = vlp_cfg or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = vlp_cfg.get('epochs', 80)
    batch_size = vlp_cfg.get('batch_size', 32)
    num_workers = vlp_cfg.get('num_workers', 4)

    lr = vlp_cfg.get('learning_rate', 5.0e-4)
    weight_decay = vlp_cfg.get('weight_decay', 1.0e-4)
    betas = tuple(vlp_cfg.get('betas', [0.9, 0.98]))

    use_text_decoder = vlp_cfg.get('use_text_decoder', True)
    mlm_loss_weight = vlp_cfg.get('mlm_loss_weight', 1.0)
    td_lr = vlp_cfg.get('text_decoder_lr', 1.0e-3)
    td_weight_decay = vlp_cfg.get('text_decoder_weight_decay', 0.0)
    td_step_every = vlp_cfg.get('text_decoder_step_every', 5)

    # Build models
    slrclip = SLRCLIP(gfslt_config).to(device)
    text_decoder = TextDecoder(gfslt_config).to(device) if use_text_decoder else None
    pad_token_id = tokenizer.pad_token_id

    # Optimizers
    main_optimizer = optim.AdamW(
        slrclip.parameters(), lr=lr, weight_decay=weight_decay, betas=betas,
    )
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        main_optimizer, T_max=epochs, eta_min=0,
    )
    td_optimizer, td_scheduler = None, None
    if text_decoder is not None:
        td_optimizer = optim.AdamW(
            text_decoder.parameters(), lr=td_lr,
            weight_decay=td_weight_decay, betas=betas,
        )
        td_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            td_optimizer, T_max=epochs, eta_min=0,
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )

    n_main = sum(p.numel() for p in slrclip.parameters() if p.requires_grad)
    n_td = sum(p.numel() for p in text_decoder.parameters() if p.requires_grad) if text_decoder else 0
    print(f'\nVLP Stage 1: SLRCLIP={n_main / 1e6:.2f}M, TextDecoder={n_td / 1e6:.2f}M')
    print(f'  epochs={epochs}, batch_size={batch_size}, lr={lr}, td_lr={td_lr}, '
          f'td_step_every={td_step_every}\n')

    global_step = 0
    for epoch in range(epochs):
        slrclip.train()
        if text_decoder is not None:
            text_decoder.train()
        total_clip_loss, total_mlm_loss, num_batches = 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f'VLP Epoch {epoch + 1}/{epochs}')

        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            pixel_mask = batch['pixel_mask'].to(device)
            labels_list = batch['labels']

            paragraph_tokens = torch.stack([l['paragraph_tokens'] for l in labels_list]).to(device)
            paragraph_attention_mask = (paragraph_tokens != pad_token_id).long()

            # CLIP contrastive forward
            outputs = slrclip(
                pixel_values=pixel_values, pixel_mask=pixel_mask,
                paragraph_tokens=paragraph_tokens,
                paragraph_attention_mask=paragraph_attention_mask,
            )
            clip_loss = outputs['loss']
            loss = clip_loss
            mlm_loss_val = 0.0

            # MLM loss (re-encode masked text via slrclip.model_txt, then decode clean)
            if text_decoder is not None:
                masked_tokens = torch.stack(
                    [l['masked_paragraph_tokens'] for l in labels_list]
                ).to(device)
                masked_attention_mask = (masked_tokens != pad_token_id).long()
                _, enc_hidden = slrclip.model_txt(masked_tokens, masked_attention_mask)
                lm_logits = text_decoder(
                    input_ids=paragraph_tokens,
                    attention_mask=paragraph_attention_mask,
                    encoder_hidden_states=enc_hidden,
                    encoder_attention_mask=masked_attention_mask,
                )
                mlm_loss = F.cross_entropy(
                    lm_logits.view(-1, lm_logits.size(-1)),
                    paragraph_tokens.view(-1),
                    ignore_index=pad_token_id,
                    label_smoothing=LABEL_SMOOTHING,
                )
                loss = loss + mlm_loss_weight * mlm_loss
                mlm_loss_val = mlm_loss.item()

            if not torch.isfinite(loss):
                main_optimizer.zero_grad()
                if td_optimizer is not None:
                    td_optimizer.zero_grad()
                continue

            main_optimizer.zero_grad()
            if td_optimizer is not None:
                td_optimizer.zero_grad()
            loss.backward()
            main_optimizer.step()

            # TextDecoder update every N iterations (VLP-v2)
            if td_optimizer is not None and ((global_step + 1) % td_step_every == 0):
                td_optimizer.step()

            total_clip_loss += clip_loss.item()
            total_mlm_loss += mlm_loss_val
            num_batches += 1
            global_step += 1
            pbar.set_postfix(clip=f'{clip_loss.item():.3f}', mlm=f'{mlm_loss_val:.3f}')

        main_scheduler.step()
        if td_scheduler is not None:
            td_scheduler.step()

        avg_clip = total_clip_loss / max(num_batches, 1)
        avg_mlm = total_mlm_loss / max(num_batches, 1)
        print(f'  VLP Epoch {epoch + 1}: clip_loss={avg_clip:.4f}, mlm_loss={avg_mlm:.4f}')
        if wandb_run:
            wandb_run.log({
                'vlp/epoch': epoch, 'vlp/clip_loss': avg_clip,
                'vlp/mlm_loss': avg_mlm,
                'vlp/lr_main': main_scheduler.get_last_lr()[0],
                'vlp/lr_td': td_scheduler.get_last_lr()[0] if td_scheduler is not None else 0.0,
            })

    # Save VLP checkpoint (SLRCLIP state_dict; wrap with base_module. for legacy).
    wrapped_state = {'base_module.' + k: v for k, v in slrclip.state_dict().items()}
    ckpt = {'model': wrapped_state, 'epoch': epochs}
    torch.save(ckpt, output_dir / 'vlp_checkpoint.pth')
    print(f'VLP checkpoint saved to {output_dir / "vlp_checkpoint.pth"}')
    return slrclip


# ─── Stage 2: Translation Training ───────────────────────────────────────────

def build_optimizer(model, train_cfg):
    '''Build optimizer with per-component learning rates (substring matching).

    Paper default: SGD lr=0.01 momentum=0.9 weight_decay=0.
    '''
    lr_cfg = train_cfg.get('learning_rate', {'default': 0.01})
    if not isinstance(lr_cfg, dict):
        lr_cfg = {'default': float(lr_cfg)}
    base_lr = lr_cfg.get('default', 0.01)

    groups_dict = defaultdict(list)
    group_lrs = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        matched_key, matched_len = 'default', 0
        for key in lr_cfg:
            if key == 'default':
                continue
            if key in name and len(key) > matched_len:
                matched_key, matched_len = key, len(key)
        lr = lr_cfg[matched_key] if matched_key != 'default' else base_lr
        groups_dict[matched_key].append(param)
        group_lrs[matched_key] = lr

    param_groups = []
    for gname in sorted(groups_dict.keys()):
        param_groups.append({
            'params': groups_dict[gname],
            'lr': group_lrs.get(gname, base_lr),
            'name': gname,
        })
    if not param_groups:
        raise ValueError('No trainable parameters found.')

    optimizer_name = train_cfg.get('optimizer', 'SGD').lower()
    weight_decay = train_cfg.get('weight_decay', 0.0)

    if optimizer_name == 'sgd':
        momentum = train_cfg.get('momentum', 0.9)
        optimizer = optim.SGD(
            param_groups, lr=base_lr, momentum=momentum, weight_decay=weight_decay,
        )
    elif optimizer_name == 'adamw':
        betas = tuple(train_cfg.get('betas', [0.9, 0.98]))
        optimizer = optim.AdamW(param_groups, lr=base_lr, betas=betas, weight_decay=weight_decay)
    else:
        betas = tuple(train_cfg.get('betas', [0.9, 0.999]))
        optimizer = optim.Adam(param_groups, lr=base_lr, betas=betas, weight_decay=weight_decay)

    print(f'Optimizer: {optimizer_name}, base_lr={base_lr}, wd={weight_decay}')
    for pg in param_groups:
        n = sum(p.numel() for p in pg['params'])
        print(f"  {pg['name']}: lr={pg['lr']:.1e}, params={n:,}")
    return optimizer


def build_scheduler(optimizer, train_cfg):
    '''Cosine annealing with eta_min=1e-8 (paper default). No warmup.'''
    epochs = train_cfg.get('t_max', train_cfg.get('epochs', 200))
    eta_min = train_cfg.get('eta_min', 1.0e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=eta_min,
    )
    print(f'Scheduler: CosineAnnealingLR (T_max={epochs}, eta_min={eta_min})')
    return scheduler


def _compute_loss_from_logits(logits, labels, criterion):
    '''Compute CE loss with label smoothing on the flattened token axis.'''
    V = logits.size(-1)
    return criterion(logits.reshape(-1, V), labels.reshape(-1))


def train_one_epoch(model, dataloader, optimizer, device, epoch,
                    train_cfg, criterion, ema=None):
    '''Train one epoch of Stage 2 translation. External CE loss with label smoothing.'''
    model.train()
    num_batches = 0
    total_loss = 0.0
    pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{train_cfg.get('epochs', 200)}")
    current_lr = optimizer.param_groups[0]['lr']

    for step, batch in enumerate(dataloader):
        pixel_values = batch['pixel_values'].to(device)
        pixel_mask = batch['pixel_mask'].to(device)
        labels_list = batch['labels']

        labels = torch.stack([l['paragraph_tokens'] for l in labels_list]).to(device)
        labels_mask = (labels != PAD_IDX).long()

        optimizer.zero_grad()
        output = model(
            pixel_values=pixel_values, pixel_mask=pixel_mask,
            paragraph_tokens=labels, paragraph_attention_mask=labels_mask,
            # Also provide labels/labels_mask for decoders that compute loss internally (BD).
            labels=labels, labels_mask=labels_mask,
        )

        if 'loss' in output and output['loss'] is not None:
            # BD decoder: loss computed internally (MDLM).
            loss = output['loss']
        else:
            # AR decoder (GFSLT): compute external CE with label_smoothing=0.2.
            loss = _compute_loss_from_logits(output['logits'], labels, criterion)

        if not torch.isfinite(loss):
            print(f'  WARNING: non-finite loss at step {step}, skipping batch')
            optimizer.zero_grad()
            continue

        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model.parameters())

        loss_val = loss.item()
        total_loss += loss_val
        num_batches += 1

        pbar.set_postfix(loss=f'{loss_val:.3f}', lr=f'{current_lr:.2e}')
        pbar.update(1)

    avg_loss = total_loss / max(num_batches, 1)
    pbar.set_postfix(loss=f'{avg_loss:.4f}', lr=f'{current_lr:.2e}')
    pbar.close()
    return {'loss': avg_loss}


@torch.no_grad()
def evaluate_one_epoch(model, dataloader, tokenizer, criterion,
                       generate_cfg=None, desc='Eval'):
    '''Evaluate: compute dev loss and generate translations for BLEU/ROUGE.'''
    model.eval()
    device = next(model.parameters()).device
    sample_results = {}
    total_loss, num_batches = 0.0, 0

    for batch in tqdm(dataloader, desc=desc):
        pixel_values = batch['pixel_values'].to(device)
        pixel_mask = batch['pixel_mask'].to(device)
        labels_list = batch['labels']
        names = batch['names']
        texts = batch['texts']

        labels = torch.stack([l['paragraph_tokens'] for l in labels_list]).to(device)
        labels_mask = (labels != PAD_IDX).long()

        output = model(
            pixel_values=pixel_values, pixel_mask=pixel_mask,
            paragraph_tokens=labels, paragraph_attention_mask=labels_mask,
            labels=labels, labels_mask=labels_mask,
        )
        if 'loss' in output and output['loss'] is not None:
            total_loss += output['loss'].item()
        else:
            total_loss += _compute_loss_from_logits(output['logits'], labels, criterion).item()
        num_batches += 1

        collect_batch_predictions(
            model, names, texts, pixel_values, pixel_mask,
            sample_results, tokenizer, generate_cfg=generate_cfg,
        )

    if not sample_results:
        return {'bleu4': 0.0, 'rouge_l': 0.0, 'dev_loss': 0.0}

    metrics = compute_metrics(sample_results)
    metrics.pop('per_sample', None)
    metrics['dev_loss'] = total_loss / max(num_batches, 1)
    return metrics


def train_model(
    model, train_dataset, dev_dataset, model_cfg, train_cfg,
    output_dir, tokenizer, device='cuda', wandb_run=None,
):
    '''Full Stage 2 training loop with validation, EMA, best-ckpt-by-BLEU4,
    early stopping, WandB + print logging.
    '''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = train_cfg.get('epochs', 200)
    batch_size = train_cfg.get('batch_size', 16)
    num_workers = train_cfg.get('num_workers', 4)
    patience = train_cfg.get('early_stopping_patience', 10)
    save_every = train_cfg.get('save_every_n_epochs', 5)
    eval_every = train_cfg.get('eval_every_n_epochs', 1)

    # Generation config for validation (paper: max_new_tokens=150, num_beams=4).
    generate_cfg = dict(model_cfg.get('validation', {}).get('translation', {
        'max_new_tokens': 150, 'num_beams': 4, 'length_penalty': 1.0,
    }))
    if model_cfg.get('decoder_type') == 'bd':
        bd_cfg = model_cfg.get('block_diffusion', {})
        generate_cfg.setdefault('diffusion_steps', bd_cfg.get('diffusion_steps', 128))
        generate_cfg.setdefault('max_length', 100)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True,
    )

    model.to(device)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    # Criterion (external CE for AR models).
    label_smoothing = train_cfg.get('label_smoothing', LABEL_SMOOTHING)
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX, label_smoothing=label_smoothing,
    )

    # EMA (used for diffusion training).
    ema_decay = train_cfg.get('ema', 0.0)
    ema = None
    if ema_decay and ema_decay > 0:
        ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
        ema.move_shadow_params_to_device(device)
        print(f'EMA enabled: decay={ema_decay}')

    start_epoch, best_bleu4 = 0, 0.0
    resume_from = train_cfg.get('resume_from')
    if resume_from and os.path.exists(resume_from):
        print(f'Resuming from {resume_from}')
        ckpt = torch.load(resume_from, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)
        best_bleu4 = ckpt.get('best_bleu4', 0.0)

    eval_cycles_no_improve = 0
    log_file = output_dir / 'training_log.jsonl'

    print(f'\nStage 2 training: {epochs} epochs, batch_size={batch_size}, '
          f'eval_every={eval_every}, patience={patience} eval cycles')
    print(f'Output: {output_dir}, label_smoothing={label_smoothing}\n')

    for epoch in range(start_epoch, epochs):
        t_start = time.time()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, epoch, train_cfg,
            criterion=criterion, ema=ema,
        )
        scheduler.step()

        if (epoch + 1) % save_every == 0:
            ckpt_data = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch, 'best_bleu4': best_bleu4,
            }
            if ema is not None:
                ckpt_data['ema'] = ema.state_dict()
            torch.save(ckpt_data, output_dir / 'checkpoint.pth')

        log_entry = {
            'epoch': epoch,
            'train_loss': train_stats['loss'],
            'lr': optimizer.param_groups[0]['lr'],
        }

        do_eval = (epoch + 1) % eval_every == 0
        if do_eval:
            if ema is not None:
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
            dev_metrics = evaluate_one_epoch(
                model, dev_loader, tokenizer, criterion,
                generate_cfg=generate_cfg, desc=f'Dev (epoch {epoch + 1})',
            )
            if ema is not None:
                ema.restore(model.parameters())

            elapsed = time.time() - t_start
            print(f"  Dev loss={dev_metrics.get('dev_loss', 0):.4f}")
            print(f"  Dev metrics: BLEU-4={dev_metrics['bleu4']:.2f}, "
                  f"ROUGE-L={dev_metrics['rouge_l']:.2f} ({elapsed:.0f}s)")

            log_entry.update({
                'dev_loss': dev_metrics.get('dev_loss', 0),
                'dev_bleu4': dev_metrics['bleu4'],
                'dev_rouge_l': dev_metrics['rouge_l'],
                'elapsed': elapsed,
            })
            if wandb_run:
                wandb_run.log({
                    'epoch': epoch,
                    'train/loss': train_stats['loss'],
                    'dev/loss': dev_metrics.get('dev_loss', 0),
                    'dev/bleu4': dev_metrics['bleu4'],
                    'dev/rouge_l': dev_metrics['rouge_l'],
                    'lr': optimizer.param_groups[0]['lr'],
                })

            if dev_metrics['bleu4'] > best_bleu4:
                best_bleu4 = dev_metrics['bleu4']
                eval_cycles_no_improve = 0
                best_ckpt = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch, 'best_bleu4': best_bleu4,
                }
                if ema is not None:
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    best_ckpt['model'] = model.state_dict()
                    ema.restore(model.parameters())
                    best_ckpt['ema'] = ema.state_dict()
                torch.save(best_ckpt, output_dir / 'best_checkpoint.pth')
                print(f'  New best BLEU-4: {best_bleu4:.2f} (saved)')
            else:
                eval_cycles_no_improve += 1
                print(f'  No improvement for {eval_cycles_no_improve}/{patience} eval cycles '
                      f'(best={best_bleu4:.2f})')
        else:
            elapsed = time.time() - t_start
            log_entry['elapsed'] = elapsed
            if wandb_run:
                wandb_run.log({
                    'epoch': epoch,
                    'train/loss': train_stats['loss'],
                    'lr': optimizer.param_groups[0]['lr'],
                })

        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        if do_eval and eval_cycles_no_improve >= patience:
            print(f'Early stopping at epoch {epoch} '
                  f'({eval_cycles_no_improve} eval cycles without improvement)')
            break

    print(f'\nTraining complete. Best dev BLEU-4: {best_bleu4:.2f}')
    return best_bleu4
