'''Block Diffusion Decoder for conditional sign language translation.

Adapts BD3LMs' discrete diffusion framework for *conditional* text generation:
  - Subclasses DDiTBlock to inherit self-attention, block-causal mask handling,
    KV cache, and rotary embedding logic from the bd3lms repository.
  - Inserts a CrossAttention sublayer to attend to MSKA visual encoder features.
  - Provides forward() and generate() matching TranslationNetwork's interface.

All diffusion logic (noise schedule, loss, sampling) follows bd3lms/diffusion.py
with minimal adaptation.  See inline references to bd3lms source lines.
'''
import os, sys
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

# ── Import building blocks from bd3lms repo ──────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'bd3lms'))

from noise_schedule import LogLinearNoise
from models.dit import (
    DDiTBlock, LayerNorm, TimestepEmbedder, EmbeddingLayer, 
    DDiTFinalLayer, Rotary, block_diff_mask, modulate_fused
)

def _sample_categorical(categorical_probs): # Gumbel-max trick for sampling from categorical distribution
    # Ref: bd3lms/diffusion.py _sample_categorical
    gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


class CrossAttention(nn.Module):
    '''Multi-head cross-attention: text queries attend to visual encoder features.

    Output projection is zero-initialized so cross-attention starts as identity,
    matching the adaLN zero-init philosophy in DiT.
    '''
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(dim)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x, encoder_out, encoder_mask=None):
        '''
        Args:
            x: (B, S, D) decoder hidden states
            encoder_out: (B, T_v, D) visual features
            encoder_mask: (B, T_v) boolean mask (True = valid, False = padding)
        '''
        residual = x
        x = self.norm(x)
        B, S, D = x.shape
        H = self.n_heads

        q = self.q_proj(x).view(B, S, H, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(encoder_out).view(B, -1, 2, H, self.head_dim)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        attn_mask = encoder_mask[:, None, None, :].bool() if encoder_mask is not None else None
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return residual + self.dropout(self.out_proj(out))


class ConditionalDDiTBlock(DDiTBlock):
    '''DDiTBlock subclass that adds cross-attention to visual features.

    Overrides attn_mlp() to insert cross-attention between self-attention residual and MLP.
    All other logic (QKV, rotary, block-causal mask, KV cache) is inherited from DDiTBlock.

    Encoder context is set via _encoder_out/_encoder_mask attributes
    before calling forward() to avoid changing DDiTBlock's signature.
    '''
    def __init__(self, n, dim, n_heads, cond_dim, block_size, mlp_ratio=4, dropout=0.1, max_seqlen=1024, attn_backend='sdpa'):
        super().__init__(
            n=n, dim=dim, n_heads=n_heads, adaLN=True, cond_dim=cond_dim, block_size=block_size,
            mlp_ratio=mlp_ratio, dropout=dropout, max_seqlen=max_seqlen, attn_backend=attn_backend
        )
        self.cross_attn_visual = CrossAttention(dim, n_heads, dropout=dropout) # Cross-attention to visual features (NEW)
        self._encoder_out = None
        self._encoder_mask = None


    def attn_mlp(self, x, c, gate_msa, gate_mlp, shift_mlp, scale_mlp, x_skip):
        '''Override: insert cross-attention between self-attention and MLP.
        Matches DDiTBlock.attn_mlp exactly, adding only cross_attn_visual.'''
        bias_dropout_scale_fn = self._get_bias_dropout_scale()
        if c is not None:
            x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)
            if self._encoder_out is not None: x = self.cross_attn_visual(x, self._encoder_out, self._encoder_mask)
            x = bias_dropout_scale_fn(
                self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
                None, gate_mlp, x, self.dropout
            )
        else:
            scale = torch.ones(1, device=x.device, dtype=x.dtype)
            x = bias_dropout_scale_fn(self.attn_out(x), None, scale, x_skip, self.dropout)
            if self._encoder_out is not None: x = self.cross_attn_visual(x, self._encoder_out, self._encoder_mask)
            x = bias_dropout_scale_fn(self.mlp(self.norm2(x)), None, scale, x, self.dropout)
        return x


class BlockDiffusionDecoder(nn.Module):
    '''Block Diffusion decoder for conditional text generation from visual features.

    Replaces MSKA's TranslationNetwork (mBART decoder) with a block diffusion
    decoder.  Supports both flex and sdpa attention backends (matching bd3lms).
    Sequences are padded to max_seq_len so the pre-compiled flex mask is valid
    for all inputs (bd3lms uses the same fixed-length approach).
    '''
    def __init__(self, vocab_size, d_model=1024, n_heads=16, n_layers=6,
                 cond_dim=128, block_size=4, mlp_ratio=4, dropout=0.3,
                 max_seq_len=128, pad_index=1, eos_index=2, bos_index=6,
                 sampling_eps_min=1e-3, sampling_eps_max=1.0,
                 antithetic_sampling=True, time_conditioning=False,
                 ignore_bos=True, nucleus_p=1.0, first_hitting=True,
                 kv_cache=True, attn_backend='flex'):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.block_size = block_size
        self.max_seq_len = max_seq_len
        self.pad_index = pad_index
        self.eos_index = eos_index
        self.bos_index = bos_index
        self.sampling_eps_min = sampling_eps_min
        self.sampling_eps_max = sampling_eps_max
        self.antithetic_sampling = antithetic_sampling
        self.time_conditioning = time_conditioning
        self.ignore_bos = ignore_bos
        self.nucleus_p = nucleus_p
        self.first_hitting = first_hitting
        self.use_kv_cache = kv_cache
        self.neg_infinity = -1000000.0
        self.attn_backend = attn_backend
        assert max_seq_len % block_size == 0, f'max_seq_len ({max_seq_len}) must be a multiple of block_size ({block_size})'

        # Vocab: append MASK token at end (matching bd3lms)
        self.mask_index = vocab_size
        self.full_vocab_size = vocab_size + 1 # MASK token is appended at end of vocab

        # Components (matching bd3lms DIT architecture)
        self.vocab_embed = EmbeddingLayer(d_model, self.full_vocab_size)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(d_model // n_heads)
        self.blocks = nn.ModuleList([ # Transformer blocks (ConditionalDDiTBlock = DDiTBlock + CrossAttention)
            ConditionalDDiTBlock(
                n=max_seq_len, dim=d_model, n_heads=n_heads, cond_dim=cond_dim, block_size=block_size,
                mlp_ratio=mlp_ratio, dropout=dropout, max_seqlen=max_seq_len, attn_backend=self.attn_backend
            ) for _ in range(n_layers)
        ])

        # Output projection and  (from bd3lms)
        self.output_layer = DDiTFinalLayer(hidden_size=d_model, out_channels=self.full_vocab_size, cond_dim=cond_dim, adaLN=True)
        self.noise = LogLinearNoise()

        # Block-causal mask (matching bd3lms DIT.gen_mask)
        # Created once at max_seq_len — sequences are padded to this length.
        if self.attn_backend == 'flex':
            self._block_diff_mask = create_block_mask(
                partial(block_diff_mask, block_size=block_size, n=max_seq_len),
                B=None, H=None, Q_LEN=max_seq_len * 2, KV_LEN=max_seq_len * 2)
        else:
            q_idx = torch.arange(max_seq_len * 2)[:, None]
            kv_idx = torch.arange(max_seq_len * 2)[None, :]
            self.register_buffer('_block_diff_mask', block_diff_mask(
                b=None, h=None, q_idx=q_idx, kv_idx=kv_idx,
                block_size=block_size, n=max_seq_len))

        print(f'BlockDiffusionDecoder: d={d_model}, heads={n_heads}, layers={n_layers}, block={block_size}, vocab={vocab_size}+1(MASK), '
              f'max_len={max_seq_len}, dropout={dropout}, attn={self.attn_backend}, time_cond={time_conditioning}, ignore_bos={ignore_bos}')

    # ── Diffusion utilities (matching bd3lms/diffusion.py) ────────────────────

    def _sigma_from_p(self, p): # Convert move_chance p to sigma — matches Diffusion._sigma_from_p
        return torch.min(-torch.log(1 - p), self.noise.sigma_max.to(p.device))

    def _process_sigma(self, sigma): # Ref: bd3lms/diffusion.py Diffusion._process_sigma lines 308-319
        assert sigma.ndim == 2
        sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0: sigma = sigma.unsqueeze(0)
        if not self.time_conditioning: sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _sample_t(self, batch_size, num_blocks, device): # Ref: bd3lms/diffusion.py Diffusion._sample_t lines 768-789
        # Sample timesteps per block with antithetic sampling
        _eps_b = torch.rand((batch_size, num_blocks), device=device)
        if self.antithetic_sampling:
            offset = torch.arange(batch_size * num_blocks, device=device).float()
            offset = offset / (batch_size * num_blocks)
            offset = offset.view(batch_size, num_blocks)
            _eps_b = (_eps_b / (batch_size * num_blocks) + offset) % 1
        t = _eps_b
        if self.block_size != self.max_seq_len: t = t.repeat_interleave(self.block_size, dim=-1)
        return t * (self.sampling_eps_max - self.sampling_eps_min) + self.sampling_eps_min

    def q_xt(self, x, p): # Ref: bd3lms/diffusion.py Diffusion.q_xt lines 502-536
        # Create noisy sample by masking tokens
        move_indices = torch.rand(*x.shape, device=x.device) <= p
        return torch.where(move_indices, self.mask_index, x)

    def _subs_parameterization(self, logits, xt): # Ref: bd3lms/diffusion.py Diffusion._subs_parameterization lines 275-291
        # Apply substitution parameterization
        logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked_indices = (xt != self.mask_index)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    @torch.no_grad()
    def _nucleus_sample(self, p_x0): # Ref: bd3lms/diffusion.py Diffusion._nucleus_sample lines 543-556
        if self.nucleus_p >= 1.0: return p_x0
        p_x0_ = p_x0[:, -self.block_size:].clone()
        sorted_probs, sorted_indices = p_x0_.sort(dim=-1, descending=True)
        cum_probs = sorted_probs.cumsum(dim=-1)
        nucleus_mask = cum_probs <= self.nucleus_p
        nucleus_mask[..., 0] = 1
        sorted_probs = sorted_probs * nucleus_mask
        p_x0_.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
        p_x0_ /= p_x0_.sum(-1, keepdim=True)
        p_x0[:, -self.block_size:] = p_x0_
        return p_x0

    # ── Encoder context management ────────────────────────────────────────────

    def _set_encoder_context(self, encoder_out, encoder_mask):
        for block in self.blocks:
            block._encoder_out = encoder_out
            block._encoder_mask = encoder_mask

    def _clear_encoder_context(self):
        for block in self.blocks:
            block._encoder_out = None
            block._encoder_mask = None

    # ── Training forward pass (matches DIT.forward structure) ──────────────────────

    def _backbone_forward(self, x_input, sigma, sample_mode=False, store_kv=False):
        '''Embedding -> transformer blocks -> output projection.
        Ref: bd3lms/models/dit.py DIT.forward lines 729-775.'''
        device = x_input.device
        x = self.vocab_embed(x_input)
        t_cond = F.silu(self.sigma_map(sigma)) if sigma is not None else None
        n = self.max_seq_len

        # Mask and rotary setup — mirrors DIT.forward exactly
        mask = self._block_diff_mask
        if sample_mode:
            if self.use_kv_cache and self.blocks[0].kv_cache is not None:
                # KV cache: full cross-attention to cached KV pairs
                # Ref: DIT.forward lines 741-748
                accum_len = self.blocks[0].cache_idx + self.block_size
                x_full = torch.zeros((x.shape[0], accum_len, x.shape[2]), device=device)
                rotary_cos_sin = self.rotary_emb(x_full)
                mask = None
            else: # No cache: index block-causal mask to x0 portion. Ref: DIT.forward lines 750-753
                rotary_cos_sin = self.rotary_emb(x)
                if self.attn_backend == 'sdpa':
                    mask = self._block_diff_mask[n:n + x_input.shape[1], n:n + x_input.shape[1]].to(device)
                else: mask = None # flex: always uses kv_cache during sampling
        else: # Training: rotary for xt portion only (first n tokens of [xt, x0]). Ref: DIT.forward line 756
            rotary_cos_sin = self.rotary_emb(x[:, :n])
            if self.attn_backend == 'sdpa': mask = mask.to(device)
            # flex: mask is already a block_mask object, passed directly

        # Blocks + output under bfloat16 autocast (matching DIT.forward)
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with torch.amp.autocast(device_type, dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, c=t_cond, mask=mask, sample_mode=sample_mode, store_kv=store_kv)
            logits = self.output_layer(x, t_cond)

        # Training: return only xt portion logits (matching DIT.forward line 773-774)
        if not sample_mode: logits = logits[:, :n]
        return logits


    def _forward_pass_diffusion(self, x0, attention_mask, encoder_out, encoder_mask):
        # Ref: bd3lms/diffusion.py Diffusion._forward_pass_diffusion + _loss
        B, L = x0.shape
        device = x0.device
        num_blocks = L // self.block_size
        
        t = self._sample_t(B, num_blocks, device) # Sample timesteps per block (matches bd3lms _sample_t)
        loss_scale, p = self.noise(t) # Get noise schedule: loss_scale = -1/t, p = t (matches LogLinearNoise)
        sigma = self._sigma_from_p(p[:, 0].unsqueeze(-1))  # (B, 1): sigma = -log(1 - p) for first block's timestep (matches bd3lms)
        
        xt = self.q_xt(x0, p) # Create noisy sample (matches bd3lms q_xt)
        if self.ignore_bos: xt[:, 0] = x0[:, 0]
        x_input = torch.cat((xt, x0), dim=-1) # Concatenate [xt, x0] for block-causal training (matches bd3lms cross_attn)
        sigma_proc = self._process_sigma(sigma) # Process sigma for conditioning: (B,1) -> (B,) (matches bd3lms _process_sigma)
        self._set_encoder_context(encoder_out, encoder_mask) # Set encoder context for cross-attention

        # Forward pass — wrapping in float32 autocast to match Diffusion.forward
        # (the inner _backbone_forward uses bfloat16 for blocks, matching DIT.forward)
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', dtype=torch.float32):
            logits = self._backbone_forward(x_input, sigma_proc)

        logits = self._subs_parameterization(logits, xt) # Apply subs parameterization OUTSIDE autocast (matches bd3lms flow)
        log_p_theta = torch.gather(input=logits, dim=-1, index=x0[:, :, None]).squeeze(-1) # Gather log-probs at GT positions
        loss = loss_scale * log_p_theta # (B, L)

        # During eval, exclude BOS from loss (matches bd3lms _loss lines 884-885)
        if self.ignore_bos and not self.training:
            attention_mask = attention_mask.clone()
            attention_mask[:, 0] = 0

        nlls = loss * attention_mask # Mask out padding positions (matches bd3lms _loss)
        return nlls.sum() / attention_mask.sum().clamp(min=1)


    def forward(self, **kwargs):
        '''Training forward pass — matches TranslationNetwork interface.

        Expected kwargs:
            input_feature: (B, T_v, D) visual features from VLMapper
            input_lengths: (B,) valid lengths of visual features
            labels: (B, L_text) target token IDs
            decoder_input_ids: (B, L_text) (unused, kept for interface compatibility)
        Returns:
            dict with 'translation_loss'
        '''
        input_feature = kwargs['input_feature']
        input_lengths = kwargs['input_lengths']
        labels = kwargs['labels']
        decoder_input_ids = kwargs.get('decoder_input_ids', None)
        B, device = input_feature.shape[0], input_feature.device

        # Build visual padding mask (on same device as input)
        T_v = input_feature.shape[1]
        encoder_mask = torch.zeros(B, T_v, dtype=torch.bool, device=device)
        for i in range(B): encoder_mask[i, :input_lengths[i].long()] = True

        # Prepare target tokens: remove ignore_index (-100) and replace with pad
        x0 = labels.clone().to(device)
        x0[x0 == -100] = self.pad_index

        if decoder_input_ids is not None: bos = decoder_input_ids[:, 0:1].to(device)
        else: bos = torch.full((B, 1), self.bos_index, dtype=x0.dtype, device=device)
        x0 = torch.cat([bos, x0], dim=1)

        # Attention mask: 1 for real tokens (BOS + text + EOS), 0 for padding
        text_mask = (x0 != self.pad_index).float()

        # Pad/truncate to max_seq_len (required for pre-compiled flex mask dimensions)
        # bd3lms uses fixed-length sequences (config.model.length); we do the same.
        if x0.shape[1] > self.max_seq_len:
            x0 = x0[:, :self.max_seq_len]
            text_mask = text_mask[:, :self.max_seq_len]
        if x0.shape[1] < self.max_seq_len:
            pad_len = self.max_seq_len - x0.shape[1]
            x0 = F.pad(x0, (0, pad_len), value=self.pad_index)
            text_mask = F.pad(text_mask, (0, pad_len), value=0.0)

        loss = self._forward_pass_diffusion(x0, text_mask, input_feature, encoder_mask)
        self._clear_encoder_context()
        return {'translation_loss': loss}

    # ── Inference ─────────────────────────────────────────────────────────────

    def reset_kv_cache(self, batch_size): # Initialize KV cache for inference
        device = next(self.parameters()).device
        for block in self.blocks:
            block.kv_cache = torch.zeros(batch_size, self.max_seq_len, self.d_model * 3, device=device, dtype=torch.bfloat16)
            block.cache_idx = 0

    def clear_kv_cache(self):
        for block in self.blocks:
            block.kv_cache = None
            block.cache_idx = 0


    @torch.no_grad()
    def _ddpm_caching_update(self, x, t, dt, p_x0=None, use_first_hitting=False):
        '''Single DDPM transition step with p_x0 caching.
        Ref: bd3lms/diffusion.py Diffusion._ddpm_caching_update lines 559-603.
        x is the context window (x_accum[:, fwd_idx]), NOT the full accumulator.
        '''
        _, move_chance_t = self.noise(t)             # (B, 1)
        _, move_chance_s = self.noise(t - dt)        # (B, 1)
        sigma_t = self._sigma_from_p(move_chance_t)  # (B, 1)
        move_chance_t = move_chance_t[:, None]
        move_chance_s = move_chance_s[:, None]
        mask_prob = move_chance_s / move_chance_t

        if p_x0 is None:
            sigma_proc = self._process_sigma(sigma_t)  # (B,)
            device_type = 'cuda' if x.device.type == 'cuda' else 'cpu'
            with torch.amp.autocast(device_type, dtype=torch.float32):
                if self.use_kv_cache and self.blocks[0].kv_cache is not None:
                    logits = self._backbone_forward(x[:, -self.block_size:], sigma_proc, sample_mode=True)
                else:
                    logits = self._backbone_forward(x, sigma_proc, sample_mode=True)
                    logits = logits[:, -self.block_size:] # Get predictions for current block only

            # SUBS parameterization + exp to get probabilities (matches bd3lms lines 572-578)
            xt_block = x[:, -self.block_size:]
            logits = self._subs_parameterization(logits.clone(), xt_block)
            p_x0 = logits.to(torch.float64).exp()
            p_x0 = self._nucleus_sample(p_x0)

        if use_first_hitting: # First-hitting sampler (exact bd3lms lines 580-587)
            x_block = _sample_categorical(p_x0)
            num_masked = (x[:, -self.block_size:] == self.mask_index).sum(-1)
            ind = torch.randint(0, num_masked, (x_block.shape[0],))
            ind = (x[:, -self.block_size:] == self.mask_index).nonzero()[ind, 1]
            mask = (torch.arange(self.block_size, device=x.device) == ind[:, None]).to(x_block.dtype)
            x_block = x_block * mask + x[:, -self.block_size:] * (1 - mask)
        else: # Standard DDPM transition (bd3lms lines 588-591)
            q_xs = p_x0 * (1 - mask_prob)
            q_xs[:, :, self.mask_index] = mask_prob.squeeze(-1)
            x_block = _sample_categorical(q_xs)

        # Carry-over: preserve already-unmasked tokens (bd3lms lines 592-594)
        copy_flag = (x[:, -self.block_size:] != self.mask_index).to(x.dtype)
        x_block = copy_flag * x[:, -self.block_size:] + (1 - copy_flag) * x_block
        x_new = torch.cat((x[:, :-self.block_size], x_block), dim=-1)

        # Store KV cache if entire block is committed (bd3lms lines 597-598)
        if self.use_kv_cache and self.mask_index not in x_block:
            sigma_proc = self._process_sigma(sigma_t)
            self._backbone_forward(x_block, sigma_proc, sample_mode=True, store_kv=True)

        # Cache p_x0 if state didn't change (for reuse, bd3lms lines 600-603)
        if not torch.allclose(x_new, x): return None, x_new
        return p_x0, x_new


    @torch.no_grad()
    def generate(self, input_feature=None, input_lengths=None, max_length=100, diffusion_steps=5000, **kwargs):
        '''Semi-AR block diffusion sampling.
        Ref: bd3lms/diffusion.py Diffusion._semi_ar_sampler lines 979-1052.
        Key: pass x_accum[:, fwd_idx] (context window) to _ddpm_caching_update,
        then write back x_accum[:, fwd_idx] = x_next (matching bd3lms exactly).
        '''
        B = input_feature.shape[0]
        device = input_feature.device

        # Build encoder mask and set context
        T_v = input_feature.shape[1]
        encoder_mask = torch.zeros(B, T_v, dtype=torch.bool, device=device)
        for i in range(B): encoder_mask[i, :input_lengths[i].long()] = True
        self._set_encoder_context(input_feature, encoder_mask)

        # First-hitting requires batch_size=1 (bd3lms constraint)
        use_first_hitting = self.first_hitting and B == 1
        num_blocks_max = max_length // self.block_size
        ones = torch.ones((B, 1), device=device)
        if self.use_kv_cache: self.reset_kv_cache(B)

        x_accum = None
        for block_idx in range(num_blocks_max):
            # Extend with MASK block (matches bd3lms _sample_prior + concat)
            new_block = torch.full((B, self.block_size), self.mask_index, dtype=torch.long, device=device)
            if x_accum is None:
                x_accum = new_block
                x_accum[:, 0] = self.bos_index
            else:
                x_accum = torch.cat((x_accum, new_block), dim=1)

            # Compute context window indices (matches bd3lms fwd_idx logic, lines 1011-1013)
            end_idx = (block_idx + 1) * self.block_size
            start_idx = max(end_idx - self.max_seq_len, 0)  # context can't exceed max_seq_len
            fwd_idx = torch.arange(start_idx, end_idx)

            dt = 1.0 / diffusion_steps
            t = 1.0
            p_x0_cache = None

            for step in range(diffusion_steps):
                if self.mask_index not in x_accum[:, fwd_idx]: break
                if use_first_hitting:
                    u = np.random.rand()
                    num_masked = (x_accum[:, fwd_idx] == self.mask_index).sum(-1).item()
                    if num_masked == 0: break
                    t *= u ** (1.0 / num_masked)
                else:
                    t = 1.0 - step / diffusion_steps

                # Pass windowed context to _ddpm_caching_update (matches bd3lms line 1034-1038)
                p_x0_cache, x_next = self._ddpm_caching_update(
                    x=x_accum[:, fwd_idx], t=t * ones, dt=dt, p_x0=p_x0_cache,
                    use_first_hitting=use_first_hitting)
                x_accum[:, fwd_idx] = x_next  # Write back to accumulator (matches bd3lms line 1042)

            if any(self.eos_index in x_accum[b, -self.block_size:] for b in range(B)):
                break # Check for EOS in current block

        self._clear_encoder_context()
        self.clear_kv_cache()
        return {'sequences': x_accum, 'decoded_sequences': None}