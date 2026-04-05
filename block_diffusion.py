'''Block Diffusion Decoder for conditional sign language translation.

Wraps BD3LMs' DIT architecture for *conditional* text generation by:
  - Importing the core building blocks (DDiTBlock, noise schedule, diffusion
    utilities) directly from the bd3lms repository.
  - Adding a CrossAttention sublayer in each block to attend to MSKA visual
    encoder features — this is the only genuinely new component.
  - Providing forward() and generate() that match TranslationNetwork's interface.

The decoder accepts:
  input_feature: (B, T_v, D_v)  visual features from VLMapper
  input_lengths: (B,)           valid lengths of visual features
  labels:        (B, L)         target text token IDs (for training)
'''
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ── Import building blocks from bd3lms repo ──────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'bd3lms'))

from noise_schedule import LogLinearNoise
from models.dit import (
    LayerNorm, TimestepEmbedder, EmbeddingLayer, DDiTFinalLayer, Rotary, rotate_half, apply_rotary_pos_emb_torchscript, 
    block_diff_mask, bias_dropout_add_scale_fused_train, bias_dropout_add_scale_fused_inference, modulate
)

class CrossAttention(nn.Module): # Multi-head cross-attention to visual encoder features
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(dim)

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

        attn_mask = None
        if encoder_mask is not None: # encoder_mask: (B, T_v) bool, True=valid → expand for SDPA
            attn_mask = encoder_mask[:, None, None, :].bool()

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)
        out = self.dropout(out)
        return residual + out


class ConditionalDDiTBlock(nn.Module): # Extends bd3lms DDiTBlock with cross-attention
    '''Transformer block with:
       - Block-causal self-attention (from BD3LMs DDiTBlock)
       - Cross-attention to visual encoder features (new)
       - MLP with adaLN modulation
    '''
    def __init__(self, n, dim, n_heads, cond_dim, block_size, mlp_ratio=4, dropout=0.1, max_seqlen=1024):
        super().__init__()
        self.n = n  # seq_len of xt (= seq_len of x0)
        self.n_heads = n_heads
        self.block_size = block_size
        self.max_seqlen = max_seqlen
        self.head_dim = dim // n_heads
        self.adaLN = True

        # Self-attention (same structure as DDiTBlock)
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        # Cross-attention to visual features (NEW)
        self.cross_attn_visual = CrossAttention(dim, n_heads, dropout=dropout)

        # MLP (same structure as DDiTBlock)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout = dropout

        # adaLN modulation (6 params: shift/scale/gate for attn + mlp)
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

        # KV cache for inference (matching DDiTBlock's cache mechanism)
        self.kv_cache = None
        self.cache_idx = 0
        self.attn_backend = 'sdpa'  # We always use SDPA (no flash_attn dependency)


    def _get_bias_dropout_scale(self):
        if self.training: return bias_dropout_add_scale_fused_train
        return bias_dropout_add_scale_fused_inference


    def get_qkv(self, x, rotary_cos_sin, store_kv=False):
        # Compute QKV with optional KV cache — matches DDiTBlock.get_qkv exactly
        if self.kv_cache is not None:
            new_qkv = self.attn_qkv(x)
            self.kv_cache[:, self.cache_idx:self.cache_idx + self.block_size] = new_qkv
            qkv = self.kv_cache[:, :self.cache_idx + self.block_size].clone()
        else:
            qkv = self.attn_qkv(x)

        if store_kv:
            self.cache_idx += self.block_size
            if self.cache_idx >= self.max_seqlen:
                self.cache_idx = self.max_seqlen - self.block_size
                self.kv_cache[:, :-self.block_size] = self.kv_cache[:, self.block_size:].clone()

        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        with torch.amp.autocast('cuda', enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb_torchscript(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        return qkv


    def _self_attn(self, qkv, mask=None):
        '''SDPA self-attention — matches DDiTBlock.cross_attn (the bd3lms naming
        is confusing: their cross_attn is actually self-attention with mask).'''
        scale = qkv.shape[-1]
        qkv = qkv.transpose(1, 3)  # (B, H, 3, S, D)
        x = F.scaled_dot_product_attention(
            query=qkv[:, :, 0], key=qkv[:, :, 1], value=qkv[:, :, 2],
            attn_mask=mask.bool() if mask is not None else None, 
            is_causal=False, scale=1 / scale**0.5
        ).transpose(1, 2)
        return rearrange(x, 'b s h d -> b s (h d)')


    def forward(self, x, rotary_cos_sin, c, encoder_out, encoder_mask=None, mask=None, sample_mode=False, store_kv=False):
        '''
        Args:
            x: (B, S, D) input embeddings ([xt|x0] during training, block during inference)
            rotary_cos_sin: tuple of rotary embeddings
            c: (B, cond_dim) timestep conditioning
            encoder_out: (B, T_v, D) visual features
            encoder_mask: (B, T_v) padding mask for visual features
            mask: (2S, 2S) block-causal mask for training
            sample_mode: whether in inference mode
            store_kv: whether to cache KV for next block
        '''
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        # adaLN modulation
        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # Self-attention (matching DDiTBlock.forward exactly)
        x_skip = x
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)

        # QKV computation with [xt|x0] split during training
        if mask is not None and not sample_mode:
            qkv_x = self.get_qkv(x_norm[:, :self.n], rotary_cos_sin)
            qkv_x0 = self.get_qkv(x_norm[:, self.n:], rotary_cos_sin)
            qkv = torch.cat((qkv_x, qkv_x0), dim=1)
        else:
            qkv = self.get_qkv(x_norm, rotary_cos_sin, store_kv=store_kv)

        x_attn = self._self_attn(qkv, mask=mask)
        if self.kv_cache is not None: x_attn = x_attn[:, -self.block_size:]

        x = bias_dropout_scale_fn(self.attn_out(x_attn), None, gate_msa, x_skip, self.dropout)
        x = self.cross_attn_visual(x, encoder_out, encoder_mask) # Cross-attention to visual features (NEW)
        return bias_dropout_scale_fn(
            self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)),
            None, gate_mlp, x, self.dropout)


class BlockDiffusionDecoder(nn.Module):
    '''Block Diffusion decoder for conditional text generation from visual features.

    Replaces MSKA's TranslationNetwork (mBART decoder) with a block diffusion
    decoder that uses DiT-style blocks with cross-attention to visual features.

    Interface matches TranslationNetwork:
        forward(input_feature, input_lengths, labels, ...) -> dict with 'translation_loss'
        generate(inputs_embeds, attention_mask, ...) -> dict with 'decoded_sequences'
    '''
    def __init__(self, vocab_size, d_model=1024, n_heads=16, n_layers=6, cond_dim=128, block_size=4, mlp_ratio=4, dropout=0.1,
                 max_seq_len=128, pad_index=0, eos_index=2, sampling_eps_min=1e-3, sampling_eps_max=1.0, antithetic_sampling=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.block_size = block_size
        self.max_seq_len = max_seq_len
        self.pad_index = pad_index
        self.eos_index = eos_index
        self.sampling_eps_min = sampling_eps_min
        self.sampling_eps_max = sampling_eps_max
        self.antithetic_sampling = antithetic_sampling

        
        self.mask_index = vocab_size
        self.full_vocab_size = vocab_size + 1 # MASK token is appended at end of vocab
        self.vocab_embed = EmbeddingLayer(d_model, self.full_vocab_size) # Token embedding (from bd3lms)
        self.sigma_map = TimestepEmbedder(cond_dim) # Timestep conditioning (from bd3lms)
        self.rotary_emb = Rotary(d_model // n_heads) # Rotary embeddings (from bd3lms)

        # Transformer blocks (ConditionalDDiTBlock = DDiTBlock + CrossAttention)
        self.blocks = nn.ModuleList([
            ConditionalDDiTBlock(
                n=max_seq_len, dim=d_model, n_heads=n_heads, cond_dim=cond_dim, block_size=block_size,
                mlp_ratio=mlp_ratio, dropout=dropout, max_seqlen=max_seq_len
            ) for _ in range(n_layers)
        ])

        # Output projection and  (from bd3lms)
        self.output_layer = DDiTFinalLayer(hidden_size=d_model, out_channels=self.full_vocab_size, cond_dim=cond_dim, adaLN=True)
        self.noise = LogLinearNoise()

        # Pre-compute block-causal mask (using bd3lms' block_diff_mask)
        self._cached_mask_len = None
        self._cached_mask = None
        self._cached_mask_device = None
        self.neg_infinity = -1000000.0

        print(f'BlockDiffusionDecoder: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, '
              f'block_size={block_size}, vocab_size={vocab_size}+1(MASK), max_seq_len={max_seq_len}')


    def _get_block_causal_mask(self, seq_len, device): # Get or build the block-causal mask for training using bd3lms logic
        if self._cached_mask_len != seq_len or self._cached_mask_device != device:
            q_idx = torch.arange(seq_len * 2)[:, None]
            kv_idx = torch.arange(seq_len * 2)[None, :]
            self._cached_mask = block_diff_mask( # Use bd3lms' block_diff_mask function directly
                b=None, h=None, q_idx=q_idx, kv_idx=kv_idx,
                block_size=self.block_size, n=seq_len
            ).to(device)
            self._cached_mask_len = seq_len
            self._cached_mask_device = device
        return self._cached_mask


    # ── Diffusion utilities (matching bd3lms/diffusion.py) ────────────────────
    
    def _sigma_from_p(self, p): # Convert move_chance p to sigma — matches Diffusion._sigma_from_p
        return torch.min(-torch.log(1 - p), self.noise.sigma_max.to(p.device))

    def _sample_t(self, batch_size, num_blocks, device):
        # Sample timesteps per block with antithetic sampling. Matches bd3lms Diffusion._sample_t
        eps_b = torch.rand((batch_size, num_blocks), device=device)
        if self.antithetic_sampling:
            offset = torch.arange(batch_size * num_blocks, device=device).float()
            offset = offset / (batch_size * num_blocks)
            offset = offset.view(batch_size, num_blocks)
            eps_b = (eps_b / (batch_size * num_blocks) + offset) % 1
        return eps_b * (self.sampling_eps_max - self.sampling_eps_min) + self.sampling_eps_min

    def q_xt(self, x0, p):
        # Create noisy sample by masking tokens — matches Diffusion.q_xt
        move_indices = torch.rand(*x0.shape, device=x0.device) <= p
        return torch.where(move_indices, self.mask_index, x0)

    def _subs_parameterization(self, logits, xt):
        # Apply substitution parameterization — matches Diffusion._subs_parameterization
        logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked_indices = (xt != self.mask_index)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _process_sigma(self, sigma): # Process sigma for conditioning — matches Diffusion._process_sigma.
        # Input: (B, 1) or (B, K). Output: (B,) — 1D for TimestepEmbedder
        assert sigma.ndim == 2, f'Expected 2D sigma, got shape {sigma.shape}'
        sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0: sigma = sigma.unsqueeze(0)
        assert sigma.ndim == 1, f'Expected 1D sigma after processing, got {sigma.shape}'
        return sigma


    def _pad_to_block_size(self, tokens, lengths): # Pad token sequences so length is divisible by block_size.
        B, L = tokens.shape
        remainder = L % self.block_size
        if remainder == 0: return tokens, lengths
        pad_len = self.block_size - remainder
        padding = torch.full((B, pad_len), self.pad_index, dtype=tokens.dtype, device=tokens.device)
        return torch.cat([tokens, padding], dim=1), lengths


    # ── Training forward pass ─────────────────────────────────────────────────
    
    def _backbone_forward(self, x_input, sigma, encoder_out, encoder_mask, self_attn_mask=None, sample_mode=False, store_kv=False):
        '''Run through embedding + blocks + output layer.

        Matches bd3lms DIT.forward structure:
          - Embedding + sigma conditioning
          - Rotary PE: training uses x[:, :n], sampling uses appropriate context
          - Blocks run under bfloat16 autocast (matching DIT.forward)
          - Output slicing: training returns first n logits (xt portion only)

        The outer Diffusion.forward wraps this in float32 autocast, but the inner
        DIT.forward overrides with bfloat16 for the blocks. We replicate the inner
        (bfloat16) since we apply _subs_parameterization outside this function.

        Args:
            x_input: (B, S) token indices (training: [xt|x0], inference: current seq)
            sigma: (B,) noise level (1D, already processed)
            encoder_out: (B, T_v, D) visual features
            encoder_mask: (B, T_v) padding mask
        Returns:
            logits: (B, S, vocab_size+1) or (B, block_size, vocab_size+1) if cached
        '''
        # Ensure encoder features are on same device as input
        device = x_input.device
        encoder_out = encoder_out.to(device)
        if encoder_mask is not None: encoder_mask = encoder_mask.to(device)
        if sigma is not None: sigma = sigma.to(device)

        x = self.vocab_embed(x_input)
        t_cond = F.silu(self.sigma_map(sigma)) if sigma is not None else None

        is_cross_attn_training = self_attn_mask is not None and not sample_mode
        if is_cross_attn_training: # Training: rotary for xt portion only (first half), matching DIT.forward
            n = x.shape[1] // 2
            rotary_cos_sin = self.rotary_emb(x[:, :n])
        else:
            if sample_mode and self.blocks[0].kv_cache is not None:
                # KV cache mode: positional encodings for full cached context
                accum_len = self.blocks[0].cache_idx + self.block_size
                x_full = torch.zeros((x.shape[0], accum_len, x.shape[2]), device=device)
                rotary_cos_sin = self.rotary_emb(x_full)
            else:
                rotary_cos_sin = self.rotary_emb(x)

        # Blocks + output under bfloat16 autocast (matching DIT.forward)
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with torch.amp.autocast(device_type, dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, t_cond, encoder_out, encoder_mask,
                          mask=self_attn_mask, sample_mode=sample_mode, store_kv=store_kv)
            logits = self.output_layer(x, t_cond)

        if is_cross_attn_training: # During training, only return logits for xt portion (matching DIT.forward)
            n = x_input.shape[1] // 2
            logits = logits[:, :n]
        return logits


    def _compute_diffusion_loss(self, x0, attention_mask, encoder_out, encoder_mask):
        '''Compute block diffusion training loss — follows bd3lms Diffusion._forward_pass_diffusion closely.

        Args:
            x0: (B, L) clean target tokens (padded to block_size multiple)
            attention_mask: (B, L) token mask (1 = real, 0 = padding)
            encoder_out: (B, T_v, D) visual features
            encoder_mask: (B, T_v) visual padding mask
        Returns:
            loss: scalar tensor
        '''
        B, L = x0.shape
        device = x0.device
        num_blocks = L // self.block_size
        
        t = self._sample_t(B, num_blocks, device) # Sample timesteps per block (matches bd3lms _sample_t)
        t_expanded = t.repeat_interleave(self.block_size, dim=-1) if self.block_size != L else t
        loss_scale, p = self.noise(t_expanded) # Get noise schedule: loss_scale = -1/t, p = t (matches LogLinearNoise)
        sigma = self._sigma_from_p(p[:, 0].unsqueeze(-1))  # (B, 1): sigma = -log(1 - p) for first block's timestep (matches bd3lms)
        
        xt = self.q_xt(x0, p) # Create noisy sample (matches bd3lms q_xt)
        x_input = torch.cat((xt, x0), dim=-1) # Concatenate [xt, x0] for block-causal training (matches bd3lms cross_attn)
        mask = self._get_block_causal_mask(L, device) # Get block-causal mask (using bd3lms block_diff_mask)
        sigma_proc = self._process_sigma(sigma) # Process sigma for conditioning: (B,1) -> (B,) (matches bd3lms _process_sigma)

        # Forward pass — wrapping in float32 autocast to match Diffusion.forward
        # (the inner _backbone_forward uses bfloat16 for blocks, matching DIT.forward)
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', dtype=torch.float32):
            logits = self._backbone_forward(x_input, sigma_proc, encoder_out, encoder_mask, self_attn_mask=mask, sample_mode=False)

        logits = self._subs_parameterization(logits, xt) # Apply subs parameterization OUTSIDE autocast (matches bd3lms flow)
        log_p_theta = torch.gather(input=logits, dim=-1, index=x0[:, :, None]).squeeze(-1) # Gather log-probs at GT positions
        loss_scale = loss_scale.clamp(min=-1000.0) # Clamp loss_scale to prevent NaN for very small t
        loss = loss_scale * log_p_theta  # (B, L)
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

        B = input_feature.shape[0]
        device = input_feature.device

        # Build visual padding mask (on same device as input)
        T_v = input_feature.shape[1]
        encoder_mask = torch.zeros(B, T_v, dtype=torch.bool, device=device)
        for i in range(B): encoder_mask[i, :input_lengths[i].long()] = True

        # Prepare target tokens: remove ignore_index (-100) and replace with pad
        x0 = labels.clone().to(device)
        x0[x0 == -100] = self.pad_index

        # Build attention mask for valid text positions
        text_mask = (labels != self.pad_index) & (labels != -100)
        text_mask = text_mask.float().to(device)

        x0, _ = self._pad_to_block_size(x0, input_lengths)
        text_mask_padded = F.pad(text_mask, (0, x0.shape[1] - text_mask.shape[1]), value=0.0)

        for block in self.blocks: block.n = x0.shape[1] # Update n for blocks (sequence length varies per batch)
        loss = self._compute_diffusion_loss(x0, text_mask_padded, input_feature, encoder_mask)
        return {'translation_loss': loss, 'logits': None}


    # ── Inference ─────────────────────────────────────────────────────────────

    def reset_kv_cache(self, batch_size): # Initialize KV cache for inference
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        for block in self.blocks:
            block.kv_cache = torch.zeros(batch_size, self.max_seq_len, self.d_model * 3, device=device, dtype=dtype)
            block.cache_idx = 0

    def clear_kv_cache(self):
        for block in self.blocks:
            block.kv_cache = None
            block.cache_idx = 0

    def _get_inference_mask(self, seq_len, device):
        '''Get the block-causal mask sliced for inference (non-KV-cache mode).
        Matches DIT.forward: mask[self.n:self.n+x.shape[1], self.n:self.n+x.shape[1]]
        where self.n is the original sequence length used to build the mask.
        '''
        # Ensure we have a mask at least big enough
        n = self.max_seq_len
        if self._cached_mask_len != n or self._cached_mask_device != device: self._get_block_causal_mask(n, device)
        # Slice the x0 portion of the mask for the current sequence length
        # In bd3lms: mask[n:n+seq_len, n:n+seq_len] gives block-causal for x0 portion
        return self._cached_mask[n:n + seq_len, n:n + seq_len]
    
    def _sample_categorical(self, categorical_probs):
        gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
        return (categorical_probs / gumbel_norm).argmax(dim=-1)


    @torch.no_grad() # Generate text using semi-AR block diffusion sampling
    def generate(self, inputs_embeds=None, attention_mask=None, input_feature=None, input_lengths=None,
                 num_beams=None, max_length=100, length_penalty=1, diffusion_steps=5000, **kwargs):
        # Follows bd3lms Diffusion._semi_ar_sampler and _ddpm_caching_update closely for the denoising logic.
        if input_feature is not None:
            encoder_out = input_feature
            B = input_feature.shape[0]
            device = input_feature.device
            T_v = input_feature.shape[1]
            encoder_mask = torch.zeros(B, T_v, dtype=torch.bool, device=device)
            for i in range(B): encoder_mask[i, :input_lengths[i].long().item()] = True
        elif inputs_embeds is not None:
            encoder_out = inputs_embeds
            device = inputs_embeds.device
            encoder_mask = attention_mask.bool() if attention_mask is not None else None
            B = inputs_embeds.shape[0]
        else: raise ValueError('Must provide either input_feature or inputs_embeds')

        # Start with empty output
        all_tokens = torch.full((B, 0), self.pad_index, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        self.clear_kv_cache()
        num_blocks_max = max_length // self.block_size
        ones = torch.ones((B, 1), device=device)

        for block_idx in range(num_blocks_max):
            # Initialize block with all MASK tokens
            x_block = torch.full((B, self.block_size), self.mask_index, dtype=torch.long, device=device)
            x_accum = torch.cat([all_tokens, x_block], dim=-1)

            
            for block in self.blocks: block.n = x_accum.shape[1] # Update n for blocks
            dt = 1.0 / diffusion_steps # Iterative denoising within this block. Following bd3lms _ddpm_caching_update logic
            t = 1.0

            for step in range(diffusion_steps):
                if self.mask_index not in x_accum[:, -self.block_size:]: break
                t_tensor = t * ones  # (B, 1)
                _, move_chance_t = self.noise(t_tensor)  # (B, 1)
                sigma_t = self._sigma_from_p(move_chance_t)  # (B, 1)
                sigma_proc = self._process_sigma(sigma_t)  # (B,)
                inf_mask = self._get_inference_mask(x_accum.shape[1], device)

                # Forward pass with float32 autocast (matching Diffusion.forward)
                with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', dtype=torch.float32):
                    logits = self._backbone_forward(x_accum, sigma_proc, encoder_out, encoder_mask,
                        self_attn_mask=inf_mask, sample_mode=True, store_kv=False)
                logits_block = logits[:, -self.block_size:] # Get predictions for current block only

                # Subs parameterization on block (matching bd3lms flow)
                xt_block = x_accum[:, -self.block_size:]
                logits_block[:, :, self.mask_index] += self.neg_infinity
                logits_block = logits_block - torch.logsumexp(logits_block, dim=-1, keepdim=True)
                p_x0 = logits_block.to(torch.float64).exp() # Convert to float64 for numerical stability (matching bd3lms)

                # DDPM update (matching bd3lms _ddpm_caching_update)
                t_next = max(t - dt, 0)
                if t_next > 0:
                    t_next_tensor = t_next * ones  # (B, 1)
                    _, move_chance_s = self.noise(t_next_tensor)  # (B, 1)
                    # move_chance_s, move_chance_t are (B, 1)
                    mask_prob = (move_chance_s / move_chance_t)[:, :, None]  # (B, 1, 1)
                    q_xs = p_x0 * (1 - mask_prob)
                    q_xs[:, :, self.mask_index] = mask_prob.squeeze(-1)
                    x_new = self._sample_categorical(q_xs)
                else:
                    x_new = p_x0.argmax(dim=-1)

                # Carry-over: preserve already-unmasked tokens (matching bd3lms)
                copy_flag = (xt_block != self.mask_index).to(x_new.dtype)
                x_block = copy_flag * xt_block + (1 - copy_flag) * x_new
                x_accum = torch.cat([x_accum[:, :-self.block_size], x_block], dim=-1)
                t = t_next

            all_tokens = x_accum # Finalize this block
            for b in range(B): # Check for EOS
                if not finished[b]:
                    block_tokens = all_tokens[b, -self.block_size:]
                    if self.eos_index in block_tokens: finished[b] = True

            if finished.all(): break
        return {'sequences': all_tokens}