'''Block Diffusion Decoder for conditional sign language translation.

Implements BD3LM (block diffusion) adapted to mBART via the A2D recipe from
dLLM (arxiv.org/abs/2602.22661), with full bd3lms fidelity:
  - Architecture: pretrained mBART decoder with block-causal self-attention.
  - Training: BD3LM masked diffusion loss with [xt | x0] concatenation,
              BD3LM attention mask (M_BD + M_OBC + M_BC), repeated position IDs,
              and cross-entropy weighted by 1/t at masked positions.
  - Inference: BD3LM semi-AR sampling with DDPM transitions / first-hitting,
               p_x0 caching, nucleus sampling, block-by-block denoising.

Key insight (from dLLM, takeaway box p.8): AR and diffusion models differ only in training
objective and attention mask, NOT in architecture. Converting mBART's decoder to BD3LM requires:
  1. Replace causal self-attention mask with BD3LM mask during training.
  2. Concatenate noised tokens xt with clean tokens x0 as model input.
  3. Use repeated position IDs [0..L-1, 0..L-1] for both halves.
  4. Compute MDLM masked diffusion loss on only the xt-half logits.

BD3LM training mask (2L × 2L) over concatenated [xt | x0] input:
  M_BD:  Block diagonal — within-block self-attention (xt↔xt, x0↔x0).
  M_OBC: Offset block causal — xt attends to x0 from *previous* blocks.
  M_BC:  Block causal — x0 attends to x0 from same and previous blocks.

Inference uses simple block-causal mask (committed prefix + current noised block)
with DDPM transitions or first-hitting sampler (bd3lms _semi_ar_sampler + _ddpm_caching_update):
  - No x0 concatenation needed; finalized prefix is clean in x_accum.
  - Simple block-causal mask (xt_current attends to clean prefix by causal mask).
  - DDPM transitions or first-hitting sampler.
  - p_x0 caching, nucleus sampling.

References:
  - dLLM paper + A2D recipe: https://arxiv.org/pdf/2602.22661
  - dLLM BD3LMTrainer: dllm/core/trainers/bd3lm.py (BD3LMTrainer.compute_loss)
  - BD3LM paper: https://arxiv.org/pdf/2503.09573
  - bd3lms reference: bd3lms/diffusion.py, bd3lms/models/hf/modeling_bd3lm.py
  - LogLinearNoise: bd3lms/noise_schedule.py
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_utils import _sample_categorical, LogLinearNoise, TimestepEmbedder
from transformers.modeling_attn_mask_utils import AttentionMaskConverter


def build_bd3lm_mask(seq_len, block_size, dtype, device):
    '''BD3LM training attention mask for concatenated [xt | x0] input.

    Mirrors _create_bd3lm_attention_mask (dLLM/dllm/core/trainers/bd3lm.py) and
    block_diff_mask (bd3lms/models/dit.py). 
    
    For input of length 2*tgt_len, creates a (1, 1, 2L, 2L) mask with 3 components:
    - M_BD:  xt_block_b ↔ all xt in same block b  (bidirectional, noisy self-attn within each block)
    - M_OBC: xt_block_b → x0_block_{0..b-1}       (clean prefix, STRICTLY prior, cross-attn for conditional context)
    - M_BC:  x0_block_b → x0_block_{0..b}         (block-causal over clean copy)
    x0 never attends to xt; inference uses build_block_causal_mask instead.

    Returns:
        (1, 1, 2L, 2L) float mask: 0 = attend, -inf = masked.
    '''
    n = seq_len
    idx = torch.arange(2 * n, device=device)
    q_idx  = idx[:, None]   # (2L, 1)
    kv_idx = idx[None, :]   # (1, 2L)

    # Indicate whether token belongs to xt or x0
    x0_flag_q, = q_idx  >= n
    x0_flag_kv = kv_idx >= n
    
    # Compute block indices
    block_q  = torch.where(x0_flag_q,  (q_idx  - n) // block_size, q_idx  // block_size)
    block_kv = torch.where(x0_flag_kv, (kv_idx - n) // block_size, kv_idx // block_size)

    # M_BD: same block, same half (xt-xt or x0-x0)
    # M_OBC: xt queries attend to x0 keys from strictly earlier/previous blocks
    # M_BC: x0 queries attend to x0 keys from same/current or earlier/previous blocks
    block_diagonal      = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
    offset_block_causal = (block_q >  block_kv) & x0_flag_kv & ~x0_flag_q
    block_causal        = (block_q >= block_kv) & x0_flag_kv & x0_flag_q

    # Combine Masks
    can_attend = block_diagonal | offset_block_causal | block_causal
    mask = torch.zeros(2 * n, 2 * n, dtype=dtype, device=device)
    mask = mask.masked_fill(~can_attend, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, 2L, 2L)


def build_block_causal_mask(batch_size, tgt_len, block_size, dtype, device):
    '''Block-causal (staircase) attention mask.

    Bidirectional within each block; block b can attend to blocks 0..b (causal
    across blocks). Based on dLLM HF quickstart build_staircase_attention_mask.

    Returns:
        (B, 1, T, T) float mask: 0 = attend, -inf = masked.
    '''
    positions  = torch.arange(tgt_len, device=device)
    block_ids  = positions // block_size     # (T,)
    q_block    = block_ids.view(tgt_len, 1)  # (T, 1)
    k_block    = block_ids.view(1, tgt_len)  # (1, T)
    can_attend = (k_block <= q_block)        # (T, T): True = can attend
    mask = torch.zeros(tgt_len, tgt_len, dtype=dtype, device=device)
    mask = mask.masked_fill(~can_attend, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, tgt_len, tgt_len)


class BlockDiffusionDecoder(nn.Module):
    '''BD3LM decoder built on the pretrained mBART decoder backbone.

    Replaces MSKA's TranslationNetwork (mBART AR decoder) with a block diffusion
    decoder that:
      - Shares the same mBART pretrained weights (encoder + decoder layers).
      - Replaces the causal decoder self-attention mask with a block-causal mask.
      - Trains with MDLM masked diffusion loss (loglinear noise schedule).
      - Generates via iterative denoising block-by-block at inference time.

    Cross-attention to visual encoder features is inherited directly from mBART's
    decoder layers — no separate cross-attention module is needed.
    '''
    def __init__(
        self, translation_network, block_size=4, sampling_eps_min=1e-3, sampling_eps_max=1.0,
        antithetic_sampling=True, ignore_bos=True, first_hitting=True, nucleus_p=1.0, time_conditioning=False
    ):
        super().__init__()
        mbart = translation_network.model # MBartForConditionalGeneration
        self.block_size = block_size
        self.sampling_eps_min = sampling_eps_min
        self.sampling_eps_max = sampling_eps_max
        self.antithetic_sampling = antithetic_sampling
        self.ignore_bos = ignore_bos
        self.first_hitting = first_hitting
        self.nucleus_p = nucleus_p
        self.neg_infinity = -1e9
        self.time_conditioning = time_conditioning # Sigma conditioning (additive TimestepEmbedder)

        # ── mBART components (pretrained weights) ────────────────────────────
        self._prepare_feature_inputs = translation_network.prepare_feature_inputs
        self.mbart_encoder = mbart.model.encoder   # MBartEncoder
        self.mbart_decoder = mbart.model.decoder   # MBartDecoder layers + norms
        self.embed_scale = translation_network.input_embed_scale  # sqrt(d_model)
        self.d_model = mbart.config.d_model

        # ── Tokenizer info ───────────────────────────────────────────────────
        self.text_tokenizer = translation_network.text_tokenizer
        self.pad_index = self.text_tokenizer.pad_index
        self.eos_index = self.text_tokenizer.eos_index
        self.bos_index = getattr(self.text_tokenizer, 'sos_index', getattr(self.text_tokenizer, 'lang_index', 0))
        vocab_size = mbart.config.vocab_size
        self.vocab_size = vocab_size

        # ── Extend vocabulary with a MASK token ─────────────────────────────
        self.mask_token_id = vocab_size # Append [MASK] at index vocab_size so existing token IDs are unchanged
        self.mask_index = self.mask_token_id # Backward-compat attribute name for model_factory._decode_sequences

        # ── Extend embedding and language model heads ───────────────────────
        old_embed = mbart.model.shared  # nn.Embedding (vocab_size, d_model)
        old_lm    = mbart.lm_head       # nn.Linear (d_model → vocab_size)
        d_model   = old_embed.embedding_dim
        self.embed_tokens = nn.Embedding(vocab_size + 1, d_model, padding_idx=self.pad_index)
        self.lm_head = nn.Linear(d_model, vocab_size + 1, bias=False)
        with torch.no_grad():
            self.embed_tokens.weight[:vocab_size].copy_(old_embed.weight)
            self.lm_head.weight[:vocab_size].copy_(old_lm.weight)
            nn.init.normal_(self.embed_tokens.weight[vocab_size:], std=0.02)
            nn.init.zeros_(self.lm_head.weight[vocab_size:])

        # Point decoder's embed_tokens to our extended version so that the
        # original (un-extended) embedding is not a duplicate parameter.
        self.mbart_decoder.embed_tokens = self.embed_tokens
        self.noise = LogLinearNoise()
        if self.time_conditioning: self.sigma_embedder = TimestepEmbedder(d_model)
        print(
            f'BlockDiffusionDecoder (mBART A2D): d_model={d_model}, vocab={vocab_size}+1(MASK), '
            f'block_size={block_size}, time_cond={time_conditioning}, first_hitting={first_hitting}, '
            f'nucleus_p={nucleus_p}, antithetic_sampling={antithetic_sampling}, ignore_bos={ignore_bos}, '
            f'sampling_eps_min={self.sampling_eps_min}, sampling_eps_max={self.sampling_eps_max}')
        
    # ── Visual encoder ────────────────────────────────────────────────────────

    def _encode_visual(self, input_feature, input_lengths):
        # Encode visual features via mBART encoder (same pipeline as AR model)
        enc_inputs = self._prepare_feature_inputs(input_feature, input_lengths)
        enc_out = self.mbart_encoder(
            inputs_embeds=enc_inputs['inputs_embeds'],
            attention_mask=enc_inputs['attention_mask'],
            return_dict=True,
        )
        return enc_out.last_hidden_state, enc_inputs['attention_mask']

    # ── Decoder with block-causal / bd3lm self-attention ──────────────────────
    
    def _decode(self, decoder_input_ids, enc_hidden, enc_mask, self_attn_mask=None, position_ids=None, sigma=None):
        '''Run mBART decoder layers with custom self-attention mask.

        Bypasses MBartDecoder.forward() to replace its internal causal mask. Everything else
        (positional embeddings, layer norms, cross-attention, FFN) is identical to the AR path.

        Args:
            decoder_input_ids: (B, T) token IDs.
            enc_hidden: encoder hidden states for cross-attention.
            enc_mask: encoder padding mask.
            self_attn_mask: optional (1, 1, T, T) or (B, 1, T, T) float mask.
                If None, builds block-causal mask from block_size.
            position_ids: optional (B, T) position IDs for embed_positions.
                If None, uses default sequential positions from embed_positions.
            sigma: optional (B,) or (B, 1) noise level for time conditioning.
        '''
        batch_size, tgt_len = decoder_input_ids.shape
        device = decoder_input_ids.device
        dtype = enc_hidden.dtype
        inputs_embeds = self.embed_tokens(decoder_input_ids) * self.embed_scale

        # if position_ids is not None:
        #     # positions for xt and x0 halves should both be [0..L-1]
        #     # When position_ids is provided, it manually indexes into embed_positions.weight[pos + 2] 
        #     # to support the repeated [0..L-1, 0..L-1] positions needed for the 2L concatenated input
        #     positions = self.mbart_decoder.embed_positions.weight[position_ids + 2]
        # else:
        #     positions = self.mbart_decoder.embed_positions(decoder_input_ids)
        
        hidden = inputs_embeds + self.mbart_decoder.embed_positions(decoder_input_ids)
        hidden = self.mbart_decoder.layernorm_embedding(hidden)
        hidden = F.dropout(hidden, p=self.mbart_decoder.dropout, training=self.training)
        
        # Self-attention mask: BD3LM mask during training or build block-causal mask for inference
        if self_attn_mask is None:
            self_mask = build_block_causal_mask(batch_size, tgt_len, self.block_size, dtype, device)
        else:
            self_mask = self_attn_mask.to(dtype=dtype, device=device)
            
        # Optional sigma conditioning: add to all sequence positions
        if sigma is not None and hasattr(self, 'sigma_embedder'):
            sigma_emb = self.sigma_embedder(sigma)    # (B, D)
            hidden = hidden + sigma_emb.unsqueeze(1)  # broadcast over seq
        
        # Cross-attention mask: expand encoder padding mask to 4D
        cross_mask = ( 
            AttentionMaskConverter._expand_mask(enc_mask, dtype, tgt_len=tgt_len)
            if enc_mask is not None else None)

        # Run each decoder layer (self-attn + cross-attn + FFN)
        for layer in self.mbart_decoder.layers:
            hidden = layer(
                hidden, attention_mask=self_mask,
                encoder_hidden_states=enc_hidden,
                encoder_attention_mask=cross_mask,
            )[0]

        # Final layer norm (present in mBART-large)
        if getattr(self.mbart_decoder, 'layer_norm', None) is not None:
            hidden = self.mbart_decoder.layer_norm(hidden)
        return self.lm_head(hidden)   # (B, T, vocab_size+1)

    # ── Parameterization and sampling ────────────────────────────────────────

    def _sample_t(self, batch_size, num_blocks, device):
        # Antithetic timestep sampling per block; mask x0 → xt (bd3lms diffusion.py _sample_t)
        t = torch.rand((batch_size, num_blocks), device=device)
        if self.antithetic_sampling:
            offset = torch.arange(batch_size * num_blocks, device=device).float()
            offset = (offset / (batch_size * num_blocks)).view(batch_size, num_blocks)
            t = (t / (batch_size * num_blocks) + offset) % 1.0
        t = t.repeat_interleave(self.block_size, dim=-1)
        return t * (self.sampling_eps_max - self.sampling_eps_min) + self.sampling_eps_min
    
    def _subs_parameterization(self, logits, xt):
        '''Substitution parameterization (bd3lms diffusion.py _subs_parameterization).
        Forces the model to predict the original token, not MASK, and preserves already-unmasked positions.
        '''
        logits = logits.clone()  # avoid in-place modification of lm_head's output tensor
        logits[..., self.mask_token_id] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked = xt != self.mask_token_id
        logits[unmasked] = self.neg_infinity
        logits[unmasked, xt[unmasked]] = 0.0
        return logits

    def _nucleus_sample(self, p_x0):
        '''Top-p nucleus filtering on the last block_size positions (bd3lms).
        p_x0: (B, L, vocab_size+1) probabilities (not log).
        Only the last block_size columns are filtered in-place.
        '''
        if self.nucleus_p >= 1.0: return p_x0
        p_x0_ = p_x0[:, -self.block_size:].clone()
        sorted_probs, sorted_indices = p_x0_.sort(dim=-1, descending=True)
        cum_probs = sorted_probs.cumsum(dim=-1)
        nucleus_mask = cum_probs <= self.nucleus_p
        nucleus_mask[..., 0] = True   # always keep top token
        sorted_probs = sorted_probs * nucleus_mask
        p_x0_.scatter_(-1, sorted_indices, sorted_probs)
        p_x0_ /= p_x0_.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        p_x0[:, -self.block_size:] = p_x0_
        return p_x0

    # ── Training forward pass ─────────────────────────────────────────────────

    def forward(self, **kwargs):
        '''BD3LM training forward pass (dLLM A2D) — matches TranslationNetwork interface.

        Implements the BD3LM training from dLLM (arxiv.org/abs/2602.22661):
          1. Build x0 (clean target), pad to multiple of block_size.
          2. Sample per-block noise and create xt (noised tokens).
          3. Concatenate [xt | x0] → (B, 2L) with BD3LM attention mask.
          4. Repeated position IDs: [0..L-1, 0..L-1].
          5. Take first L logits (xt positions) for loss computation.
          6. Cross-entropy weighted by 1/t at masked positions only.

        Expected kwargs:
            input_feature: (B, T_v, D) visual features from VLMapper.
            input_lengths: (B,) valid lengths of visual feature sequences.
            labels: (B, L_text) target token IDs (-100 for HF ignore positions).
            decoder_input_ids: (B, L_text) BOS-shifted inputs (provides BOS token).
        Returns:
            dict with 'translation_loss' (scalar).
        '''
        input_feature = kwargs['input_feature']
        input_lengths = kwargs['input_lengths']
        labels = kwargs['labels']
        decoder_input_ids = kwargs.get('decoder_input_ids', None)
        B, device = input_feature.shape[0], input_feature.device

        # ── 1. Build x0 (clean target): [BOS, label_tokens...] ───────────────
        x0 = labels.clone().to(device)
        x0[x0 == -100] = self.pad_index
        if decoder_input_ids is not None: bos = decoder_input_ids[:, 0:1].to(device)
        else: bos = torch.full((B, 1), self.bos_index, dtype=x0.dtype, device=device)
        x0 = torch.cat([bos, x0], dim=1)             # (B, L+1)

        # Text attention mask: 1 for real tokens, 0 for padding
        text_mask = (x0 != self.pad_index)           # (B, L+1) bool
        if self.ignore_bos: text_mask[:, 0] = False  # BOS never masked

        # Align length to a multiple of block_size
        L = x0.shape[1]
        num_blocks = max(1, math.ceil(L / self.block_size))
        L_aligned = num_blocks * self.block_size
        if L < L_aligned:
            x0 = F.pad(x0, (0, L_aligned - L), value=self.pad_index)
            text_mask = F.pad(text_mask, (0, L_aligned - L), value=False)
        else:
            x0 = x0[:, :L_aligned]
            text_mask = text_mask[:, :L_aligned]
        L = L_aligned

        # ── 2. Sample noise and create xt (noised tokens) ────────────────────
        t = self._sample_t(B, num_blocks, device) # (B, L) per-block t; mask x0 → xt
        _, p = self.noise(t)                      # loglinear schedule: p=t (mask probability)
        rand = torch.rand_like(x0.float())
        masked_mask = (rand < p) & text_mask      # (B, L) bool, True = masked AND valid text position
        xt = torch.where(masked_mask, self.mask_token_id, x0)

        # ── 3. Decoder forward with BD3LM mask ───────────────────────────────
        enc_hidden, enc_mask = self._encode_visual(input_feature, input_lengths)
        
        # BD3LM attention mask: (1, 1, 2L, 2L)
        bd3lm_mask = build_bd3lm_mask(L, self.block_size, enc_hidden.dtype, device)

        # Repeated position IDs: [0..L-1, 0..L-1]
        base_pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        position_ids = torch.cat([base_pos, base_pos], dim=1)  # (B, 2L)
        
        # Sigma: not used for mBART A2D (dLLM BD3LMTrainer does not pass sigma).
        sigma = None # TimestepEmbedder is DDiT-specific (AdaLN); for mBART it adds noise.

        # BD3LM forward: [xt | x0] with 3-component mask
        logits = self._decode(
            torch.cat([xt, x0], dim=1), # (B, 2L), concatenate [xt | x0] with SHARED positional embeddings
            enc_hidden, enc_mask, self_attn_mask=bd3lm_mask,
            position_ids=position_ids, sigma=sigma
        )  # (B, 2L, V+1)
        logits = logits[:, :L]  # (B, L, V+1), take only first L logits (xt half)

        # ── 4. Compute weighted cross-entropy loss ────────────────────────────
        loss_weights = 1.0 / t.clamp(min=1e-6)  # (B, L), 1/t per block (MDLM loglinear schedule)
        
        # Substitution parameterization is simply equivalent to cross-entropy
        # with targets = x0 and logits masked to force MASK prediction at masked positions.
        token_nll = F.cross_entropy( 
            logits.transpose(1, 2),             # (B, V+1, L)
            x0,                                 # (B, L) — targets
            reduction='none',                   # (B, L)
        )
        # Mask: only count loss at masked & maskable positions
        loss_mask = masked_mask.float()         # (B, L)
        weighted_nll = token_nll * loss_weights * loss_mask  # Zero out unmasked positions

        # Normalize by total maskable tokens (dLLM "token" normalization)
        translation_loss = weighted_nll.sum() / loss_mask.sum().clamp(min=1)
        return {'translation_loss': translation_loss}

    # ── Inference ─────────────────────────────────────────────────────────────
    
    def _sigma_from_p(self, p): # Diffusion utilities (noise schedule helpers)
        # Convert masking probability p to sigma (bd3lms diffusion._sigma_from_p)
        sigma_max = self.noise.sigma_max.to(device=p.device, dtype=p.dtype)
        return torch.min(-torch.log1p(-p.clamp(max=1.0 - 1e-7)), sigma_max)

    @torch.no_grad()
    def _ddpm_caching_update(self, x, t, dt, enc_hidden, enc_mask, p_x0=None):
        '''One BD3LM DDPM denoising step (bd3lms._ddpm_caching_update, A2D variant).

        Uses full forward pass with block-causal mask (correct dLLM A2D approach).
        KV caching is DDiT-specific and not applicable to mBART A2D.

        Args:
            x:          (B, context_len) tokens; last block_size positions = current block.
            t:          (B, 1) current timestep in [0, 1].
            dt:         float, timestep decrement (1 / num_steps_per_block).
            enc_hidden: visual encoder hidden states.
            enc_mask:   visual encoder padding mask.
            p_x0:       cached model output from previous step (None = recompute).

        Returns:
            p_x0:  model output reused next step if unchanged, else None.
            x_new: updated context window.
        '''
        _, move_chance_t = self.noise(t)                           # (B, 1)
        _, move_chance_s = self.noise((t - dt).clamp(min=1e-10))   # (B, 1)
        alpha_t = move_chance_t[:, :, None]                        # (B, 1, 1)
        alpha_s = move_chance_s[:, :, None]                        # (B, 1, 1)
        mask_prob = alpha_s / alpha_t.clamp(min=1e-10)             # (B, 1, 1)
        cur_block = x[:, -self.block_size:]                        # (B, block_size)

        if p_x0 is None:
            sigma_t = self._sigma_from_p(move_chance_t) if self.time_conditioning else None
            logits = self._decode(x, enc_hidden, enc_mask, sigma=sigma_t)  # (B, context_len, V+1)
            logits = logits[:, -self.block_size:]                          # (B, block_size, V+1)
            logits = self._subs_parameterization(logits, cur_block)
            p_x0 = logits.float().exp()
            p_x0 = self._nucleus_sample(p_x0)

        if self.first_hitting: # Unmask exactly one randomly-chosen masked position per sample
            x_block_pred = _sample_categorical(p_x0)               # (B, block_size)
            x_block = cur_block.clone()
            for b in range(x_block.shape[0]):
                masked_pos = (cur_block[b] == self.mask_token_id).nonzero(as_tuple=True)[0]
                if len(masked_pos) > 0:
                    pick = torch.randint(len(masked_pos), (1,), device=x.device).item()
                    x_block[b, masked_pos[pick]] = x_block_pred[b, masked_pos[pick]]
        else: # DDPM transition: q(xs|xt,x0) (bd3lms eq.)
            q_xs = p_x0 * (1 - mask_prob)                          # (B, block_size, V+1)
            q_xs[..., self.mask_token_id] = mask_prob.squeeze(-1)  # broadcast (B,1)→(B,block)
            x_block = _sample_categorical(q_xs)

        # Preserve already-unmasked tokens
        copy_flag = (cur_block != self.mask_token_id)
        x_block = torch.where(copy_flag, cur_block, x_block)
        x_new = torch.cat([x[:, :-self.block_size], x_block], dim=-1)

        # Cache p_x0 only if tokens didn't change (avoid redundant forward pass)
        return (None, x_new) if not torch.equal(x_new, x) else (p_x0, x_new)


    @torch.no_grad()
    def generate(self, input_feature=None, input_lengths=None, max_length=100, diffusion_steps=128, **kwargs):
        '''BD3LM semi-AR block diffusion inference (dLLM A2D for mBART).

        Faithful adaptation of bd3lms._semi_ar_sampler() for mBART:
          - Full forward pass with block-causal mask each step (dLLM A2D)
          - DDPM transitions or first-hitting sampler (controlled by first_hitting)
          - p_x0 caching: reuse model output when no tokens changed
          - Nucleus sampling: top-p filtering (controlled by nucleus_p)
        '''
        B = input_feature.shape[0]
        device = input_feature.device
        enc_hidden, enc_mask = self._encode_visual(input_feature, input_lengths)

        num_blocks_max = max(1, max_length // self.block_size)
        num_steps = max(1, diffusion_steps // num_blocks_max)  # steps per block
        dt = 1.0 / num_steps
        ones = torch.ones((B, 1), device=device)

        # First block: position 0 = BOS (never masked), rest = MASK
        x_accum = torch.full((B, self.block_size), self.mask_token_id, dtype=torch.long, device=device)
        x_accum[:, 0] = self.bos_index
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for stride_num in range(num_blocks_max):
            if finished.all(): break
            if stride_num > 0:
                new_block = torch.full((B, self.block_size), self.mask_token_id, dtype=torch.long, device=device)
                x_accum = torch.cat([x_accum, new_block], dim=1)

            end_idx = (stride_num + 1) * self.block_size
            start_idx = max(end_idx - 1024, 0)  # Context window (≤1024 tokens)
            p_x0_cache, t_val = None, 1.0
            timesteps = torch.linspace(1.0, 0.0, num_steps, device=device)

            for step in range(num_steps):
                context = x_accum[:, start_idx:end_idx]  # (B, context_len)
                cur_blk = context[:, -self.block_size:]
                if not (cur_blk == self.mask_token_id).any(): break  # Block fully denoised

                if self.first_hitting: # Geometric timestep: t *= u^(1/num_masked)
                    u = torch.rand(1, device=device).item()
                    n_masked = float((cur_blk == self.mask_token_id).sum())
                    n_per_sample = max(n_masked / B, 1.0)
                    t_val = t_val * (u ** (1.0 / n_per_sample))
                else:
                    t_val = float(timesteps[step].item())

                t_tensor = max(t_val, 1e-10) * ones  # (B, 1)
                p_x0_cache, context_new = self._ddpm_caching_update(
                    x=context, t=t_tensor, dt=dt,
                    enc_hidden=enc_hidden, enc_mask=enc_mask, p_x0=p_x0_cache,
                )
                x_accum[:, start_idx:end_idx] = context_new

            # Check for EOS in the newly finished block
            cur_blk_final = x_accum[:, end_idx - self.block_size:end_idx]
            for b in range(B):
                if not finished[b] and (cur_blk_final[b] == self.eos_index).any(): finished[b] = True
        return {'sequences': x_accum}