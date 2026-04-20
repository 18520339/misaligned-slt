'''Block Diffusion Decoder for conditional sign language translation.

Implements BD3LM (block diffusion) adapted to mBART via the A2D recipe from
dLLM (arxiv.org/abs/2602.22661), with full bd3lms fidelity:
  - Architecture: pretrained mBART decoder with block-causal self-attention.
  - Training: BD3LM masked diffusion loss with [xt | x0] concatenation,
              BD3LM attention mask (M_BD + M_OBC + M_BC), repeated position IDs,
              and cross-entropy weighted by 1/t at masked positions.
  - Inference: dLLM-style BD3LM semi-AR sampling with confidence-based remasking,
               temperature-controlled Gumbel-max, block-by-block denoising.

Adapted for GFSLT-VLP encoder pipeline:
  PoseFeatureExtractor (CoSign) → VisualEncoder → mBART encoder → cross-attention

References:
  - dLLM paper + A2D recipe: https://arxiv.org/pdf/2602.22661
  - BD3LM paper: https://arxiv.org/pdf/2503.09573
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import AttentionMaskConverter


def build_bd3lm_mask(seq_len, block_size, dtype, device):
    '''BD3LM training attention mask for concatenated [xt | x0] input.

    For input of length 2*seq_len, creates a (1, 1, 2L, 2L) mask with 3 components:
    - M_BD:  xt_block_b <-> all xt in same block b  (bidirectional within block)
    - M_OBC: xt_block_b -> x0_block_{0..b-1}       (strictly prior clean blocks)
    - M_BC:  x0_block_b -> x0_block_{0..b}          (block-causal over clean copy)

    Returns:
        (1, 1, 2L, 2L) float mask: 0 = attend, -inf = masked.
    '''
    n = seq_len
    idx = torch.arange(2 * n, device=device)
    q_idx = idx[:, None]
    kv_idx = idx[None, :]

    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n

    block_q = torch.where(x0_flag_q, (q_idx - n) // block_size, q_idx // block_size)
    block_kv = torch.where(x0_flag_kv, (kv_idx - n) // block_size, kv_idx // block_size)

    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
    offset_block_causal = (block_q > block_kv) & x0_flag_kv & ~x0_flag_q
    block_causal = (block_q >= block_kv) & x0_flag_kv & x0_flag_q

    can_attend = block_diagonal | offset_block_causal | block_causal
    mask = torch.zeros(2 * n, 2 * n, dtype=dtype, device=device)
    mask = mask.masked_fill(~can_attend, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0)


def build_block_causal_mask(batch_size, tgt_len, block_size, dtype, device):
    '''Block-causal (staircase) attention mask for inference.

    Returns:
        (B, 1, T, T) float mask: 0 = attend, -inf = masked.
    '''
    positions = torch.arange(tgt_len, device=device)
    block_ids = positions // block_size
    can_attend = block_ids.view(1, tgt_len) <= block_ids.view(tgt_len, 1)
    mask = torch.zeros(tgt_len, tgt_len, dtype=dtype, device=device)
    mask = mask.masked_fill(~can_attend, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, tgt_len, tgt_len)


class BlockDiffusionDecoder(nn.Module):
    '''BD3LM decoder built on pretrained mBART, adapted for GFSLT-VLP encoder.

    Replaces the AR mBART decoder with a block diffusion decoder that:
      - Shares the same mBART pretrained weights (encoder + decoder layers).
      - Replaces causal decoder self-attention with block-causal mask.
      - Trains with MDLM masked diffusion loss (loglinear noise schedule).
      - Generates via iterative denoising block-by-block at inference.

    Cross-attention to visual features is inherited from mBART decoder layers.
    '''
    def __init__(
        self, mbart, backbone, sign_emb, embed_scale, tokenizer,
        block_size=4, sampling_eps_min=1e-3, sampling_eps_max=1.0,
        antithetic_sampling=True, ignore_bos=True, temperature=0.0,
        remasking='low_confidence', steps_per_block=None,
    ):
        super().__init__()
        self.block_size = block_size
        self.sampling_eps_min = sampling_eps_min
        self.sampling_eps_max = sampling_eps_max
        self.antithetic_sampling = antithetic_sampling
        self.ignore_bos = ignore_bos

        # Inference params
        self.temperature = temperature
        self.remasking = remasking
        self.steps_per_block = steps_per_block

        # Visual pipeline components (shared with SLTModel, stored as non-module refs)
        # These are NOT registered as submodules to avoid duplicate parameters.
        # They are accessed via the parent SLTModel for forward/generate.
        self._backbone = backbone
        self._sign_emb = sign_emb
        self._embed_scale = embed_scale

        # mBART components
        self.mbart_encoder = mbart.model.encoder
        self.mbart_decoder = mbart.model.decoder
        self.d_model = mbart.config.d_model

        # Tokenizer info (HuggingFace tokenizer)
        self.pad_index = tokenizer.pad_token_id
        self.eos_index = tokenizer.eos_token_id
        self.bos_index = tokenizer.bos_token_id
        vocab_size = mbart.config.vocab_size
        self.vocab_size = vocab_size

        # Extend vocabulary with MASK token
        self.mask_token_id = vocab_size
        self.mask_index = self.mask_token_id

        # Extend embedding and lm_head
        old_embed = mbart.model.shared
        old_lm = mbart.lm_head
        d_model = old_embed.embedding_dim
        self.embed_tokens = nn.Embedding(vocab_size + 1, d_model, padding_idx=self.pad_index)
        self.lm_head = nn.Linear(d_model, vocab_size + 1, bias=False)
        with torch.no_grad():
            self.embed_tokens.weight[:vocab_size].copy_(old_embed.weight)
            self.lm_head.weight[:vocab_size].copy_(old_lm.weight)
            nn.init.normal_(self.embed_tokens.weight[vocab_size:], std=0.02)
            nn.init.zeros_(self.lm_head.weight[vocab_size:])

        self.mbart_decoder.embed_tokens = self.embed_tokens
        self.embed_scale_factor = math.sqrt(d_model)
        print(
            f'BlockDiffusionDecoder (mBART A2D): d_model={d_model}, vocab={vocab_size}+1(MASK), '
            f'block_size={block_size}, remasking={remasking}, temperature={temperature}')

    # ── Visual encoder ────────────────────────────────────────────────────────

    def _prepare_visual(self, pixel_values, pixel_mask):
        '''Visual encoding: backbone → sign_emb → mBART encoder.'''
        frame_features, attention_mask = self._backbone(pixel_values, pixel_mask)
        inputs_embeds = self._sign_emb(frame_features)
        inputs_embeds = self._embed_scale * inputs_embeds
        return inputs_embeds, attention_mask

    def _encode_visual(self, pixel_values, pixel_mask):
        inputs_embeds, attention_mask = self._prepare_visual(pixel_values, pixel_mask)
        enc_out = self.mbart_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return enc_out.last_hidden_state, attention_mask

    # ── Decoder with block-causal / bd3lm self-attention ──────────────────────

    def _decode(self, decoder_input_ids, enc_hidden, enc_mask,
                self_attn_mask=None, position_ids=None):
        '''Run mBART decoder layers with custom self-attention mask.'''
        batch_size, tgt_len = decoder_input_ids.shape
        device = decoder_input_ids.device
        dtype = enc_hidden.dtype
        inputs_embeds = self.embed_tokens(decoder_input_ids) * self.embed_scale_factor

        if position_ids is not None:
            positions = self.mbart_decoder.embed_positions.weight[position_ids + 2]
        else:
            positions = self.mbart_decoder.embed_positions(decoder_input_ids)

        hidden = inputs_embeds + positions
        hidden = self.mbart_decoder.layernorm_embedding(hidden)
        hidden = F.dropout(hidden, p=self.mbart_decoder.dropout, training=self.training)

        if self_attn_mask is None:
            self_mask = build_block_causal_mask(batch_size, tgt_len, self.block_size, dtype, device)
        else:
            self_mask = self_attn_mask.to(dtype=dtype, device=device)

        cross_mask = (
            AttentionMaskConverter._expand_mask(enc_mask, dtype, tgt_len=tgt_len)
            if enc_mask is not None else None
        )

        for layer in self.mbart_decoder.layers:
            hidden = layer(
                hidden, attention_mask=self_mask,
                encoder_hidden_states=enc_hidden,
                encoder_attention_mask=cross_mask,
            )[0]

        if getattr(self.mbart_decoder, 'layer_norm', None) is not None:
            hidden = self.mbart_decoder.layer_norm(hidden)
        return self.lm_head(hidden)

    # ── Timestep sampling ────────────────────────────────────────────────────

    def _sample_t(self, batch_size, num_blocks, device):
        t = torch.rand((batch_size, num_blocks), device=device)
        if self.antithetic_sampling:
            offset = torch.arange(batch_size * num_blocks, device=device).float()
            offset = (offset / (batch_size * num_blocks)).view(batch_size, num_blocks)
            t = (t / (batch_size * num_blocks) + offset) % 1.0
        t = t.repeat_interleave(self.block_size, dim=-1)
        return t * (self.sampling_eps_max - self.sampling_eps_min) + self.sampling_eps_min

    # ── Training forward pass ─────────────────────────────────────────────────

    def forward(self, pixel_values, pixel_mask, labels, **kwargs):
        '''BD3LM training forward pass.

        Args:
            pixel_values: (B, T, 77, 3) pose sequences.
            pixel_mask: (B, T) attention mask.
            labels: (B, L) target token IDs (pad_token_id for padding).
        Returns:
            dict with 'loss'.
        '''
        B, device = pixel_values.shape[0], pixel_values.device

        # 1. Build x0 (clean target): [BOS, label_tokens...]
        x0 = labels.clone().to(device)
        # Replace pad with pad_index for consistency (labels from HF tokenizer already use pad_token_id)
        bos = torch.full((B, 1), self.bos_index, dtype=x0.dtype, device=device)
        x0 = torch.cat([bos, x0], dim=1)

        text_mask = (x0 != self.pad_index)
        if self.ignore_bos:
            text_mask[:, 0] = False

        # Align to block_size multiple
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

        # 2. Sample noise and create xt
        t = self._sample_t(B, num_blocks, device)
        rand = torch.rand_like(x0.float())
        masked_mask = (rand < t) & text_mask
        xt = torch.where(masked_mask, self.mask_token_id, x0)

        # 3. Encode visual features
        enc_hidden, enc_mask = self._encode_visual(pixel_values, pixel_mask)

        # 4. BD3LM forward: [xt | x0] with 3-component mask
        bd3lm_mask = build_bd3lm_mask(L, self.block_size, enc_hidden.dtype, device)
        base_pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        position_ids = torch.cat([base_pos, base_pos], dim=1)

        logits = self._decode(
            torch.cat([xt, x0], dim=1),
            enc_hidden, enc_mask,
            self_attn_mask=bd3lm_mask,
            position_ids=position_ids,
        )
        logits = logits[:, :L]  # Take xt-half logits

        # 5. Weighted cross-entropy loss
        loss_weights = 1.0 / t.clamp(min=1e-6)
        token_nll = F.cross_entropy(
            logits.transpose(1, 2), x0, reduction='none',
        )
        loss_mask = masked_mask.float()
        weighted_nll = token_nll * loss_weights * loss_mask
        loss = weighted_nll.sum() / loss_mask.sum().clamp(min=1)
        return {'loss': loss}

    # ── Inference ─────────────────────────────────────────────────────────────

    @staticmethod
    def _add_gumbel_noise(logits, temperature):
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    @staticmethod
    def _get_num_transfer_tokens(mask_index, steps):
        mask_num = mask_index.sum(dim=1, keepdim=True)
        B = mask_num.size(0)
        device = mask_index.device
        num_transfer = torch.zeros(B, steps, dtype=torch.int64, device=device)

        for i in range(B):
            remaining = mask_num[i, 0].clone()
            for j in range(steps):
                t = (steps - j) / steps
                s = (steps - j - 1) / steps
                if t <= 0:
                    break
                reverse_transfer_prob = 1.0 - (s / t)
                k = torch.round(remaining.float() * reverse_transfer_prob).to(torch.int64)
                k = torch.clamp(k, min=0, max=remaining)
                num_transfer[i, j] = k
                remaining -= k
                if remaining <= 0:
                    break

        rows, max_len = [], 0
        for i in range(B):
            nonzero = num_transfer[i][num_transfer[i] > 0]
            rows.append(nonzero)
            max_len = max(max_len, nonzero.numel())
        return torch.stack([
            torch.cat([r, torch.zeros(max_len - r.numel(), dtype=r.dtype, device=r.device)])
            if r.numel() < max_len else r for r in rows
        ], dim=0)

    def _diffusion_step_block(self, logits, x_block, mask_block, num_transfer_step):
        B, L, _ = logits.shape
        device = logits.device
        if not mask_block.any():
            return x_block

        logits_noisy = self._add_gumbel_noise(logits, self.temperature)
        x0 = torch.argmax(logits_noisy, dim=-1)

        if self.remasking == 'low_confidence':
            p = F.softmax(logits.float(), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        elif self.remasking == 'random':
            x0_p = torch.rand((B, L), device=device)
        else:
            raise ValueError(f'Unknown remasking: {self.remasking}')

        x0 = torch.where(mask_block, x0, x_block)
        neg_inf = torch.full_like(x0_p, -float('inf'))
        confidence = torch.where(mask_block, x0_p, neg_inf)

        transfer = torch.zeros_like(x0, dtype=torch.bool)
        for j in range(B):
            k = int(num_transfer_step[j].item())
            if k <= 0:
                continue
            valid_count = (confidence[j] > -float('inf')).sum().item()
            if valid_count == 0:
                continue
            k = min(k, valid_count)
            _, sel = torch.topk(confidence[j], k)
            transfer[j, sel] = True

        x_new = x_block.clone()
        x_new[transfer] = x0[transfer]
        return x_new

    @torch.no_grad()
    def generate(self, pixel_values=None, pixel_mask=None,
                 max_length=100, diffusion_steps=128, **kwargs):
        '''BD3LM block diffusion inference.'''
        B = pixel_values.shape[0]
        device = pixel_values.device
        enc_hidden, enc_mask = self._encode_visual(pixel_values, pixel_mask)

        num_blocks = max(1, max_length // self.block_size)
        spb = self.steps_per_block or max(1, diffusion_steps // num_blocks)

        x = torch.full((B, self.block_size), self.mask_token_id, dtype=torch.long, device=device)
        x[:, 0] = self.bos_index
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for b_idx in range(num_blocks):
            if finished.all():
                break
            if b_idx > 0:
                new_block = torch.full((B, self.block_size), self.mask_token_id,
                                       dtype=torch.long, device=device)
                x = torch.cat([x, new_block], dim=1)
            cur_len = self.block_size

            block_mask = (x[:, -cur_len:] == self.mask_token_id)
            num_transfer = self._get_num_transfer_tokens(block_mask, spb)
            effective_steps = num_transfer.shape[1]

            for i_step in range(effective_steps):
                x_block = x[:, -cur_len:]
                mask_block = (x_block == self.mask_token_id)
                if not mask_block.any():
                    break
                logits = self._decode(x, enc_hidden, enc_mask)
                logits_block = logits[:, -cur_len:]
                x_block_new = self._diffusion_step_block(
                    logits_block, x_block, mask_block, num_transfer[:, i_step])
                x[:, -cur_len:] = x_block_new

            if self.eos_index is not None:
                eos_in_block = (x[:, -cur_len:] == self.eos_index).any(dim=1)
                finished = finished | eos_in_block
        return x
