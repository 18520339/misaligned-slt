'''Model factory for Phase 2: builds AR or BD model variants.

Creates a SignLanguageModel with either:
  - AR decoder (original MSKA TranslationNetwork)
  - BD decoder (BlockDiffusionDecoder with cross-attention)

Both share the same Recognition network and VLMapper from MSKA.
'''
import os, sys, copy, math
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'MSKA'))

from model import SignLanguageModel
from block_diffusion import BlockDiffusionDecoder


class SLTModel(torch.nn.Module): # Unified SLT model supporting both AR and BD decoders.
    # Wraps MSKA's Recognition + VLMapper + a decoder (AR or BD). Provides a consistent interface for training and evaluation
    def __init__(self, mska_cfg, args, model_cfg, decoder_type='ar'):
        super().__init__()
        self.decoder_type = decoder_type
        self.mska_cfg = mska_cfg
        self.mska_model = SignLanguageModel(cfg=mska_cfg, args=args) # Build full MSKA model to get Recognition + VLMapper + tokenizers

        # Keep references for convenience
        self.recognition_network = self.mska_model.recognition_network
        self.vl_mapper = self.mska_model.vl_mapper
        self.gloss_tokenizer = self.mska_model.gloss_tokenizer
        self.text_tokenizer = self.mska_model.text_tokenizer

        # Loss weights
        model_section = mska_cfg.get('model', {})
        self.recognition_weight = model_section.get('recognition_weight', 1.0)
        self.translation_weight = model_section.get('translation_weight', 1.0)

        if decoder_type == 'ar': # Use MSKA's original TranslationNetwork
            self.translation_network = self.mska_model.translation_network
            self.bd_decoder = None
        elif decoder_type == 'bd': # Replace TranslationNetwork with BlockDiffusionDecoder
            bd_cfg = model_cfg.get('block_diffusion', {})

            # Get vocab size from MSKA's text tokenizer
            if hasattr(self.text_tokenizer, 'pruneids'): vocab_size = max(self.text_tokenizer.pruneids.values()) + 1
            elif hasattr(self.text_tokenizer, 'id2token'): vocab_size = len(self.text_tokenizer.id2token)
            else: vocab_size = 1024  # fallback

            d_model = self.mska_model.translation_network.input_dim # Get d_model from VLMapper output (= mBART d_model)
            pad_index = self.text_tokenizer.pad_index
            eos_index = self.text_tokenizer.eos_index

            self.bd_decoder = BlockDiffusionDecoder(
                vocab_size=vocab_size, 
                d_model=d_model,
                n_heads=bd_cfg.get('n_heads', 16),
                n_layers=bd_cfg.get('n_layers', 6),
                cond_dim=bd_cfg.get('cond_dim', 128),
                block_size=bd_cfg.get('block_size', 4),
                mlp_ratio=bd_cfg.get('mlp_ratio', 4),
                dropout=bd_cfg.get('dropout', 0.1),
                max_seq_len=bd_cfg.get('max_seq_len', 128),
                pad_index=pad_index, 
                eos_index=eos_index,
                sampling_eps_min=bd_cfg.get('sampling_eps_min', 1e-3),
                sampling_eps_max=bd_cfg.get('sampling_eps_max', 1.0),
                antithetic_sampling=bd_cfg.get('antithetic_sampling', True),
            )

            # Remove MSKA's translation network to save memory
            del self.mska_model.translation_network
            self.translation_network = None
        else: raise ValueError(f'Unknown decoder_type: {decoder_type}')
        

    def load_pretrained(self, checkpoint_path, strict=False): # Load pretrained MSKA checkpoint (Recognition + VLMapper + AR decoder)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        if self.decoder_type == 'ar': ret = self.mska_model.load_state_dict(state_dict, strict=strict)
        else: # For BD: load Recognition + VLMapper, skip TranslationNetwork
            filtered = {}
            for k, v in state_dict.items():
                if 'translation_network' not in k: filtered[k] = v
            ret = self.mska_model.load_state_dict(filtered, strict=False)

        print(f'Loaded checkpoint from {checkpoint_path}')
        if ret.missing_keys:
            print(f'  Missing keys: {len(ret.missing_keys)}')
            for k in ret.missing_keys[:10]: print(f'    {k}')
            if len(ret.missing_keys) > 10: print(f'    ... and {len(ret.missing_keys) - 10} more')
        if ret.unexpected_keys: print(f'  Unexpected keys: {len(ret.unexpected_keys)}')
        return ret


    def forward(self, src_input): # Full forward pass: Recognition -> VLMapper -> Decoder
        recognition_outputs = self.recognition_network(src_input) # Recognition
        mapped_feature = self.vl_mapper(visual_outputs=recognition_outputs) # VLMapper

        if self.decoder_type == 'ar': # AR: use MSKA's TranslationNetwork
            translation_inputs = {
                **src_input['translation_inputs'],
                'input_feature': mapped_feature,
                'input_lengths': recognition_outputs['input_lengths'],
            }
            translation_outputs = self.translation_network(**translation_inputs)
            model_outputs = {**translation_outputs, **recognition_outputs}
            model_outputs['transformer_inputs'] = translation_outputs['transformer_inputs']
            model_outputs['total_loss'] = (
                self.recognition_weight * recognition_outputs['recognition_loss']
                + self.translation_weight * translation_outputs['translation_loss']
            )
        else: # BD: use BlockDiffusionDecoder
            translation_inputs = {
                'input_feature': mapped_feature,
                'input_lengths': recognition_outputs['input_lengths'],
                **src_input['translation_inputs'],
            }
            translation_outputs = self.bd_decoder(**translation_inputs)
            model_outputs = {**recognition_outputs}
            model_outputs['translation_loss'] = translation_outputs['translation_loss']
            model_outputs['total_loss'] = (
                self.recognition_weight * recognition_outputs['recognition_loss']
                + self.translation_weight * translation_outputs['translation_loss']
            )
            # Store for generate_txt
            model_outputs['transformer_inputs'] = {'input_feature': mapped_feature, 'input_lengths': recognition_outputs['input_lengths']}
        return model_outputs


    @torch.no_grad()
    def generate_txt(self, transformer_inputs=None, generate_cfg=None, **kwargs):
        '''Generate text translations.

        For AR: uses MSKA's beam search generation
        For BD: uses block diffusion semi-AR sampling
        '''
        if generate_cfg is None: generate_cfg = {}
        if self.decoder_type == 'ar': return self.translation_network.generate(**transformer_inputs, **generate_cfg)
        else: # BD inference
            diffusion_steps = generate_cfg.get('diffusion_steps', 10)
            max_length = generate_cfg.get('max_length', 100)
            output = self.bd_decoder.generate(
                input_feature=transformer_inputs['input_feature'],
                input_lengths=transformer_inputs['input_lengths'],
                max_length=max_length, diffusion_steps=diffusion_steps,
            )
            # Decode token IDs to text
            sequences = output['sequences']
            decoded = self._decode_sequences(sequences)
            output['decoded_sequences'] = decoded
            return output


    def _decode_sequences(self, sequences): # Convert token IDs to text strings
        decoded = []
        for seq in sequences:
            tokens = []
            for tok_id in seq:
                tok_id = tok_id.item()
                # Stop at EOS or MASK
                if tok_id == self.text_tokenizer.eos_index: break
                if tok_id == self.bd_decoder.mask_index: continue
                if tok_id == self.text_tokenizer.pad_index: continue
                tokens.append(tok_id)

            if hasattr(self.text_tokenizer, 'level') and self.text_tokenizer.level == 'word':
                text = ' '.join(self.text_tokenizer.id2token[t] for t in tokens if t < len(self.text_tokenizer.id2token))
            else: # Sentencepiece: need to reverse prune
                tok_tensor = torch.tensor([tokens], dtype=torch.long)
                if hasattr(self.text_tokenizer, 'prune_reverse'): tok_tensor = self.text_tokenizer.prune_reverse(tok_tensor)
                text = self.text_tokenizer.tokenizer.batch_decode(tok_tensor, skip_special_tokens=True)[0]

            # Clean up German punctuation (matching MSKA's TextTokenizer.batch_decode)
            if hasattr(self.text_tokenizer, 'tokenizer') and \
               hasattr(self.text_tokenizer.tokenizer, 'tgt_lang') and \
               'de' in self.text_tokenizer.tokenizer.tgt_lang:
                if len(text) > 2 and text[-1] == '.' and text[-2] != ' ': text = text[:-1] + ' .'
            decoded.append(text)
        return decoded