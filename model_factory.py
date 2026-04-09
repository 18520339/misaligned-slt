'''Model factory for Phase 2: builds AR or BD model variants.

Creates an SLTModel with either:
  - AR decoder (original MSKA TranslationNetwork)
  - BD decoder (BlockDiffusionDecoder with cross-attention)

Both share the same Recognition network and VLMapper from MSKA.
'''
import os, sys, torch

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

        # Loss weights — read from model_cfg training section (Phase 2 configs)
        train_section = model_cfg.get('training', {})
        self.recognition_weight = train_section.get('recognition_weight', 1.0)
        self.translation_weight = train_section.get('translation_weight', 1.0)

        if decoder_type == 'ar': # Use MSKA's original TranslationNetwork
            self.translation_network = self.mska_model.translation_network
            self.bd_decoder = None
        elif decoder_type == 'bd': # Replace TranslationNetwork with BlockDiffusionDecoder
            bd_cfg = model_cfg.get('block_diffusion', {})

            # A2D (AR-to-Diffusion): build BD decoder using pretrained mBART weights.
            # translation_network is passed so BlockDiffusionDecoder can extract its
            # pretrained encoder/decoder layers, embeddings, and prepare_feature_inputs.
            print('  BD decoder (A2D): building from pretrained mBART in TranslationNetwork...')
            self.bd_decoder = BlockDiffusionDecoder(
                translation_network=self.mska_model.translation_network,
                block_size=bd_cfg.get('block_size', 4),
                sampling_eps_min=bd_cfg.get('sampling_eps_min', 1e-3),
                sampling_eps_max=bd_cfg.get('sampling_eps_max', 1.0),
                antithetic_sampling=bd_cfg.get('antithetic_sampling', True),
                ignore_bos=bd_cfg.get('ignore_bos', True),
                first_hitting=bd_cfg.get('first_hitting', True),
                nucleus_p=bd_cfg.get('nucleus_p', 1.0),
                time_conditioning=bd_cfg.get('time_conditioning', False)
            )
            # Remove the original TranslationNetwork from mska_model to avoid
            # duplicate parameters (bd_decoder already holds refs to its submodules).
            del self.mska_model.translation_network
            self.translation_network = None
        else: raise ValueError(f'Unknown decoder_type: {decoder_type}')


    def load_pretrained(self, checkpoint_path, strict=False): # Load pretrained MSKA checkpoint (Recognition + VLMapper + AR decoder)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        if self.decoder_type == 'ar': ret = self.mska_model.load_state_dict(state_dict, strict=strict)
        else: # For BD: load Recognition + VLMapper, skip TranslationNetwork
            filtered = {k: v for k, v in state_dict.items() if 'translation_network' not in k}
            ret = self.mska_model.load_state_dict(filtered, strict=False)

        print(f'Loaded checkpoint from {checkpoint_path}')
        if ret.missing_keys:
            print(f'  Missing keys: {len(ret.missing_keys)}')
            for k in ret.missing_keys[:10]: print(f'    {k}')
            if len(ret.missing_keys) > 10: print(f'    ... and {len(ret.missing_keys) - 10} more')
        if ret.unexpected_keys: print(f'  Unexpected keys: {len(ret.unexpected_keys)}')
        return ret


    def forward(self, src_input): # Full forward pass: Recognition -> VLMapper -> Decoder
        recognition_outputs = self.recognition_network(src_input)
        mapped_feature = self.vl_mapper(visual_outputs=recognition_outputs)

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
                + self.translation_weight * translation_outputs['translation_loss'])
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
                + self.translation_weight * translation_outputs['translation_loss'])
            model_outputs['transformer_inputs'] = { # Store for generate_txt
                'input_feature': mapped_feature,
                'input_lengths': recognition_outputs['input_lengths'],
            }
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
            diffusion_steps = generate_cfg.get('diffusion_steps', 128)
            max_length = generate_cfg.get('max_length', 100)
            output = self.bd_decoder.generate(
                input_feature=transformer_inputs['input_feature'],
                input_lengths=transformer_inputs['input_lengths'],
                max_length=max_length, diffusion_steps=diffusion_steps)
            output['decoded_sequences'] = self._decode_sequences(output['sequences'])
            return output


    def _decode_sequences(self, sequences): # Convert token IDs to text strings
        decoded = []
        bos_index = self.bd_decoder.bos_index if self.bd_decoder else -1
        # mask_token_id is the new attribute name; mask_index kept for compat
        mask_id = getattr(self.bd_decoder, 'mask_token_id', getattr(self.bd_decoder, 'mask_index', -1)) if self.bd_decoder else -1

        for seq in sequences:
            tokens = []
            for tok_id in seq:
                tok_id = tok_id.item()
                if tok_id == self.text_tokenizer.eos_index: break
                if tok_id == mask_id: continue
                if tok_id == self.text_tokenizer.pad_index: continue
                if tok_id == bos_index: continue
                tokens.append(tok_id)

            if hasattr(self.text_tokenizer, 'level') and self.text_tokenizer.level == 'word':
                text = ' '.join(self.text_tokenizer.id2token[t] for t in tokens if t < len(self.text_tokenizer.id2token))
            else: # Sentencepiece: need to reverse prune
                if hasattr(self.text_tokenizer, 'pruneids_reverse'):
                    tokens = [self.text_tokenizer.pruneids_reverse.get(t, t) for t in tokens]

                tok_tensor = torch.tensor([tokens], dtype=torch.long)
                text = self.text_tokenizer.tokenizer.batch_decode(tok_tensor, skip_special_tokens=True)[0]

            # Clean up German punctuation (matching MSKA's TextTokenizer.batch_decode)
            if hasattr(self.text_tokenizer, 'tokenizer') and \
               hasattr(self.text_tokenizer.tokenizer, 'tgt_lang') and \
               'de' in self.text_tokenizer.tokenizer.tgt_lang:
                if len(text) > 2 and text[-1] == '.' and text[-2] != ' ': text = text[:-1] + ' .'
            decoded.append(text)
        return decoded