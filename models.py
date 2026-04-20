''' GFSLT-VLP Models adapted for:
- CoSign pose embeddings backbone (ST-GCN based) instead of ResNet
- Compatible with DVCDataset data loader
- HuggingFace Trainer API compatibility

References:
- GFSLT-VLP: https://github.com/zhoubenjia/GFSLT-VLP
- CoSign: https://openaccess.thecvf.com/content/ICCV2023/papers/Jiao_CoSign_Exploring_Co-occurrence_Signals_in_Skeleton-based_Continuous_Sign_Language_Recognition_ICCV_2023_paper.pdf
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from einops import repeat
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from transformers import MBartConfig, MBartForConditionalGeneration
from transformers.models.mbart.modeling_mbart import shift_tokens_right

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backbones.cosign import CoSign1s


# ======================== Configuration Dataclass ========================
@dataclass
class GFSLTConfig:
    # Model Setup
    embed_dim: int = 1024
    hidden_size: int = 1024
    temporal_kernel: int = 3
    logit_scale_init: float = 0.07
    mbart_name: str = './trimmed_mbart'
    label_smoothing: float = 0.2

    # Modality selection
    input_modality: str = 'rgb'  # 'rgb' or 'pose'
    rgb_backbone: str = 'resnet18'

    # Pose input dimensions (from config.py)
    num_keypoints: int = 77  # Selected keypoints from CoSign
    input_channels: int = 3  # x, y, confidence


# ======================== Pose Visual Backbone (CoSign-based) ========================
class TemporalConv1D(nn.Module): # 1D temporal convolution for downsampling temporal dimension
    def __init__(self, input_size: int, hidden_size: int, conv_type: int = 2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if conv_type == 0: self.kernel_size = ['K3']
        elif conv_type == 1: self.kernel_size = ['K5', 'P2']
        elif conv_type == 2: self.kernel_size = ['K5', 'P2', 'K5', 'P2']
        else: raise ValueError(f'Unsupported conv_type: {conv_type}')

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P': modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0))
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: (B, T, C) - input features
        Returns:
            (B, T', C) - temporally downsampled features
        '''
        x = self.temporal_conv(x.permute(0, 2, 1)) # (B, C, T) -> conv -> (B, C, T')
        return x.permute(0, 2, 1) # (B, T', C)

    
class PoseFeatureExtractor(nn.Module):
    '''
    Combines CoSign backbone with optional temporal convolution for downsampling.
    Replaces the ResNet-based FeatureExtracter from original GFSLT-VLP.
    '''
    def __init__(self, config: GFSLTConfig, level: str = 'spatial', adaptive: bool = True, use_temporal_conv: bool = True):
        super().__init__()
        self.config = config
        self.cosign = CoSign1s( # Use CoSign backbone for pose feature extraction
            temporal_kernel=config.temporal_kernel, hidden_size=config.hidden_size,
            level=level, adaptive=adaptive
        )
        self.use_temporal_conv = use_temporal_conv
        if self.use_temporal_conv:
            self.temporal_conv = TemporalConv1D(input_size=config.hidden_size, hidden_size=config.embed_dim, conv_type=2)
    
    def forward(self, poses: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
            poses: (B, T, K, 3) - pose sequences [batch, time, keypoints, channels (x,y,conf)]
            attention_mask: (B, T) - mask for valid frames (1 = valid, 0 = padding)
            
        Returns:
            features: (B, T', embed_dim) - extracted and downsampled features
            new_attention_mask: (B, T') - downsampled attention mask
        '''
        features = self.cosign(poses)  # CoSign expects (B, T, K, 3) and outputs (B, T, hidden_size)
        if self.use_temporal_conv:
            features = self.temporal_conv(features)  # (B, T', embed_dim)
            
            # Update attention mask after temporal conv (downsampled by factor ~4 with conv_type=2)
            if attention_mask is not None: # Downsample attention mask to match feature length
                new_length = features.size(1) # Calculate the temporal reduction factor
                if new_length < attention_mask.size(1): # Use adaptive pooling to match dimensions
                    # attention_mask = F.interpolate(attention_mask.float().unsqueeze(1), size=new_length, mode='nearest').squeeze(1)
                    attention_mask = F.adaptive_max_pool1d(attention_mask.float().unsqueeze(1), new_length).squeeze(1)
            else: attention_mask = torch.ones(features.size(0), features.size(1), device=features.device)
        return features, attention_mask


# ======================== RGB Visual Backbone (ResNet-based) ========================
class RGBFeatureExtractor(nn.Module):
    '''ResNet-based RGB feature extractor matching original GFSLT-VLP.

    Pipeline: (B, T, 3, H, W) -> ResNet-18 -> (B, T, 512) -> TemporalConv1D -> (B, T', embed_dim).
    '''
    def __init__(self, config: GFSLTConfig):
        super().__init__()
        from torchvision import models
        if config.rgb_backbone == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            feat_dim = 512
        elif config.rgb_backbone == 'resnet34':
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            feat_dim = 512
        elif config.rgb_backbone == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            feat_dim = 2048
        else:
            raise ValueError(f'Unsupported rgb_backbone: {config.rgb_backbone}')

        # Strip final avgpool and fc
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feat_dim = feat_dim
        self.temporal_conv = TemporalConv1D(input_size=feat_dim, hidden_size=config.embed_dim, conv_type=2)

    def forward(self, frames: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
            frames: (B, T, 3, H, W) - RGB frames
            attention_mask: (B, T) - mask for valid frames (1 = valid, 0 = padding)
        Returns:
            features: (B, T', embed_dim)
            new_attention_mask: (B, T')
        '''
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)
        x = self.cnn(x)                # (B*T, feat_dim, h, w)
        x = self.pool(x).flatten(1)    # (B*T, feat_dim)
        x = x.view(B, T, self.feat_dim)
        features = self.temporal_conv(x)  # (B, T', embed_dim)

        new_length = features.size(1)
        if attention_mask is not None:
            if new_length < attention_mask.size(1):
                attention_mask = F.adaptive_max_pool1d(
                    attention_mask.float().unsqueeze(1), new_length
                ).squeeze(1)
        else:
            attention_mask = torch.ones(features.size(0), features.size(1), device=features.device)
        return features, attention_mask


def build_backbone(config: GFSLTConfig) -> nn.Module:
    '''Factory for visual backbone based on input modality.'''
    if config.input_modality == 'pose':
        return PoseFeatureExtractor(config, use_temporal_conv=True)
    elif config.input_modality == 'rgb':
        return RGBFeatureExtractor(config)
    raise ValueError(f"Unknown input_modality: {config.input_modality}")


# ======================== Stage 1: Visual-Language Pre-training (CLIP-style) ========================
class TextCLIP(nn.Module): # Text encoder for CLIP-style contrastive learning using MBart encoder
    def __init__(self, config: GFSLTConfig, head_type: str = 'linear'):
        super().__init__()
        self.config = config
        mbart_config = MBartConfig.from_pretrained(config.mbart_name)
        self.backbone = MBartForConditionalGeneration.from_pretrained(config.mbart_name).get_encoder()
        self.lm_head = nn.Linear(mbart_config.d_model, config.embed_dim, bias=False) if head_type == 'linear' else nn.Identity()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
            input_ids: (B, L) - tokenized text input
            attention_mask: (B, L) - attention mask for text
            
        Returns:
            text_features: (B, embed_dim) - CLS-like text features
            hidden_states: (B, L, hidden_dim) - full encoder hidden states
        '''
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]  # (B, L, hidden_dim)
        
        # Get CLS-like representation (from EOS token position)
        eos_positions = input_ids.argmax(dim=-1)  # Find EOS token position
        text_features = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), eos_positions]
        return self.lm_head(text_features), hidden_states


class ImageCLIP(nn.Module): # Pose encoder for CLIP-style contrastive learning using CoSign + MBart encoder
    def __init__(self, config: GFSLTConfig, head_type: str = 'linear'):
        super().__init__()
        self.config = config
        mbart_config = MBartConfig.from_pretrained(config.mbart_name)
        self.backbone = build_backbone(config)
        self.trans_encoder = MBartForConditionalGeneration.from_pretrained(config.mbart_name).get_encoder()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim)) # CLS token for aggregation
        self.lm_head = nn.Linear(mbart_config.d_model, config.embed_dim, bias=False) if head_type == 'linear' else nn.Identity()
        self.proj = nn.Linear(config.embed_dim, mbart_config.d_model) if config.embed_dim != mbart_config.d_model else nn.Identity()
        
        
    def forward(self, poses: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            poses: (B, T, K, 3) - pose sequences
            attention_mask: (B, T) - attention mask for poses
            
        Returns:
            pose_features: (B, embed_dim) - CLS-like pooled visual features
        '''
        # Extract pose features using CoSign backbone
        x, attention_mask = self.backbone(poses, attention_mask)  # (B, T', embed_dim)
        x = self.proj(x)  # (B, T', mbart_d_model)
        B, N, C = x.shape
        
        # Add CLS token
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=B)
        if cls_token.size(-1) != C: cls_token = self.proj(cls_token)
        x = torch.cat((cls_token, x), dim=1)  # (B, 1+T', C)
        
        # Update attention mask for CLS token
        if attention_mask is not None:
            attention_mask = F.pad(attention_mask.float(), (1, 0), value=1.0)
        else:
            attention_mask = torch.ones(B, N + 1, device=x.device)
        
        # Pass through transformer encoder
        outputs = self.trans_encoder(inputs_embeds=x, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = outputs['last_hidden_state']
        return self.lm_head(last_hidden_state[:, 0, :]) # Return CLS token representation


class SLRCLIP(nn.Module):
    '''
    Sign Language Recognition CLIP model for Stage 1 pre-training.
    Performs contrastive learning between pose sequences and text.
    '''
    def __init__(self, config: GFSLTConfig):
        super().__init__()
        self.config = config
        self.model_txt = TextCLIP(config, head_type='identity')
        self.model_images = ImageCLIP(config, head_type='linear')
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / config.logit_scale_init))


    def forward(
        self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor,
        paragraph_tokens: torch.Tensor, paragraph_attention_mask: torch.Tensor,
        masked_paragraph_tokens: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        ''' Forward pass for contrastive learning
        
        Args:
            pixel_values: (B, T, K, 3) - pose keypoints
            pixel_mask: (B, T) - attention mask for poses
            paragraph_tokens: (B, L) - tokenized text input IDs
            
        Returns:
            dict with logits_per_image, logits_per_text, ground_truth, loss
        '''
        # Get features
        image_features = self.model_images(pixel_values, pixel_mask)
        text_features, encoder_hidden_states = self.model_txt(paragraph_tokens, paragraph_attention_mask)
        self.encoder_hidden_states = encoder_hidden_states # Store for potential use in text decoder
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        
        # Ground truth for contrastive loss
        ground_truth = torch.eye(
            logits_per_image.shape[0], 
            device=logits_per_text.device, 
            dtype=logits_per_image.dtype,
            requires_grad=False
        )
        probs_ground_truth = F.softmax(ground_truth * 10, dim=1) # Sharpened ground truth
        
        # Compute contrastive loss using KL Divergence
        probs_per_image = F.log_softmax(logits_per_image, dim=1)
        probs_per_text = F.log_softmax(logits_per_text, dim=1)
        loss_img = F.kl_div(probs_per_image, probs_ground_truth, reduction='batchmean')
        loss_txt = F.kl_div(probs_per_text, probs_ground_truth, reduction='batchmean')
        return {
            'loss': (loss_img + loss_txt) / 2.0,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'ground_truth': ground_truth,
            'image_features': image_features,
            'text_features': text_features,
        }


class TextDecoder(nn.Module):
    '''
    Text decoder for masked language modeling during Stage 1.
    Uses MBart decoder to reconstruct masked text from encoder hidden states.
    '''
    def __init__(self, config: GFSLTConfig):
        super().__init__()
        self.config = config
        mbart_config = MBartConfig.from_pretrained(config.mbart_name)
        mbart = MBartForConditionalGeneration.from_pretrained(config.mbart_name)
        
        self.text_decoder = mbart.get_decoder()
        self.lm_head = mbart.get_output_embeddings()
        self.register_buffer('final_logits_bias', torch.zeros((1, mbart.model.shared.num_embeddings)))
        
        self.pad_token_id = mbart_config.pad_token_id
        self.mbart_config = mbart_config
        

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor, encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Args:
            input_ids: (B, L) - target paragraph token IDs
            attention_mask: (B, L) - attention mask for target
            encoder_hidden_states: (B, L, D) - encoder hidden states from TextCLIP
            encoder_attention_mask: (B, L) - attention mask for encoder
            
        Returns:
            logits: (B, L, vocab_size) - predicted logits
        '''
        decoder_out = self.text_decoder(
            input_ids=shift_tokens_right(input_ids, self.pad_token_id),
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        return self.lm_head(decoder_out[0]) + self.final_logits_bias


# ======================== Stage 2: Gloss-Free Sign Language Translation ========================
class VisualEncoder(nn.Module): # Visual encoder that projects pose features to mBART embedding space for Stage 2
    def __init__(self, emb_size: int, feature_size: int):
        super().__init__()
        self.src_emb = nn.Linear(feature_size, emb_size)
        self.bn_ac = nn.Sequential(nn.BatchNorm1d(emb_size), nn.ReLU(inplace=True))
        
        for m in self.modules(): # Initialize weights
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            src: (B, T, feature_size) - input features
        Returns:
            (B, T, emb_size) - projected features
        '''
        src = self.src_emb(src)
        return self.bn_ac(src.permute(0, 2, 1)).permute(0, 2, 1)


class GFSLT(nn.Module):
    '''
    Gloss-Free Sign Language Translation model for Stage 2.
    Uses CoSign backbone + MBart for end-to-end translation.
    '''
    def __init__(self, config: GFSLTConfig):
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config) # Modality-dispatched feature extraction
        self.mbart = MBartForConditionalGeneration.from_pretrained(config.mbart_name) # MBart for translation
        self.sign_emb = VisualEncoder(emb_size=self.mbart.config.d_model, feature_size=config.embed_dim) # Visual encoder projection
        self.embed_scale = 1.0


    def _prepare_inputs(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Prepare pose inputs for MBart
        frame_features, attention_mask = self.backbone(pixel_values, pixel_mask) # Extract pose features
        inputs_embeds = self.sign_emb(frame_features)
        inputs_embeds = self.embed_scale * inputs_embeds
        return inputs_embeds, attention_mask
    
    
    def forward(
        self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor,
        paragraph_tokens: torch.Tensor, paragraph_attention_mask: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        ''' Forward pass for training. Returns logits only (loss computed externally).

        Args:
            pixel_values: (B, T, K, 3) pose or (B, T, 3, H, W) RGB
            pixel_mask: (B, T) - attention mask for visual input
            paragraph_tokens: (B, L) - target tokens
            paragraph_attention_mask: (B, L) - attention mask for target

        Returns:
            dict with 'logits': (B, L, vocab_size)
        '''
        inputs_embeds, attention_mask = self._prepare_inputs(pixel_values, pixel_mask)
        decoder_input_ids = shift_tokens_right(
            paragraph_tokens,
            self.mbart.config.pad_token_id,
            self.mbart.config.decoder_start_token_id,
        )
        out = self.mbart(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=paragraph_attention_mask,
            return_dict=True,
        )
        return {'logits': out.logits}


    @torch.no_grad()
    def generate(
        self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor,
        max_new_tokens: int = 150, num_beams: int = 4,
        decoder_start_token_id: Optional[int] = None,
        **generate_kwargs
    ) -> torch.Tensor:
        ''' Generate translations using HuggingFace's generate method.

        Args:
            pixel_values: (B, T, K, 3) pose or (B, T, 3, H, W) RGB
            pixel_mask: (B, T) - attention mask for visual input
            max_new_tokens: maximum number of tokens to generate (default 150)
            num_beams: beam search width (default 4)
            decoder_start_token_id: start token for decoder

        Returns:
            generated token ids
        '''
        inputs_embeds, attention_mask = self._prepare_inputs(pixel_values, pixel_mask)
        return self.mbart.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, num_beams=num_beams,
            decoder_start_token_id=decoder_start_token_id,
            **generate_kwargs
        )


# ======================== Stage 1 -> Stage 2 Weight Loading ========================
def load_stage1_weights(model: 'GFSLT', vlp_ckpt_path: str, mbart_name: str) -> Dict[str, List[str]]:
    '''Map SLRCLIP (Stage 1) weights onto a GFSLT (Stage 2) model.

    Mapping:
        model_images.backbone.*         -> backbone.*
        model_images.trans_encoder.*    -> mbart.model.encoder.*
        model_images.proj.*             -> sign_emb.src_emb.*  (only if shapes match)
        model_images.lm_head.*          -> skipped
    Additionally, decoder embed_tokens/embed_positions are reloaded from the
    pretrained mBART (required: Stage 1 trains only the encoder via CLIP+MLM).

    Args:
        model: GFSLT model (Stage 2, target).
        vlp_ckpt_path: path to Stage 1 checkpoint (torch .pth).
        mbart_name: HF name/path of the trimmed mBART used for Stage 2.

    Returns:
        dict with 'missing' and 'unexpected' key counts and lists.
    '''
    ckpt = torch.load(vlp_ckpt_path, map_location='cpu', weights_only=False)
    raw_state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt

    # Strip any 'base_module.' prefix (older Wrapper4Trainer checkpoints)
    stripped = {}
    for k, v in raw_state.items():
        nk = k[len('base_module.'):] if k.startswith('base_module.') else k
        stripped[nk] = v

    mapped = {}
    current_state = model.state_dict()

    for k, v in stripped.items():
        if k.startswith('model_images.backbone.'):
            new_k = 'backbone.' + k[len('model_images.backbone.'):]
            mapped[new_k] = v
        elif k.startswith('model_images.trans_encoder.'):
            new_k = 'mbart.model.encoder.' + k[len('model_images.trans_encoder.'):]
            mapped[new_k] = v
        elif k.startswith('model_images.proj.'):
            new_k = 'sign_emb.src_emb.' + k[len('model_images.proj.'):]
            if new_k in current_state and current_state[new_k].shape == v.shape:
                mapped[new_k] = v
            # Else: skip (shape mismatch)
        # lm_head and other keys: skipped

    # Reload decoder embed_tokens and embed_positions from pretrained mBART
    pretrained_mbart = MBartForConditionalGeneration.from_pretrained(mbart_name)
    pretrained_state = pretrained_mbart.state_dict()
    for subkey in ('embed_tokens', 'embed_positions'):
        prefix = f'model.decoder.{subkey}.'
        for pk, pv in pretrained_state.items():
            if pk.startswith(prefix):
                mapped['mbart.' + pk] = pv

    result = model.load_state_dict(mapped, strict=False)
    print(f'[load_stage1_weights] mapped {len(mapped)} tensors from {vlp_ckpt_path}')
    print(f'  missing: {len(result.missing_keys)}, unexpected: {len(result.unexpected_keys)}')
    return {'missing': list(result.missing_keys), 'unexpected': list(result.unexpected_keys)}