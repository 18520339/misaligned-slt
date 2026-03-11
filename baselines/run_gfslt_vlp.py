"""
GFSLT-VLP inference wrapper for misalignment benchmark.

This module loads a trained GFSLT-VLP model and runs inference on our
misaligned evaluation data. It handles the full pipeline:
  1. Load the GFSLT-VLP model (backbone + mbart decoder)
  2. Preprocess video frames matching GFSLT-VLP's exact pipeline
  3. Generate translations via beam search
  4. Save predictions to text files

The key insight from reverse-engineering GFSLT-VLP's codebase:
  - Frames are loaded, resized to (resize x resize), center-cropped to (input_size x input_size)
  - Normalized with ImageNet mean/std
  - Padded with first/last frame replication (8 left, right to align with TemporalConv)
  - Flat-concatenated across batch as src_input['input_ids'] with src_length_batch
  - src_input['attention_mask'] is computed after TemporalConv downsampling

Usage:
    python baselines/run_gfslt_vlp.py \
        --gfslt_dir baselines/GFSLT-VLP \
        --checkpoint baselines/GFSLT-VLP/out/Gloss-Free/best_checkpoint.pth \
        --data_root data/phoenix14t \
        --split test \
        --output_dir results/predictions/gfslt_vlp
"""

import argparse
import os
import sys
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# ============================================================================
# Constants from GFSLT-VLP's definition.py
# ============================================================================
PAD_IDX = 1


# ============================================================================
# Image preprocessing matching GFSLT-VLP exactly
# ============================================================================
def load_and_preprocess_frames(
    frame_paths: List[str],
    input_size: int = 224,
    resize: int = 256,
    max_length: int = 300,
) -> torch.Tensor:
    """Load and preprocess frames exactly as GFSLT-VLP does.

    Args:
        frame_paths: List of paths to frame image files.
        input_size: Crop size (224 for GFSLT-VLP).
        resize: Resize before crop (256 for GFSLT-VLP).
        max_length: Maximum number of frames (subsample if exceeded).

    Returns:
        Tensor of shape (T, 3, input_size, input_size).
    """
    # Subsample if too long (matching GFSLT-VLP's strategy)
    if len(frame_paths) > max_length:
        tmp = sorted(
            np.random.choice(len(frame_paths), size=max_length, replace=False).tolist()
        )
        frame_paths = [frame_paths[i] for i in tmp]

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Evaluation crop: center crop
    # data_augmentation with is_train=False gives center crop
    resize_dims = (resize, resize)
    crop_h = (resize - input_size) // 2
    crop_w = (resize - input_size) // 2
    crop_rect = (crop_w, crop_h, crop_w + input_size, crop_h + input_size)

    imgs = torch.zeros(len(frame_paths), 3, input_size, input_size)

    for i, img_path in enumerate(frame_paths):
        img = cv2.imread(img_path)
        if img is None:
            # Black frame fallback
            imgs[i] = torch.zeros(3, input_size, input_size)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize(resize_dims)
        img = data_transform(img).unsqueeze(0)
        imgs[i, :, :, :] = img[:, :, crop_rect[1]:crop_rect[3], crop_rect[0]:crop_rect[2]]

    return imgs


def pad_video_for_temporal_conv(video: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Pad a video tensor with first/last frame replication.

    GFSLT-VLP pads with 8 frames at the start (first frame replicated)
    and enough at the end to make the length divisible by 4 + 8.

    Args:
        video: Tensor of shape (T, C, H, W).

    Returns:
        Tuple of (padded_video, padded_length).
    """
    T = len(video)
    left_pad = 8
    right_pad = int(math.ceil(T / 4.0)) * 4 - T + 8
    total_len = T + left_pad + right_pad

    # Compute the length after padding but before temporal conv truncation
    padded_length = int(math.ceil(T / 4.0)) * 4 + 16

    padded = torch.cat([
        video[0:1].expand(left_pad, -1, -1, -1),
        video,
        video[-1:].expand(total_len - T - left_pad, -1, -1, -1),
    ], dim=0)

    # Truncate to padded_length
    padded = padded[:padded_length]

    return padded, padded_length


def prepare_src_input_single(
    video: torch.Tensor,
) -> Tuple[torch.Tensor, int, int]:
    """Prepare a single video for GFSLT-VLP's model.

    Args:
        video: Tensor of shape (T, C, H, W).

    Returns:
        Tuple of (padded_video, src_length, new_src_length_after_conv).
    """
    padded, src_length = pad_video_for_temporal_conv(video)

    # After TemporalConv with conv_type=2: two K5+P2 stages
    # Length after conv: (((L - 5 + 1) / 2) - 5 + 1) / 2
    new_length = int((((src_length - 5 + 1) / 2) - 5 + 1) / 2)

    return padded, src_length, new_length


def prepare_batch_src_input(
    videos: List[torch.Tensor],
    device: torch.device,
) -> Dict[str, Any]:
    """Prepare a batch of videos into GFSLT-VLP's src_input format.

    GFSLT-VLP concatenates all frames into a single tensor and uses
    src_length_batch to track per-video lengths. The attention mask
    is computed based on the post-TemporalConv lengths.

    Args:
        videos: List of tensors, each (T_i, C, H, W).
        device: Target device.

    Returns:
        src_input dict matching GFSLT-VLP's expected format.
    """
    padded_videos = []
    src_lengths = []
    new_src_lengths = []

    for video in videos:
        padded, src_len, new_len = prepare_src_input_single(video)
        padded_videos.append(padded)
        src_lengths.append(src_len)
        new_src_lengths.append(new_len)

    # Concatenate all frames (GFSLT-VLP uses flat concat, not padding)
    img_batch = torch.cat(padded_videos, dim=0).to(device)

    src_length_batch = torch.tensor(src_lengths)
    new_src_lengths = torch.tensor(new_src_lengths).long()

    # Create attention mask based on post-conv lengths
    mask_gen = []
    for length in new_src_lengths:
        tmp = torch.ones([length]) + 7  # Arbitrary nonzero values
        mask_gen.append(tmp)
    mask_gen = pad_sequence(mask_gen, padding_value=PAD_IDX, batch_first=True)
    img_padding_mask = (mask_gen != PAD_IDX).long()

    src_input = {
        'input_ids': img_batch,
        'attention_mask': img_padding_mask,
        'src_length_batch': src_length_batch,
        'new_src_length_batch': new_src_lengths,
    }

    return src_input


# ============================================================================
# Model loading
# ============================================================================
def load_gfslt_model(
    gfslt_dir: str,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[nn.Module, Any]:
    """Load a trained GFSLT-VLP model.

    This adds the GFSLT-VLP directory to the Python path, imports their
    model builder, creates the model, and loads the checkpoint.

    Args:
        gfslt_dir: Path to the cloned GFSLT-VLP repository.
        checkpoint_path: Path to the trained checkpoint (.pth file).
        device: Target device.

    Returns:
        Tuple of (model, tokenizer).
    """
    gfslt_dir = Path(gfslt_dir).resolve()

    # Add GFSLT-VLP to Python path
    if str(gfslt_dir) not in sys.path:
        sys.path.insert(0, str(gfslt_dir))

    # Load their config
    import yaml
    config_path = gfslt_dir / "configs" / "config_gloss_free.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Fix paths in config to be relative to gfslt_dir
    for key in ['tokenizer', 'transformer', 'visual_encoder']:
        orig_path = config['model'][key]
        # If it's an absolute path that doesn't exist, try relative to gfslt_dir
        if not os.path.exists(orig_path):
            # Try relative path from gfslt_dir
            relative = gfslt_dir / orig_path.lstrip('./')
            if relative.exists():
                config['model'][key] = str(relative)
            else:
                # Try common locations
                for candidate in [
                    gfslt_dir / "pretrain_models" / "MBart_trimmed",
                    gfslt_dir / "pretrain_models" / "mytran",
                ]:
                    if candidate.exists() and key in ['tokenizer', 'transformer']:
                        config['model'][key] = str(gfslt_dir / "pretrain_models" / "MBart_trimmed")
                        break
                    elif candidate.exists() and key == 'visual_encoder':
                        config['model'][key] = str(gfslt_dir / "pretrain_models" / "mytran")
                        break

    print(f"Model config paths:")
    print(f"  tokenizer: {config['model']['tokenizer']}")
    print(f"  transformer: {config['model']['transformer']}")
    print(f"  visual_encoder: {config['model']['visual_encoder']}")

    # Create a minimal args namespace matching what gloss_free_model expects
    class Args:
        input_size = 224
        resize = 256

    args = Args()

    # Import and create model
    from models import gloss_free_model as GFSLTModel

    # Need to set the hpman defaults before creating model
    try:
        from hpman.m import _
        _('decoder_type', 'LD')
        _('freeze_backbone', False)
    except Exception:
        pass

    model = GFSLTModel(config, args)
    model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    # Load state dict
    ret = model.load_state_dict(state_dict, strict=True)
    print(f"Model loaded successfully.")

    model.eval()

    # Load tokenizer
    from transformers import MBartTokenizer
    tokenizer = MBartTokenizer.from_pretrained(
        config['model']['tokenizer'],
        src_lang='de_DE',
        tgt_lang='de_DE'
    )

    return model, tokenizer, config


# ============================================================================
# Inference
# ============================================================================
@torch.no_grad()
def run_inference(
    model: nn.Module,
    tokenizer: Any,
    videos: List[torch.Tensor],
    device: torch.device,
    batch_size: int = 4,
    max_new_tokens: int = 150,
    num_beams: int = 4,
) -> List[str]:
    """Run inference on a list of preprocessed video tensors.

    Args:
        model: Loaded GFSLT-VLP model.
        tokenizer: MBart tokenizer.
        videos: List of tensors, each (T_i, C, H, W).
        device: Target device.
        batch_size: Batch size for inference.
        max_new_tokens: Maximum output tokens.
        num_beams: Beam search width.

    Returns:
        List of predicted translation strings.
    """
    model.eval()
    all_predictions = []

    decoder_start_token_id = tokenizer.lang_code_to_id['de_DE']

    for start_idx in tqdm(range(0, len(videos), batch_size), desc="Inference"):
        batch_videos = videos[start_idx:start_idx + batch_size]

        # Prepare src_input in GFSLT-VLP format
        src_input = prepare_batch_src_input(batch_videos, device)

        # Generate
        output_ids = model.generate(
            src_input,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            decoder_start_token_id=decoder_start_token_id,
        )

        # Decode
        predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        all_predictions.extend(predictions)

    return all_predictions


# ============================================================================
# Data loading (our format → GFSLT-VLP format)
# ============================================================================
def load_misaligned_data(
    data_root: str,
    split: str,
    gfslt_dir: str,
    condition_name: str = "clean",
    severity: float = 0.0,
    input_size: int = 224,
    resize: int = 256,
) -> Tuple[List[torch.Tensor], List[str], List[str]]:
    """Load data with misalignment applied and preprocess for GFSLT-VLP.

    This function loads the PHOENIX14T data using our dataset classes,
    applies the specified misalignment condition, and preprocesses frames
    to match GFSLT-VLP's expected input format.

    For the 'clean' condition, loads directly from GFSLT-VLP's own data format
    if available; otherwise falls back to our dataset loader.

    Args:
        data_root: Path to PHOENIX14T data root.
        split: Dataset split ('dev', 'test').
        gfslt_dir: Path to GFSLT-VLP directory (for loading their label files).
        condition_name: Misalignment condition name.
        severity: Severity level.
        input_size: Crop size.
        resize: Resize before crop.

    Returns:
        Tuple of (videos, references, sample_ids) where:
            videos: List of preprocessed frame tensors (T_i, C, H, W)
            references: List of reference translation strings
            sample_ids: List of sample IDs
    """
    # Add GFSLT-VLP to path for utils
    gfslt_path = Path(gfslt_dir).resolve()
    if str(gfslt_path) not in sys.path:
        sys.path.insert(0, str(gfslt_path))

    import yaml
    config_path = gfslt_path / "configs" / "config_gloss_free.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load GFSLT-VLP's label file
    import utils as gfslt_utils

    label_key = f'{split}_label_path'
    label_path = config['data'].get(label_key, f'./data/Phonexi-2014T/labels.{split}')
    if not os.path.isabs(label_path):
        label_path = str(gfslt_path / label_path.lstrip('./'))

    print(f"Loading labels from: {label_path}")
    raw_data = gfslt_utils.load_dataset_file(label_path)

    img_path_root = config['data']['img_path']
    # Fix path if needed
    if not os.path.exists(img_path_root):
        # Try common alternatives
        candidates = [
            os.path.join(data_root, "features", "fullFrame-210x260px"),
            os.path.join(data_root, "fullFrame-210x260px"),
            data_root,
        ]
        for cand in candidates:
            if os.path.exists(cand):
                img_path_root = cand
                break

    print(f"Image root: {img_path_root}")

    videos = []
    references = []
    sample_ids = []

    # Build ordered list of samples
    sample_keys = list(raw_data.keys())

    if condition_name == "clean" and severity == 0.0:
        # Clean: load frames directly
        for key in tqdm(sample_keys, desc=f"Loading {split} [{condition_name}]"):
            sample = raw_data[key]
            frame_paths = [img_path_root + x if not os.path.isabs(x) else x
                          for x in sample['imgs_path']]

            # Check if files exist; if paths include train/dev/test prefix, adjust
            if not os.path.exists(frame_paths[0]):
                # Try prepending the split
                frame_paths = [os.path.join(img_path_root, x.lstrip('/'))
                              for x in sample['imgs_path']]

            video = load_and_preprocess_frames(
                frame_paths, input_size=input_size, resize=resize
            )
            videos.append(video)
            references.append(sample['text'].lower())
            sample_ids.append(sample.get('name', key))
    else:
        # Misaligned: apply our misalignment simulation
        from data.misalign import (
            MisalignmentConfig, apply_misalignment, compute_max_offset,
            get_condition_name,
        )

        # Parse condition into delta_s_sign and delta_e_sign
        ds_sign, de_sign = _parse_condition_signs(condition_name)

        for idx, key in enumerate(tqdm(sample_keys, desc=f"Loading {split} [{condition_name}@{severity:.0%}]")):
            sample = raw_data[key]
            frame_paths = [img_path_root + x if not os.path.isabs(x) else x
                          for x in sample['imgs_path']]

            if not os.path.exists(frame_paths[0]):
                frame_paths = [os.path.join(img_path_root, x.lstrip('/'))
                              for x in sample['imgs_path']]

            T = len(frame_paths)

            # Compute offset for this sample
            offset = compute_max_offset(T, severity, 0.25, 50)
            delta_s = ds_sign * offset
            delta_e = de_sign * offset

            # Get adjacent sample frame paths for contamination
            prev_paths = None
            next_paths = None

            if idx > 0 and ds_sign < 0:  # Need prev for head contamination
                prev_key = sample_keys[idx - 1]
                prev_sample = raw_data[prev_key]
                prev_paths = [img_path_root + x if not os.path.isabs(x) else x
                             for x in prev_sample['imgs_path']]
                if not os.path.exists(prev_paths[0]):
                    prev_paths = [os.path.join(img_path_root, x.lstrip('/'))
                                 for x in prev_sample['imgs_path']]

            if idx < len(sample_keys) - 1 and de_sign > 0:  # Need next for tail contam
                next_key = sample_keys[idx + 1]
                next_sample = raw_data[next_key]
                next_paths = [img_path_root + x if not os.path.isabs(x) else x
                             for x in next_sample['imgs_path']]
                if not os.path.exists(next_paths[0]):
                    next_paths = [os.path.join(img_path_root, x.lstrip('/'))
                                 for x in next_sample['imgs_path']]

            # Build misaligned frame path list
            misaligned_paths = _build_misaligned_frame_paths(
                frame_paths, delta_s, delta_e, prev_paths, next_paths
            )

            video = load_and_preprocess_frames(
                misaligned_paths, input_size=input_size, resize=resize
            )
            videos.append(video)
            references.append(sample['text'].lower())
            sample_ids.append(sample.get('name', key))

    return videos, references, sample_ids


def _parse_condition_signs(condition_name: str) -> Tuple[int, int]:
    """Parse a condition name into (delta_s_sign, delta_e_sign).

    Args:
        condition_name: e.g., "head_trunc+tail_contam"

    Returns:
        Tuple of (ds_sign, de_sign) where each is -1, 0, or +1.
    """
    ds_sign = 0
    de_sign = 0

    parts = condition_name.split("+")
    for part in parts:
        part = part.strip()
        if part == "head_trunc":
            ds_sign = 1
        elif part == "head_contam":
            ds_sign = -1
        elif part == "tail_trunc":
            de_sign = -1
        elif part == "tail_contam":
            de_sign = 1

    return ds_sign, de_sign


def _build_misaligned_frame_paths(
    current_paths: List[str],
    delta_s: int,
    delta_e: int,
    prev_paths: Optional[List[str]],
    next_paths: Optional[List[str]],
) -> List[str]:
    """Build the misaligned frame path list.

    Args:
        current_paths: Frame paths for the current sample.
        delta_s: Start offset (positive=trunc, negative=contam).
        delta_e: End offset (negative=trunc, positive=contam).
        prev_paths: Frame paths for previous sample (for head contamination).
        next_paths: Frame paths for next sample (for tail contamination).

    Returns:
        List of frame paths representing the misaligned segment.
        For black frames (no adjacent sample), returns None entries
        which load_and_preprocess_frames handles as black.
    """
    result = []
    T = len(current_paths)

    # Head contamination: prepend frames from previous sample
    if delta_s < 0:
        contam_count = abs(delta_s)
        if prev_paths is not None and len(prev_paths) > 0:
            start_in_prev = max(0, len(prev_paths) - contam_count)
            result.extend(prev_paths[start_in_prev:])
        else:
            # Use the first frame of current as a fallback (black would be better
            # but we need valid paths for cv2.imread)
            result.extend([current_paths[0]] * contam_count)
        effective_start = 0
    elif delta_s > 0:
        effective_start = min(delta_s, T)
    else:
        effective_start = 0

    # Current segment frames
    if delta_e < 0:
        effective_end = max(0, T + delta_e)
    else:
        effective_end = T

    result.extend(current_paths[effective_start:effective_end])

    # Tail contamination: append frames from next sample
    if delta_e > 0:
        contam_count = delta_e
        if next_paths is not None and len(next_paths) > 0:
            end_in_next = min(contam_count, len(next_paths))
            result.extend(next_paths[:end_in_next])
        else:
            result.extend([current_paths[-1]] * contam_count)

    return result


# ============================================================================
# Save/load predictions
# ============================================================================
def save_predictions(
    predictions: List[str],
    references: List[str],
    sample_ids: List[str],
    output_dir: str,
    condition_name: str,
    severity: float,
):
    """Save predictions, references, and sample IDs to files.

    Args:
        predictions: Model output translations.
        references: Ground truth translations.
        sample_ids: Sample identifiers.
        output_dir: Output directory.
        condition_name: Misalignment condition name.
        severity: Severity level.
    """
    os.makedirs(output_dir, exist_ok=True)

    sev_str = f"{int(severity * 100):02d}" if severity > 0 else "00"
    prefix = f"{condition_name}_{sev_str}"

    with open(os.path.join(output_dir, f"{prefix}_hyp.txt"), "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred.strip() + "\n")

    with open(os.path.join(output_dir, f"{prefix}_ref.txt"), "w", encoding="utf-8") as f:
        for ref in references:
            f.write(ref.strip() + "\n")

    with open(os.path.join(output_dir, f"{prefix}_ids.txt"), "w", encoding="utf-8") as f:
        for sid in sample_ids:
            f.write(str(sid).strip() + "\n")

    print(f"  Saved {len(predictions)} predictions to {output_dir}/{prefix}_*.txt")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="GFSLT-VLP inference on misaligned PHOENIX14T data"
    )
    parser.add_argument(
        "--gfslt_dir",
        default="baselines/GFSLT-VLP",
        help="Path to cloned GFSLT-VLP repository",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained GFSLT-VLP checkpoint (.pth)",
    )
    parser.add_argument(
        "--data_root",
        default="data/phoenix14t",
        help="PHOENIX14T data root directory",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["dev", "test"],
    )
    parser.add_argument(
        "--output_dir",
        default="results/predictions/gfslt_vlp",
    )
    parser.add_argument(
        "--condition",
        default="clean",
        help="Misalignment condition name (e.g., 'clean', 'head_trunc', "
             "'head_trunc+tail_contam') or 'all' for full benchmark",
    )
    parser.add_argument(
        "--severity",
        type=float,
        default=0.0,
        help="Severity level (e.g., 0.05, 0.10, 0.20)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=150,
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    print("=" * 60)
    print("Loading GFSLT-VLP model...")
    print("=" * 60)
    model, tokenizer, config = load_gfslt_model(
        args.gfslt_dir, args.checkpoint, device
    )

    if args.condition == "all":
        # Run all 46 conditions
        from data.misalign import MisalignmentConfig
        cfg = MisalignmentConfig()

        conditions = [("clean", 0.0)]
        sign_combos = [
            ("head_contam+tail_trunc", -1, -1),
            ("head_contam", -1, 0),
            ("head_contam+tail_contam", -1, 1),
            ("tail_trunc", 0, -1),
            ("tail_contam", 0, 1),
            ("head_trunc+tail_trunc", 1, -1),
            ("head_trunc", 1, 0),
            ("head_trunc+tail_contam", 1, 1),
        ]

        for severity in cfg.severity_levels:
            for cond_name, _, _ in sign_combos:
                conditions.append((cond_name, severity))

        print(f"\nRunning {len(conditions)} conditions...")
        for cond_name, severity in conditions:
            print(f"\n{'='*60}")
            print(f"Condition: {cond_name} | Severity: {severity:.0%}")
            print(f"{'='*60}")

            videos, references, sample_ids = load_misaligned_data(
                args.data_root, args.split, args.gfslt_dir,
                condition_name=cond_name, severity=severity,
            )

            predictions = run_inference(
                model, tokenizer, videos, device,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )

            save_predictions(
                predictions, references, sample_ids,
                args.output_dir, cond_name, severity,
            )
    else:
        # Single condition
        print(f"\nCondition: {args.condition} | Severity: {args.severity:.0%}")

        videos, references, sample_ids = load_misaligned_data(
            args.data_root, args.split, args.gfslt_dir,
            condition_name=args.condition, severity=args.severity,
        )

        predictions = run_inference(
            model, tokenizer, videos, device,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )

        save_predictions(
            predictions, references, sample_ids,
            args.output_dir, args.condition, args.severity,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
