'''Generic Sign Language Translation dataset loader.

Supports both pose-keypoint and RGB-frame modalities, with matching preprocessing and augmentation to original GFSLT-VLP. 
Dataset-agnostic: works for Phoenix-2014T, CSL-Daily, etc., as long as the data follows the pickle schema below.

Pickle format (gzip or plain):
    {sample_key: {
        'keypoint': np.ndarray(T, 133, 2|3),   # pose modality
        'imgs_path': List[str],                # rgb modality
        'text': str, 'name': str, 'num_frames': int,
    }}
'''
import os
import gzip
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from poses import normalize_keypoints, threshold_confidence, mska_augment


def noise_inject(text, noise_rate=0.15, noise_type='omit_last', is_train=True):
    '''Word-level noise injection matching original GFSLT-VLP NoiseInjecting.
    
    https://github.com/zhoubenjia/GFSLT-VLP/blob/main/utils.py#L349
    noise_type='omit_last': mask words from the end (up to noise_rate fraction)
    noise_type='omit':      randomly omit noise_rate fraction of words
    '''
    words = text.split()
    if not words or not is_train: return text
    noise_words = words
    
    if noise_type == 'omit_last':
        frac = np.random.uniform(0, noise_rate)
        keep_len = max(1, len(words) - int(np.ceil(len(words) * frac)))
        noise_words = words[:keep_len] + ['<mask>'] * (len(words) - keep_len)
    elif noise_type == 'omit':
        n_keep = int(len(words) * (1.0 - noise_rate))
        keep_indices = set(sorted(random.sample(range(len(words)), min(n_keep, len(words)))))
        noise_words = [w if i in keep_indices else '<mask>' for i, w in enumerate(words)]
    return ' '.join(noise_words)


class BaseDataset(Dataset):
    '''Generic Sign Language Translation dataset supporting pose & RGB modalities.

    Args:
        data_path: path to pickle/gzip pickle.
        tokenizer: HuggingFace mBART tokenizer.
        input_modality: 'pose' or 'rgb'.
        phase: 'train' / 'val' / 'test'.
        max_frames: max temporal frames (original GFSLT-VLP = 300).
        frame_dims: (width, height) tuple for pose normalization.
        frames_dir: root dir for RGB frames (required if modality='rgb').
        crop_size: crop size for RGB (224).
        resize: resize size for RGB (256).
        training_refurbish: enable noise injection for VLP masked LM.
        noise_rate: 0.15 (original GFSLT-VLP).
        noise_type: 'omit_last' (original GFSLT-VLP default).
        pose_augment: enable MSKA-style pose rotation augmentation.
        rgb_augment: enable RGB augmentation (vidaug rotate/resize/translate).
    '''
    def __init__(
        self, data_path, tokenizer, input_modality='rgb', phase='train', max_frames=300, 
        frame_dims=(210, 260), frames_dir=None, crop_size=224, resize=256, training_refurbish=False, 
        noise_rate=0.15, noise_type='omit_last', pose_augment=False, rgb_augment=True,
    ):
        assert input_modality in ('pose', 'rgb'), f"input_modality must be 'pose' or 'rgb'; got {input_modality}"
        if input_modality == 'rgb' and frames_dir is None:
            raise ValueError("frames_dir must be provided when input_modality='rgb'")

        self.tokenizer = tokenizer
        self.input_modality = input_modality
        self.phase = phase
        self.max_frames = max_frames
        self.frame_width, self.frame_height = frame_dims
        self.frames_dir = frames_dir
        self.crop_size = crop_size
        self.resize = resize

        self.training_refurbish = training_refurbish and (phase == 'train')
        self.noise_rate = noise_rate
        self.noise_type = noise_type
        self.pose_augment = pose_augment and (phase == 'train')
        self.rgb_augment = rgb_augment and (phase == 'train')
        
        try: # Load dataset from gzip pickle or plain pickle
            with gzip.open(data_path, 'rb') as f:
                self.raw_data = pickle.load(f)
        except (gzip.BadGzipFile, OSError):
            with open(data_path, 'rb') as f:
                self.raw_data = pickle.load(f)

        self.list = list(self.raw_data.keys())
        print(f'SLTDataset [{input_modality}/{phase}]: {len(self.list)} samples, '
              f'refurbish={self.training_refurbish}, pose_aug={self.pose_augment}, rgb_aug={self.rgb_augment}')

        # RGB augmentation pipeline
        self._rgb_transform = None
        if self.input_modality == 'rgb' and self.rgb_augment:
            try:
                from vidaug import augmentors as va
                sometimes = lambda aug: va.Sometimes(0.5, aug)
                self._rgb_transform = va.Sequential([
                    sometimes(va.RandomRotate(30)),
                    sometimes(va.RandomResize(0.2)),
                    sometimes(va.RandomTranslate(x=10, y=10)),
                ])
            except ImportError:
                self._rgb_transform = None

    def __len__(self):
        return len(self.list)
    
    # ---- Pose modality  ───────────────────────────────────────────────────────
    def get_raw_keypoints(self, index): # Load raw keypoints (T, 133, 2|3) for a sample. Pose modality only
        key = self.list[index]
        sample = self.raw_data[key]
        keypoints = sample['keypoint']
        if isinstance(keypoints, torch.Tensor): keypoints = keypoints.numpy()
        return keypoints, sample

    def _preprocess_pose(self, raw_kp): # Full preprocessing pipeline: (T, 133, 2|3) -> (T, 77, 3)
        T, V, C = raw_kp.shape
        assert V == 133, f'Expected 133 keypoints, got {V}'
        keypoints = np.ones((T, 133, 3), dtype=np.float32)
        keypoints[:, :, :C] = raw_kp
        keypoints = normalize_keypoints(keypoints, self.frame_width, self.frame_height)
        keypoints = threshold_confidence(keypoints)
        return keypoints

    def _maybe_subsample_pose(self, keypoints):
        T = keypoints.shape[0]
        if T <= self.max_frames: return keypoints
        if self.phase == 'train': indices = sorted(random.sample(range(T), self.max_frames))
        else: indices = np.linspace(0, T - 1, self.max_frames, dtype=int)
        return keypoints[indices]

    def _load_pose(self, index):
        raw_kp, sample = self.get_raw_keypoints(index)
        if self.pose_augment: raw_kp = mska_augment(raw_kp, max_angle_deg=15.0, prob=0.5)
        keypoints = self._preprocess_pose(raw_kp)
        keypoints = self._maybe_subsample_pose(keypoints)
        return torch.from_numpy(keypoints).float(), sample
    
    # ---- RGB modality  ───────────────────────────────────────────────────────
    def get_raw_frame_paths(self, index): # Return list of relative frame_paths for a sample. RGB modality only
        key = self.list[index]
        sample = self.raw_data[key]
        return list(sample['imgs_path']), sample
    
    def _maybe_subsample_rgb(self, frame_paths):
        T = len(frame_paths)
        if T <= self.max_frames: return frame_paths
        if self.phase == 'train': indices = sorted(random.sample(range(T), self.max_frames))
        else: indices = np.linspace(0, T - 1, self.max_frames, dtype=int).tolist()
        return [frame_paths[i] for i in indices]

    def _load_rgb(self, index): # Load and preprocess RGB frames matching original GFSLT-VLP
        import cv2
        from PIL import Image
        from torchvision import transforms

        img_paths, sample = self.get_raw_frame_paths(index)
        frame_paths = [os.path.join(self.frames_dir, p) for p in img_paths]
        frame_paths = self._maybe_subsample_rgb(frame_paths)
        batch_images = []
        
        for p in frame_paths:
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            batch_images.append(img)

        if self.phase == 'train' and self._rgb_transform is not None:
            batch_images = self._rgb_transform(batch_images)

        if self.phase == 'train':
            left = np.random.randint(0, max(1, self.resize - self.crop_size))
            top = np.random.randint(0, max(1, self.resize - self.crop_size))
        else:
            left = (self.resize - self.crop_size) // 2
            top = (self.resize - self.crop_size) // 2
        crop_rect = (left, top, left + self.crop_size, top + self.crop_size)
        
        frames = torch.zeros(len(batch_images), 3, self.crop_size, self.crop_size)
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        for i, img in enumerate(batch_images):
            img = img.resize(self.resize)
            img = data_transform(img).unsqueeze(0)
            frames[i] = img[:, :, crop_rect[1]:crop_rect[3], crop_rect[0]:crop_rect[2]]
        return frames, sample

    # ─── Dataset API ────────────────────────────────────────────────────────
    def _tokenize(self, text, apply_noise=False): # Tokenize text, optionally with noise injection for VLP
        labels = {'text': text}
        tgt = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        labels['paragraph_tokens'] = tgt['input_ids'].squeeze(0)
        labels['paragraph_attention_mask'] = tgt['attention_mask'].squeeze(0)

        if apply_noise and self.training_refurbish:
            masked_text = noise_inject(
                text, noise_rate=self.noise_rate,
                noise_type=self.noise_type, is_train=(self.phase == 'train'),
            )
            masked_tgt = self.tokenizer(masked_text, return_tensors='pt', padding=True, truncation=True)
            labels['masked_paragraph_tokens'] = masked_tgt['input_ids'].squeeze(0)
            labels['masked_paragraph_attention_mask'] = masked_tgt['attention_mask'].squeeze(0)
        return labels

    def __getitem__(self, index):
        frames, sample = self._load_pose(index) if self.input_modality == 'pose' else self._load_rgb(index)
        name, text = sample['name'], sample['text']
        frame_mask = torch.ones(frames.shape[0], dtype=torch.bool)
        labels = self._tokenize(text, apply_noise=(self.phase == 'train'))
        return name, frames, frame_mask, text, labels


def collate_fn(batch): # Collate variable-length visual sequences & text with zero-padding.
    # Handles pose (B, T, 77, 3) OR RGB (B, T, 3, H, W). Text tokens pad to the batch max with pad_token_id=1.
    names, seqs, masks, texts, labels_list = zip(*batch)
    max_frames_batch = max(s.size(0) for s in seqs)
    padded_seqs, padded_masks = [], []
    
    for seq, mask in zip(seqs, masks):
        T = seq.size(0)
        if T < max_frames_batch:
            pad_shape = [max_frames_batch - T] + list(seq.shape[1:])
            seq = torch.cat([seq, torch.zeros(pad_shape, dtype=seq.dtype)], dim=0)
            mask = torch.cat([mask, torch.zeros(max_frames_batch - T, dtype=torch.bool)], dim=0)
        padded_seqs.append(seq)
        padded_masks.append(mask)

    pad_token_id = 1 # mBART PAD_IDX
    max_tokens_batch = max(l['paragraph_tokens'].size(0) for l in labels_list)
    padded_labels = []
    
    for l in labels_list:
        padded_l = {'text': l['text']}
        for key in l:
            if key == 'text': continue
            tokens = l[key]
            
            if tokens.dim() == 1 and tokens.size(0) < max_tokens_batch:
                pad_len = max_tokens_batch - tokens.size(0)
                tokens = torch.cat([tokens, tokens.new_full((pad_len,), pad_token_id)])
            padded_l[key] = tokens
        padded_labels.append(padded_l)
    
    return {
        'names': names,
        'pixel_values': torch.stack(padded_seqs),
        'pixel_mask': torch.stack(padded_masks),
        'texts': texts, 'labels': padded_labels,
    }