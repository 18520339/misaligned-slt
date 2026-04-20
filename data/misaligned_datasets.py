'''Misalignment-aware dataset wrappers for SLT.

Works with both pose and RGB modalities via BaseDataset.

EvalDataset: fixed misalignment condition applied to all samples.
TrainDataset: random per-sample misalignment augmentation for training.

For pose: misalignment operates on raw (T, 133, 2|3) keypoints before normalization.
For RGB: misalignment operates on the frame-path list before image loading.
'''
import numpy as np
import torch
from .base_dataset import BaseDataset
from .misalign import apply_misalignment

# 8 misalignment condition types (4 basic + 4 compound)
CONDITION_TYPES = ['HT', 'TT', 'HC', 'TC', 'HT+TT', 'HT+TC', 'HC+TT', 'HC+TC']


class EvalDataset(BaseDataset):
    '''BaseDataset with a fixed misalignment condition.

    Usage:
        dataset = EvalDataset(path, tokenizer, input_modality='pose', ...)
        dataset.set_condition('HT_20', 0.20, 0.0)
        loader = DataLoader(dataset, collate_fn=collate_fn, ...)
    '''
    def __init__(self, *args, min_frames=4, subsample_indices=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_frames = min_frames
        self._delta_s = 0.0
        self._delta_e = 0.0
        self._condition_name = 'clean'
        self._skipped_indices = set()

        if subsample_indices is not None: # Optional subsampling (for train_eval mode)
            self.list = [self.list[i] for i in subsample_indices]

    @property
    def num_skipped(self):
        return len(self._skipped_indices)

    @property
    def sample_keys(self):
        return list(self.list)
    
    def set_condition(self, name: str, delta_s: float, delta_e: float): # Set the current misalignment condition
        self._condition_name = name
        self._delta_s = delta_s
        self._delta_e = delta_e
        self._precompute_skips()

    def _sample_num_frames(self, index):
        key = self.list[index]
        sample = self.raw_data[key]
        if 'num_frames' in sample: return int(sample['num_frames'])
        if self.input_modality == 'pose': return int(sample['keypoint'].shape[0])
        return int(len(sample['imgs_path']))

    def _precompute_skips(self): # Identify samples whose truncated length falls below min_frames
        self._skipped_indices = set()
        if self._delta_s == 0.0 and self._delta_e == 0.0: return
        for i in range(len(self.list)):
            T = self._sample_num_frames(i)
            head_trunc = int(np.floor(self._delta_s * T)) if self._delta_s > 0 else 0
            tail_trunc = int(np.floor(abs(self._delta_e) * T)) if self._delta_e < 0 else 0
            if T - head_trunc - tail_trunc < self.min_frames: self._skipped_indices.add(i)

    def _get_contamination_source(self, index, delta_s=None, delta_e=None):
        '''Return (prev, next) contamination source for the given index.

        For pose: returns np.ndarray keypoints (or None).
        For RGB: returns list of frame paths (or None).
        Args:
            index: position in self.list
            delta_s: start-offset ratio to use for HC check (negative → contamination from prev).
                     If None, falls back to self._delta_s (set by set_condition() for eval).
            delta_e: end-offset ratio to use for TC check (positive → contamination from next).
                     If None, falls back to self._delta_e (set by set_condition() for eval).
        '''
        _ds = delta_s if delta_s is not None else self._delta_s
        _de = delta_e if delta_e is not None else self._delta_e
        prev_src, next_src = None, None
        
        if _ds < 0 and index > 0:
            prev_key = self.list[index - 1]
            prev_sample = self.raw_data[prev_key]
            if self.input_modality == 'pose':
                keypoints = prev_sample['keypoint']
                prev_src = keypoints.numpy() if isinstance(keypoints, torch.Tensor) else keypoints
            else: prev_src = list(prev_sample['imgs_path'])

        if _de > 0 and index < len(self.list) - 1:
            next_key = self.list[index + 1]
            next_sample = self.raw_data[next_key]
            if self.input_modality == 'pose':
                keypoints = next_sample['keypoint']
                next_src = keypoints.numpy() if isinstance(keypoints, torch.Tensor) else keypoints
            else: next_src = list(next_sample['imgs_path'])
        return prev_src, next_src

    # ─── Pose path ──────────────────────────────────────────────────────────
    def _getitem_pose_misaligned(self, index, delta_s, delta_e):
        raw_keypoints, sample = self.get_raw_keypoints(index)
        prev_keypoints, next_keypoints = self._get_contamination_source(index, delta_s=delta_s, delta_e=delta_e)
        misaligned_keypoints, info = apply_misalignment(
            raw_keypoints, delta_s, delta_e,
            prev_keypoints=prev_keypoints, next_keypoints=next_keypoints,
            max_length=self.max_length, min_frames=self.min_frames,
        )
        if info.get('skipped', False): misaligned_keypoints = raw_keypoints
        keypoints = self._preprocess_pose(misaligned_keypoints)
        keypoints = self._maybe_subsample_pose(keypoints)
        frames = torch.from_numpy(keypoints).float()
        return frames, sample

    # ─── RGB path ───────────────────────────────────────────────────────────
    def _splice_frame_paths(self, img_paths, delta_s, delta_e, prev_paths, next_paths):
        # Apply truncation/contamination to a list of frame paths
        T = len(img_paths)
        out = list(img_paths)

        # Head side
        if delta_s > 0:  # head truncation
            head_trunc = int(np.floor(delta_s * T))
            out = out[head_trunc:]
        elif delta_s < 0 and prev_paths is not None:  # head contamination
            n = int(np.floor(abs(delta_s) * T))
            if n > 0: out = list(prev_paths[-n:]) + out

        # Tail side
        if delta_e < 0:  # tail truncation
            tail_trunc = int(np.floor(abs(delta_e) * T))
            if tail_trunc > 0: out = out[:-tail_trunc] if tail_trunc < len(out) else out[:1]
        elif delta_e > 0 and next_paths is not None:  # tail contamination
            n = int(np.floor(delta_e * T))
            if n > 0: out = out + list(next_paths[:n])
        return out

    def _getitem_rgb_misaligned(self, index, delta_s, delta_e):
        img_paths, sample = self.get_raw_frame_paths(index)
        prev_paths, next_paths = self._get_contamination_source(index, delta_s=delta_s, delta_e=delta_e)
        spliced = self._splice_frame_paths(img_paths, delta_s, delta_e, prev_paths, next_paths)
        if len(spliced) < self.min_frames: spliced = img_paths  # fallback
        frames = self._load_rgb_from_paths(spliced)
        return frames, sample

    def __getitem__(self, index):
        if (self._delta_s == 0.0 and self._delta_e == 0.0) or index in self._skipped_indices:
            return super().__getitem__(index) # Clean or skipped -> return original data

        func = self._getitem_pose_misaligned if self.input_modality == 'pose' else self._getitem_rgb_misaligned
        frames, sample = func(index, self._delta_s, self._delta_e)
        name, text = sample['name'], sample['text']
        frame_mask = torch.ones(frames.shape[0], dtype=torch.bool)
        labels = self._tokenize(text, apply_noise=(self.phase == 'train'))
        return name, frames, frame_mask, text, labels


class TrainDataset(EvalDataset):
    '''Random per-sample misalignment augmentation for training.

    For each sample: with probability p_aug, sample a random condition+severity
    and apply it; otherwise return the clean sample.
    '''
    def __init__(
        self, *args,
        p_aug=0.5, knee_thresholds=None, min_severity=0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.p_aug = p_aug
        self.min_severity = min_severity
        self.knee_thresholds = knee_thresholds or {
            'HT': 0.25, 'TT': 0.40, 'HC': 0.30, 'TC': 0.30,
        }
        print(f'TrainDataset: p_aug={p_aug}, min_severity={min_severity}')
        print(f'  Knee thresholds: {self.knee_thresholds}')

    def _sample_misalignment(self):
        cond_type = CONDITION_TYPES[np.random.randint(len(CONDITION_TYPES))]
        delta_s, delta_e = 0.0, 0.0

        if 'HT' in cond_type:
            delta_s = np.random.uniform(self.min_severity, self.knee_thresholds['HT'])
        elif 'HC' in cond_type:
            delta_s = -np.random.uniform(self.min_severity, self.knee_thresholds['HC'])

        if 'TT' in cond_type:
            delta_e = -np.random.uniform(self.min_severity, self.knee_thresholds['TT'])
        elif 'TC' in cond_type:
            delta_e = np.random.uniform(self.min_severity, self.knee_thresholds['TC'])
        return delta_s, delta_e

    def __getitem__(self, index):
        # Decide augmentation this call
        if self.p_aug > 0 and np.random.random() < self.p_aug:
            delta_s, delta_e = self._sample_misalignment()
            # Guard against skip: check length after truncation
            T = self._sample_num_frames(index)
            head_trunc = int(np.floor(delta_s * T)) if delta_s > 0 else 0
            tail_trunc = int(np.floor(abs(delta_e) * T)) if delta_e < 0 else 0
            if T - head_trunc - tail_trunc < self.min_frames:
                return BaseDataset.__getitem__(self, index)

            if self.input_modality == 'pose':
                frames, sample = self._getitem_pose_misaligned(index, delta_s, delta_e)
            else:
                frames, sample = self._getitem_rgb_misaligned(index, delta_s, delta_e)

            name = sample['name']
            text = sample['text']
            T_out = frames.shape[0]
            frame_mask = torch.ones(T_out, dtype=torch.bool)
            labels = self._tokenize(text, apply_noise=(self.phase == 'train'))
            return name, frames, frame_mask, text, labels

        return BaseDataset.__getitem__(self, index)
