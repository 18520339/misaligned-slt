'''Misalignment-aware dataset wrapper for MSKA's S2T_Dataset.

Subclasses MSKA's dataset to apply temporal misalignment on-the-fly during
inference, while reusing all of MSKA's preprocessing.
'''
import os, sys
import numpy as np
import torch

# Add MSKA to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'MSKA'))
from datasets import S2T_Dataset
from data.misalign import apply_misalignment


class MisalignedDataset(S2T_Dataset):
    '''S2T_Dataset that applies temporal misalignment in __getitem__.

    Usage:
        dataset = MisalignedDataset(path, tokenizer, config, args, phase)
        dataset.set_condition('HT_20', 0.20, 0.0)
        loader = DataLoader(dataset, collate_fn=dataset.collate_fn, ...)
    '''
    def __init__(
        self, path, tokenizer, config, args, phase, training_refurbish=False, 
        min_frames=8, max_input_length=400, subsample_indices=None
    ):
        super().__init__(path, tokenizer, config, args, phase, training_refurbish=training_refurbish)
        self.min_frames = min_frames
        self.max_input_length = max_input_length
        self._delta_s = 0.0
        self._delta_e = 0.0
        self._condition_name = 'clean'
        self._skipped_indices = set()

        if subsample_indices is not None: # Optional subsampling (for train_eval mode)
            self.list = [self.list[i] for i in subsample_indices]
        self._check_sample_ordering() # Check sample ordering for adjacency

    def __len__(self):
        # Override parent: after optional subsampling, self.list may be shorter
        # than self.raw_data. DataLoader uses __len__ to bound iteration indices,
        # so we must return len(self.list) to prevent out-of-range __getitem__ calls.
        return len(self.list)

    def _check_sample_ordering(self): # Log whether samples appear to be sequentially ordered
        if len(self.list) < 3: return
        names = self.list[:10]
        print('\nSample ordering check (first 10 names):')
        for i, n in enumerate(names): print(f'  [{i}] {n}')
        print(f'  ... ({len(self.list)} total)')
        print('If names share a video-session prefix and are sequential,')
        print('contamination frames come from the same continuous stream.\n')

    def set_condition(self, name: str, delta_s: float, delta_e: float): # Set the current misalignment condition
        self._condition_name = name
        self._delta_s = delta_s
        self._delta_e = delta_e
        self._precompute_skips()

    def _precompute_skips(self):
        '''Identify samples whose truncated length falls below the architecture floor.

        MSKA's get_selected_index rounds T down to the nearest multiple of 4,
        then two stride-2 stages give T_enc = ⌊(⌊(T−1)/2⌋+1)/2⌋.

        For T ∈ [1, 7]: valid_len ≤ 4 → T_enc = 1 (entire sentence collapses
        to one feature vector; temporal attention and positional encodings carry
        no information — degenerate operating point).

        For T = 8: valid_len = 8 → T_enc = 2, the minimum state where the
        temporal encoder captures at least one relative position pair.

        Therefore min_frames = 8 is the exact architectural boundary, not a
        tuned hyperparameter. Samples where truncation leaves T < 8 frames are
        excluded from that condition's evaluation.
        '''
        self._skipped_indices = set()
        if self._delta_s == 0.0 and self._delta_e == 0.0: return
        for i, key in enumerate(self.list):
            T = self.raw_data[key]['num_frames']
            ht = int(np.floor(self._delta_s * T)) if self._delta_s > 0 else 0
            tt = int(np.floor(abs(self._delta_e) * T)) if self._delta_e < 0 else 0
            if T - ht - tt < self.min_frames: self._skipped_indices.add(i)

    @property
    def num_skipped(self):
        return len(self._skipped_indices)

    @property
    def sample_keys(self):
        return list(self.list)

    def __getitem__(self, index):
        key = self.list[index]
        sample = self.raw_data[key]
        name_sample = sample['name']
        gloss = sample['gloss']
        text = sample['text'] if self.config['task'] != 'S2G' else None

        # Clean or skipped -> return original data
        if (self._delta_s == 0.0 and self._delta_e == 0.0) or index in self._skipped_indices:
            keypoint = sample['keypoint'].permute(2, 0, 1).to(torch.float32)
            length = sample['num_frames']
            return name_sample, keypoint, gloss, text, length

        # Get raw keypoints as numpy (T, V, C)
        kp_np = sample['keypoint'].numpy()

        # Get adjacent samples for contamination
        prev_kp = None
        if self._delta_s < 0 and index > 0:
            prev_key = self.list[index - 1]
            prev_kp = self.raw_data[prev_key]['keypoint'].numpy()

        next_kp = None
        if self._delta_e > 0 and index < len(self.list) - 1:
            next_key = self.list[index + 1]
            next_kp = self.raw_data[next_key]['keypoint'].numpy()

        # Apply misalignment — architecture floor (T_enc ≥ 2 requires T ≥ 8)
        misaligned_kp, _ = apply_misalignment(
            kp_np, self._delta_s, self._delta_e,
            prev_keypoints=prev_kp, next_keypoints=next_kp,
            max_length=self.max_input_length,
            min_frames=self.min_frames,
        )
        # Convert back: (T, V, C) -> (C, T, V) via permute(2, 0, 1)
        keypoint = torch.from_numpy(misaligned_kp).permute(2, 0, 1).to(torch.float32)
        length = misaligned_kp.shape[0]
        return name_sample, keypoint, gloss, text, length