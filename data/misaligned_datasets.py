import os, sys
import numpy as np
import torch

# Add MSKA to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'MSKA'))

from datasets import S2T_Dataset
from data.misalign import apply_misalignment

# 8 misalignment condition types (4 basic + 4 compound)
CONDITION_TYPES = ['HT', 'TT', 'HC', 'TC', 'HT+TT', 'HT+TC', 'HC+TT', 'HC+TC']


class EvalDataset(S2T_Dataset):
    '''S2T_Dataset that applies temporal misalignment in __getitem__.

    Usage:
        dataset = EvalDataset(path, tokenizer, config, args, phase)
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
        
    def _check_sample_ordering(self): # Log whether samples appear to be sequentially ordered
        if len(self.list) < 3: return
        names = self.list[:10]
        print('\nSample ordering check (first 10 names):')
        for i, n in enumerate(names): print(f'  [{i}] {n}')
        print(f'  ... ({len(self.list)} total)')
        print('If names share a video-session prefix and are sequential,')
        print('contamination frames come from the same continuous stream.\n')


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
            head_trunc = int(np.floor(self._delta_s * T)) if self._delta_s > 0 else 0
            tail_trunc = int(np.floor(abs(self._delta_e) * T)) if self._delta_e < 0 else 0
            if T - head_trunc - tail_trunc < self.min_frames: self._skipped_indices.add(i)
            
            
    def _get_sample(self, index):
        key = self.list[index]
        sample = self.raw_data[key]
        name_sample = sample['name']
        gloss = sample['gloss']
        text = sample['text'] if self.config['task'] != 'S2G' else None
        return name_sample, sample, gloss, text
            
            
    def _get_contamination_keypoints(self, index, delta_s=None, delta_e=None):
        '''Get previous and next keypoint samples for potential contamination.

        Args:
            index: position in self.list
            delta_s: start-offset ratio to use for HC check (negative → contamination from prev).
                     If None, falls back to self._delta_s (set by set_condition() for eval).
            delta_e: end-offset ratio to use for TC check (positive → contamination from next).
                     If None, falls back to self._delta_e (set by set_condition() for eval).
        '''
        _ds = delta_s if delta_s is not None else self._delta_s
        _de = delta_e if delta_e is not None else self._delta_e

        prev_keypoints = None
        if _ds < 0 and index > 0:
            prev_key = self.list[index - 1]
            prev_keypoints = self.raw_data[prev_key]['keypoint'].numpy()

        next_keypoints = None
        if _de > 0 and index < len(self.list) - 1:
            next_key = self.list[index + 1]
            next_keypoints = self.raw_data[next_key]['keypoint'].numpy()
        return prev_keypoints, next_keypoints
    

    def __getitem__(self, index):
        name_sample, sample, gloss, text = self._get_sample(index)

        # Clean or skipped -> return original data
        if (self._delta_s == 0.0 and self._delta_e == 0.0) or index in self._skipped_indices:
            keypoints = sample['keypoint'].permute(2, 0, 1).to(torch.float32)
            length = sample['num_frames']
            return name_sample, keypoints, gloss, text, length

        # Get raw keypoints as numpy (T, V, C) and adjacent samples for contamination
        keypoints_np = sample['keypoint'].numpy()
        prev_keypoints, next_keypoints = self._get_contamination_keypoints(index)

        # Apply misalignment — architecture floor (T_enc ≥ 2 requires T ≥ 8)
        misaligned_keypoints, info = apply_misalignment(
            keypoints_np, self._delta_s, self._delta_e,
            prev_keypoints=prev_keypoints, next_keypoints=next_keypoints,
            max_length=self.max_input_length, min_frames=self.min_frames,
        )
        # Convert back: (T, V, C) -> (C, T, V) via permute(2, 0, 1)
        keypoints = torch.from_numpy(misaligned_keypoints).permute(2, 0, 1).to(torch.float32)
        length = misaligned_keypoints.shape[0]
        return name_sample, keypoints, gloss, text, length
    
    
class TrainDataset(EvalDataset):
    '''S2T_Dataset with on-the-fly misalignment augmentation for training.

    For each sample in each epoch:
      - With probability p_aug, apply a random misalignment
      - With probability 1 - p_aug, use clean sample
    '''
    def __init__(
        self, path, tokenizer, config, args, phase, training_refurbish=True, 
        p_aug=0.5, knee_thresholds=None, min_severity=0.05, min_frames=8, max_input_length=400,
    ):
        super().__init__(
            path, tokenizer, config, args, phase, training_refurbish=training_refurbish, 
            min_frames=min_frames, max_input_length=max_input_length
        )
        self.p_aug = p_aug
        self.min_severity = min_severity

        # Per-condition knee thresholds (max severity for augmentation sampling)
        self.knee_thresholds = knee_thresholds or {'HT': 0.25, 'TT': 0.40, 'HC': 0.30, 'TC': 0.30}
        print(f'TrainDataset: p_aug={p_aug}, min_severity={min_severity}, min_frames={min_frames}')
        print(f'  Knee thresholds: {self.knee_thresholds}')


    def _sample_misalignment(self):
        '''Sample a random misalignment condition and severity.

        Returns:
            delta_s_ratio: float, start offset ratio
            delta_e_ratio: float, end offset ratio
        '''
        cond_type = CONDITION_TYPES[np.random.randint(len(CONDITION_TYPES))]
        delta_s, delta_e = 0.0, 0.0

        if 'HT' in cond_type:
            max_sev = self.knee_thresholds['HT']
            delta_s = np.random.uniform(self.min_severity, max_sev)
        elif 'HC' in cond_type:
            max_sev = self.knee_thresholds['HC']
            delta_s = -np.random.uniform(self.min_severity, max_sev)

        if 'TT' in cond_type:
            max_sev = self.knee_thresholds['TT']
            delta_e = -np.random.uniform(self.min_severity, max_sev)
        elif 'TC' in cond_type:
            max_sev = self.knee_thresholds['TC']
            delta_e = np.random.uniform(self.min_severity, max_sev)
        return delta_s, delta_e


    def __getitem__(self, index):
        name_sample, sample, gloss, text = self._get_sample(index)
        if self.p_aug > 0 and np.random.random() < self.p_aug: # Decide whether to apply augmentation
            delta_s, delta_e = self._sample_misalignment()
            
            # Get raw keypoints as numpy (T, V, C) and Get adjacent samples for contamination
            # Pass the locally-sampled delta_s/delta_e so HC (delta_s<0) and TC (delta_e>0)
            keypoints_np = sample['keypoint'].numpy()
            prev_keypoints, next_keypoints = self._get_contamination_keypoints(index, delta_s=delta_s, delta_e=delta_e)

            misaligned_keypoints, info = apply_misalignment(
                keypoints_np, delta_s, delta_e,
                prev_keypoints=prev_keypoints, next_keypoints=next_keypoints,
                max_length=self.max_input_length, min_frames=self.min_frames,
            )
            if info.get('skipped', False): # If misalignment was skipped (too few frames), fall back to clean
                keypoints = sample['keypoint'].permute(2, 0, 1).to(torch.float32)
                length = sample['num_frames']
            else:
                keypoints = torch.from_numpy(misaligned_keypoints).permute(2, 0, 1).to(torch.float32)
                length = misaligned_keypoints.shape[0]
        else: # Clean sample
            keypoints = sample['keypoint'].permute(2, 0, 1).to(torch.float32)
            length = sample['num_frames']
        return name_sample, keypoints, gloss, text, length # Text label is ALWAYS the original translation, unchanged