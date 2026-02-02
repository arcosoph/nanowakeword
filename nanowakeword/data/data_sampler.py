# ==============================================================================
#  NanoWakeWord: Lightweight, Intelligent Wake Word Detection
#  Copyright 2025 Arcosoph. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Project: https://github.com/arcosoph/nanowakeword
# ==============================================================================

import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
from nanowakeword.utils.logger import print_info

def stitch_batch_generator(source_registry, blueprints, batch_size, input_shape):
    """
    Standard Classification Batch Generator.
    - Creates balanced batches of (features, labels).
    - Uses blueprints to create complex acoustic scenes for both positive and negative samples.
    """
    import numpy as np
    import torch

    memmaps = {}
    indices = {}
    target_keys, negative_keys, background_keys = [], [], []

    for alias, meta in source_registry.items():
        try:
            path = meta['path']
            memmaps[alias] = np.load(path, mmap_mode='r')
            indices[alias] = len(memmaps[alias])
            t = meta['type']
            if t == 'target': target_keys.append(alias)
            elif t == 'negative': negative_keys.append(alias)
            elif t == 'background': background_keys.append(alias)
        except Exception as e:
            import logging
            logging.warning(f"[Data Warning] Could not load source '{alias}': {e}")

    if not target_keys:
        raise ValueError("[CRITICAL] No Target sources found! Cannot train.")
    if not (negative_keys or background_keys):
        raise ValueError("[CRITICAL] No Negative or Background sources found! Cannot train.")

    bp_list = [b['composition'] for b in blueprints]
    bp_weights = np.array([b.get('weight', 1.0) for b in blueprints], dtype=np.float32)
    bp_probs = bp_weights / np.sum(bp_weights)
    
    required_samples, feature_dim = input_shape

    while True:
        batch_x = np.zeros((batch_size, required_samples, feature_dim), dtype=np.float32)
        batch_y = np.zeros(batch_size, dtype=np.float32)

        for i in range(batch_size):
            template_idx = np.random.choice(len(bp_list), p=bp_probs)
            template = bp_list[template_idx]
            
            stitched_clips = []
            is_target_present = False
            target_clip_len = 0 
            
            for item in template:
                key = None
                source_pool = []
                
                if item == 'targets': source_pool = target_keys
                elif item == 'negatives': source_pool = negative_keys
                elif item == 'backgrounds': source_pool = background_keys
                elif item in memmaps: key = item

                if source_pool:
                    key = np.random.choice(source_pool)

                if key:
                    idx = np.random.randint(0, indices[key])
                    clip = memmaps[key][idx]
                    stitched_clips.append(clip)
                    if key in target_keys:
                        is_target_present = True
                        target_clip_len = clip.shape[0] 
            
            if not stitched_clips:
                continue

            full_audio = np.vstack(stitched_clips)
            curr_len = full_audio.shape[0]

            final_clip = np.zeros((required_samples, feature_dim), dtype=np.float32)

            if is_target_present:
                target_start_in_full = curr_len - target_clip_len
                
                if required_samples >= target_clip_len:
                    max_start_pos_in_final = required_samples - target_clip_len
                    start_pos_in_final = np.random.randint(0, max_start_pos_in_final + 1)                    
                    start_copy_from_full = max(0, target_start_in_full - start_pos_in_final)                    
                    start_paste_in_final = max(0, start_pos_in_final - target_start_in_full)
                    len_to_copy = min(required_samples - start_paste_in_final, curr_len - start_copy_from_full)
                    final_clip[start_paste_in_final : start_paste_in_final + len_to_copy] = \
                        full_audio[start_copy_from_full : start_copy_from_full + len_to_copy]

                else: 
                    start = np.random.randint(0, target_clip_len - required_samples + 1)
                    final_clip = full_audio[target_start_in_full + start : target_start_in_full + start + required_samples]
            else: 
                if curr_len > required_samples:
                    start = np.random.randint(0, curr_len - required_samples + 1)
                    final_clip = full_audio[start : start + required_samples]
                else: 
                    start = np.random.randint(0, required_samples - curr_len + 1)
                    final_clip[start : start + curr_len, :] = full_audio

            batch_x[i] = final_clip
            if is_target_present:
                batch_y[i] = 1.0
        
        yield (
            torch.from_numpy(batch_x),
            torch.from_numpy(batch_y)
        )


class HardnessCurriculumDataset(Dataset):

    def __init__(self, feature_manifests: dict):
        """
        Initializes the dataset by loading memory-mapped files from a structured manifest.
        It creates separate index pools for each unique key in the manifest and
        initializes a hardness score for each sample.
        """
        super().__init__()

        # Initialize all instance attributes inside the constructor
        self.memmaps = []
        self.source_info = []

        # This dictionary will store the global indices for each unique source key.
        # e.g., {'t': tensor([0, 1, ...]), 'n': tensor([1000, 1001, ...])}
        self.index_pools = {}

        cumulative_len = 0

        # Process each category (e.g., 'targets', 'negatives', 'backgrounds')
        for category, manifest in feature_manifests.items():
            if not manifest: continue
            
            # Process each source file (key-path pair) within the category
            for key, path in manifest.items():
                if not path: continue
                
                try:
                    memmap = np.load(path, mmap_mode='r')
                    length = len(memmap)
                    
                    self.memmaps.append(memmap)
                    
                    # Determine the numeric label based on the category
                    label = 1.0 if category == 'targets' else 0.0
                    
                    # Store information about this data source for __getitem__
                    self.source_info.append({
                        'label': label,
                        'length': length,
                        'start_index': cumulative_len,
                    })

                    # Create and populate the index pool for this specific key
                    indices_for_this_key = list(range(cumulative_len, cumulative_len + length))
                    self.index_pools[key] = torch.tensor(indices_for_this_key, dtype=torch.long)
                    
                    cumulative_len += length

                except FileNotFoundError:
                    print(f"WARNING: File not found for key '{key}', skipping: {path}")
                except Exception as e:
                    print(f"WARNING: Could not load file for key '{key}'. Error: {e}")

        self.total_samples = cumulative_len

        # This tensor tracks the "hardness" of each individual sample across the entire dataset.
        # It is initialized to 1.0 for all samples.
        self.sample_hardness = torch.ones(self.total_samples, dtype=torch.float32)

        print_info(f"Dataset initialized with {len(self.index_pools)} sources | Total samples: {self.total_samples}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        """
        Fetches a single data sample and its label using a global index.
        """
        if index < 0 or index >= self.total_samples:
            raise IndexError(f"Index {index} out of bounds for dataset with size {self.total_samples}")

        # Find the correct file and local index for the given global index.
        file_idx = -1
        for i, info in enumerate(self.source_info):
            # The file's range is [start_index, start_index + length - 1]
            if info['start_index'] <= index < (info['start_index'] + info['length']):
                file_idx = i
                break
        
        if file_idx == -1:
            # This should theoretically never happen if the initial index check passes
            raise RuntimeError(f"Could not find a data source for index {index}")

        # Calculate the local index within the found file
        local_index = index - self.source_info[file_idx]['start_index']
        
        feature = self.memmaps[file_idx][local_index]
        label = torch.tensor(self.source_info[file_idx]['label'], dtype=torch.float32)

        # Return the feature, its label, and its global index
        return torch.from_numpy(feature.astype(np.float32)), label, index


class DynamicClassAwareSampler(Sampler):
    """
    A sampler that builds each batch with a fixed number of samples from different classes
    (positive, speech_negative, noise_negative). The selection within each class is
    weighted by the sample's "hardness" score, which is updated dynamically during training.
    """
    def __init__(self, dataset: HardnessCurriculumDataset, batch_composition: dict, feature_manifests: dict):
        self.dataset = dataset
        self.batch_composition = batch_composition
        self.feature_manifests = feature_manifests
        
        self.num_samples_per_batch = sum(self.batch_composition.values())
        self.num_batches = self._calculate_num_batches()


        # self.hardness_smoothing_factor = self.config.get("hardness_smoothing_factor", 0.75)
        self.hardness_smoothing_factor = 0.75

    def _calculate_num_batches(self):
        """
        Calculates the maximum number of batches that can be created.
        This is limited by the smallest data pool relative to its batch quota.
        """
        min_possible_batches = float('inf')

        # Iterate through each rule (e.g., 'targets': 32) in the composition
        for key_or_category, quota in self.batch_composition.items():
            if quota == 0:
                continue  # Skip rules that don't request any samples

            # Determine the total number of available samples for this rule
            total_available_samples = 0
            if key_or_category in self.dataset.index_pools:
                # Rule is a specific key (e.g., 'n')
                total_available_samples = len(self.dataset.index_pools[key_or_category])
            else:
                # Rule is a category (e.g., 'targets'), so sum up all keys under it
                keys_in_category = self._get_keys_for_category(key_or_category)
                for k in keys_in_category:
                    total_available_samples += len(self.dataset.index_pools.get(k, []))

            if total_available_samples == 0:
                # If any required pool is empty, we can't form any batches
                return 0

            # Calculate how many batches this specific pool can support
            possible_batches_for_this_pool = total_available_samples // quota

            # The true number of batches is limited by the smallest pool
            if possible_batches_for_this_pool < min_possible_batches:
                min_possible_batches = possible_batches_for_this_pool
        
        # If the composition was empty or no limiting factor was found, return 0
        if min_possible_batches == float('inf'):
            return 0

        return min_possible_batches


    # HELPER method inside the class to get all keys for a category 
    def _get_keys_for_category(self, category_name: str) -> list[str]:
        return list(self.feature_manifests.get(category_name, {}).keys())

    def __iter__(self):
        for _ in range(self.num_batches):
            final_batch_indices = []
            hardness = self.dataset.sample_hardness

            # Iterate through the user-defined batch composition
            for key_or_category, num_samples in self.batch_composition.items():
                if num_samples == 0: continue

                # Check if it's a specific key (e.g., 'n') or a category (e.g., 'targets')
                if key_or_category in self.dataset.index_pools:
                    # It's a specific key
                    keys_to_sample_from = [key_or_category]
                else:
                    # It's a category, get all keys under it
                    keys_to_sample_from = self._get_keys_for_category(key_or_category)
                
                if not keys_to_sample_from: continue

                # Combine all indices from the relevant pools
                combined_indices = torch.cat([self.dataset.index_pools[k] for k in keys_to_sample_from])
                
                # Get the hardness scores for these combined indices
                # weights = hardness[combined_indices]

                raw_weights = hardness[combined_indices]
                
                smoothed_weights = raw_weights ** self.hardness_smoothing_factor
                
                weights = smoothed_weights + 1e-6 
                
                
                # Perform weighted sampling
                selected_local_indices = torch.multinomial(weights, num_samples, replacement=True)
                selected_global_indices = combined_indices[selected_local_indices]
                
                final_batch_indices.append(selected_global_indices)
            
            # Combine indices from all composition rules and shuffle
            if not final_batch_indices:
                continue # Skip if batch is empty

            batch = torch.cat(final_batch_indices)
            batch = batch[torch.randperm(len(batch))]
            
            yield batch.tolist()

    def __len__(self):
        return self.num_batches


class ValidationDataset(Dataset):
    """
    A safe, standalone Dataset for validation. It loads data on-the-fly from file paths.
    """
    def __init__(self, feature_manifest: dict):
        super().__init__()
        self.file_paths = []
        self.local_indices = []
        self.labels = []
        
        for category, manifest_paths in feature_manifest.items():
            label = 1.0 if category == 'targets' else 0.0
            
            for key, path in manifest_paths.items():
                try:
                    data = np.load(path, mmap_mode='r')
                    length = len(data)
                    
                    for i in range(length):
                        self.file_paths.append(path)
                        self.local_indices.append(i)
                        self.labels.append(label)
                        
                except FileNotFoundError:
                    print(f"[WARNING] Validation file not found, skipping: {path}")
                except Exception as e:
                    print(f"[WARNING] Could not probe validation file '{path}'. Error: {e}")
    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.file_paths[index]
        local_index = self.local_indices[index]
        label = self.labels[index]
        
        data = np.load(path, mmap_mode='r')
        feature = data[local_index]

        feature_tensor = torch.from_numpy(feature.astype(np.float32))
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return feature_tensor, label_tensor, index
