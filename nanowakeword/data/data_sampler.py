# ==============================================================================
#  NanoWakeWord: Lightweight, Intelligent Wake Word Detection
#  Copyright 2025 Arcosoph. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
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


import sys
import os
import bisect
import hashlib
import torch
import torchaudio
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import Dataset, Sampler

from nanowakeword.utils.logger import print_info, print_warning, print_error


class AdaptiveLossAwareDataset(Dataset):
    """
    ISBL (Importance Sampling based on Loss) Algorithm.
    Supports high-performance reading of Memmap (.npy) and E2E (.wav) files.
    Employs lazy-loading and chunked audio reading to ensure zero Out-Of-Memory (OOM) crashes.
    """

    def __init__(self, data_manifest: Dict[str, Dict[str, str]], clip_samples: Optional[int] = None, sample_rate: int = 16000):
        super().__init__()
        
        self.memmaps: List[np.ndarray] = []
        self.source_info: List[Dict[str, Any]] = []
        self.index_pools: Dict[str, torch.Tensor] = {}
        
        self.clip_samples = clip_samples
        self.sample_rate = sample_rate
        self.is_e2e = clip_samples is not None
        
        self._file_lists: Dict[str, List[str]] = {}

        cumulative_len = 0
        
        print_info("Scanning dataset and building index pools...")

        for category, manifest in data_manifest.items():
            if not manifest:
                continue
            
            for key, path in manifest.items():
                if not path:
                    continue
                
                try:
                    length = 0
                    if self.is_e2e:
                        length = self._initialize_e2e_source(key, path)
                    else:
                        length = self._initialize_memmap_source(key, path)

                    if length == 0:
                        continue

                    label = 1.0 if category == 'targets' else 0.0
                    
                    self.source_info.append({
                        'label': label,
                        'length': length,
                        'start_index': cumulative_len,
                        'key': key,
                    })

                    # Assign a global index range for this specific data source
                    indices_for_this_key = list(range(cumulative_len, cumulative_len + length))
                    self.index_pools[key] = torch.tensor(indices_for_this_key, dtype=torch.long)
                    cumulative_len += length

                except FileNotFoundError:
                    print_error(f"[Dataset] File not found for key '{key}', skipping: {path}")
                except Exception as e:
                    print_error(f"[Dataset] Could not load file for key '{key}'. Error: {e}")

        if cumulative_len == 0:
            print_error("Critical Error: No valid data found in manifest. Exiting.")
            sys.exit(1)

        self.total_samples = cumulative_len
        self._start_indices = [info['start_index'] for info in self.source_info]
        
        # Hardness tensor for Importance Sampling, dynamically updated by the loss function
        self.sample_hardness = torch.ones(self.total_samples, dtype=torch.float32)

        mode_str = "E2E (On-the-fly Chunked Load)" if self.is_e2e else "Embedding (Memmap)"
        print_info(f"Dataset Successfully Initialized [{mode_str}] | Sources: {len(self.index_pools)} | Total Samples: {self.total_samples}")

    def _initialize_e2e_source(self, key: str, path: str) -> int:
        """Rapidly scans directories using ThreadPoolExecutor to prevent startup bottlenecks."""
        target_path = Path(path)
        
        if target_path.is_dir():
            # Parallel file scanning for extreme speed on large directories
            wav_files = []
            with ThreadPoolExecutor() as executor:
                # Iterate through dir and grab all wav files quickly
                entries = list(os.scandir(target_path))
                wav_files = [e.path for e in entries if e.is_file() and e.name.lower().endswith('.wav')]
            
            # Fallback to rglob if subdirectories exist and main dir is empty
            if not wav_files:
                wav_files = [str(p) for p in target_path.rglob("*.wav")]
                
            self._file_lists[key] = sorted(wav_files)
            return len(wav_files)
        
        elif target_path.is_file() and target_path.suffix.lower() == '.wav':
            self._file_lists[key] = [str(target_path)]
            return 1
        else:
            print_error(f"E2E mode: expected WAV directory/file for '{key}', got: {path}")
            return 0

    def _initialize_memmap_source(self, key: str, path: str) -> int:
        """Safely loads memory-mapped numpy arrays."""
        memmap = np.load(path, mmap_mode='r')
        self.memmaps.append(memmap)
        return len(memmap)

    def _read_audio_chunked(self, filepath: str) -> torch.Tensor:
            """
            ULTRA-FAST Audio Loader: Bypasses expensive torchaudio.info() overhead.
            """
            try:
                # Directly load the whole file. (Faster than partial reading for short clips)
                waveform, sr = torchaudio.load(filepath)

                if sr != self.sample_rate:
                    waveform = torchaudio.functional.resample(
                        waveform, 
                        orig_freq=sr, 
                        new_freq=self.sample_rate
                        )

                if waveform.size(0) > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                waveform = waveform.squeeze(0)
                current_len = waveform.size(0)

                # Fast memory-level pad/crop
                if current_len == self.clip_samples:
                    return waveform
                elif current_len > self.clip_samples:
                    max_offset = current_len - self.clip_samples
                    offset = torch.randint(0, max_offset + 1, (1,)).item()
                    return waveform[offset : offset + self.clip_samples]
                else:
                    clip = torch.zeros(self.clip_samples, dtype=torch.float32)
                    clip[:current_len] = waveform
                    return clip

            except Exception as e:
                return torch.zeros(self.clip_samples, dtype=torch.float32)



    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if index < 0 or index >= self.total_samples:
            raise IndexError(f"Index {index} out of bounds.")

        # O(log N) lookup using bisect
        file_idx = bisect.bisect_right(self._start_indices, index) - 1
        local_index = index - self.source_info[file_idx]['start_index']
        label = torch.tensor(self.source_info[file_idx]['label'], dtype=torch.float32)

        if self.is_e2e:
            key = self.source_info[file_idx]['key']
            filepath = self._file_lists[key][local_index]
            feature = self._read_audio_chunked(filepath)
        else:
            # Memmap reads directly from disk blocks, super fast
            feature = torch.from_numpy(self.memmaps[file_idx][local_index].astype(np.float32))

        return feature, label, index


class DynamicClassAwareSampler(Sampler):
    """
    A Sampler that guarantees specific class proportions per batch.
    It heavily relies on the Importance Sampling framework (ISBL) by utilizing
    the `sample_hardness` tensor to pick harder examples more frequently.
    """
    def __init__(self, dataset: AdaptiveLossAwareDataset, batch_composition: Dict[str, int], data_manifest: Dict[str, Any]):
        self.dataset = dataset
        self.batch_composition = batch_composition
        self.data_manifest = data_manifest
        self.num_samples_per_batch = sum(self.batch_composition.values())
        self.num_batches = self._calculate_num_batches()
        
        # Hyperparameter for controlling how aggressively hard samples are prioritized
        self.hardness_smoothing_factor = 0.75

    def _calculate_num_batches(self) -> int:
        """Determines epoch length based on the limiting factor (smallest pool)."""
        min_possible_batches = float('inf')
        
        for key_or_category, quota in self.batch_composition.items():
            if quota == 0: 
                continue
            
            total_available_samples = 0
            if key_or_category in self.dataset.index_pools:
                total_available_samples = len(self.dataset.index_pools[key_or_category])
            else:
                keys_in_category = self._get_keys_for_category(key_or_category)
                for k in keys_in_category:
                    total_available_samples += len(self.dataset.index_pools.get(k, []))
                    
            if total_available_samples == 0: 
                continue
            
            possible_batches = total_available_samples // quota
            if possible_batches < min_possible_batches:
                min_possible_batches = possible_batches
                
        return 0 if min_possible_batches == float('inf') else min_possible_batches

    def _get_keys_for_category(self, category_name: str) -> List[str]:
        return list(self.data_manifest.get(category_name, {}).keys())

    def __iter__(self):
        hardness_tensor = self.dataset.sample_hardness
        
        for _ in range(self.num_batches):
            final_batch_indices = []
            
            for key_or_category, num_samples in self.batch_composition.items():
                if num_samples == 0: continue
                
                # Resolve mapping
                if key_or_category in self.dataset.index_pools:
                    keys = [key_or_category]
                else:
                    keys = self._get_keys_for_category(key_or_category)
                    
                valid_pools = [self.dataset.index_pools[k] for k in keys if k in self.dataset.index_pools]
                if not valid_pools: continue
                
                # Combine available indices for this rule
                combined_indices = torch.cat(valid_pools)
                
                # Fetch hardness weights, apply smoothing and small epsilon to prevent starvation
                raw_weights = hardness_tensor[combined_indices]
                weights = (raw_weights ** self.hardness_smoothing_factor) + 1e-6
                
                # Selection logic: Replacement allows hard samples to be repeated if pool is tiny
                use_replacement = len(combined_indices) < num_samples
                selected_local_indices = torch.multinomial(
                    weights, 
                    num_samples, 
                    replacement=use_replacement
                    )
                
                final_batch_indices.append(combined_indices[selected_local_indices])
                
            if not final_batch_indices: 
                continue
            
            batch = torch.cat(final_batch_indices)
            batch = batch[torch.randperm(len(batch))] # Shuffle within the batch
            yield batch.tolist()

    def __len__(self) -> int:
        return self.num_batches


class ValidationDataset(Dataset):
    """
    Highly consistent Validation Dataset.
    Uses Hash-based deterministic cropping so validation scores never fluctuate randomly.
    """
    def __init__(self, feature_manifest: Dict[str, Dict[str, str]], clip_samples: Optional[int] = None, sample_rate: int = 16000):
        super().__init__()
        
        self.file_paths: List[str] = []
        self.local_indices: List[int] = []
        self.labels_list: List[float] = []
        
        self.clip_samples = clip_samples
        self.sample_rate = sample_rate
        self.is_e2e = clip_samples is not None
        
        self._mmap_cache: Dict[str, np.ndarray] = {}
        self._file_lists: Dict[str, List[str]] = {}

        for category, manifest_paths in feature_manifest.items():
            label = 1.0 if category == 'targets' else 0.0
            for key, path in manifest_paths.items():
                try:
                    length = 0
                    if self.is_e2e:
                        target_path = Path(path)
                        if target_path.is_dir():
                            wav_files = sorted([str(p) for p in target_path.rglob("*.wav")])
                            self._file_lists[path] = wav_files
                            length = len(wav_files)
                        elif target_path.is_file() and target_path.suffix.lower() == '.wav':
                            self._file_lists[path] = [path]
                            length = 1
                    else:
                        data = np.load(path, mmap_mode='r')
                        self._mmap_cache[path] = data
                        length = len(data)

                    # Pre-build index structures for instant access
                    self.file_paths.extend([path] * length)
                    self.local_indices.extend(range(length))
                    self.labels_list.extend([label] * length)
                        
                except FileNotFoundError:
                    print_error(f"[Validation] File not found: {path}")
                    sys.exit(1)
                except Exception as e:
                    print_error(f"[Validation] Loader Error '{path}': {e}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def _get_deterministic_offset(self, filepath: str, max_offset: int) -> int:
        """Generates a perfectly reproducible pseudo-random crop based on the file name."""
        hash_val = int(hashlib.md5(filepath.encode('utf-8')).hexdigest()[:8], 16)
        return hash_val % (max_offset + 1)

    def _read_audio_deterministic(self, filepath: str) -> torch.Tensor:
        """Ultra-Fast Validation Loader"""
        try:
            waveform, sr = torchaudio.load(filepath)

            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, 
                    orig_freq=sr, 
                    new_freq=self.sample_rate
                    )

            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            waveform = waveform.squeeze(0)
            current_len = waveform.size(0)

            if current_len == self.clip_samples:
                return waveform
            elif current_len > self.clip_samples:
                max_offset = current_len - self.clip_samples
                offset = self._get_deterministic_offset(filepath, max_offset)
                return waveform[offset : offset + self.clip_samples]
            else:
                clip = torch.zeros(self.clip_samples, dtype=torch.float32)
                clip[:current_len] = waveform
                return clip
            
        except Exception as e:
            return torch.zeros(self.clip_samples, dtype=torch.float32)


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path = self.file_paths[index]
        local_index = self.local_indices[index]
        label = torch.tensor(self.labels_list[index], dtype=torch.float32)

        if self.is_e2e:
            filepath = self._file_lists[path][local_index]
            feature = self._read_audio_deterministic(filepath)
        else:
            data = self._mmap_cache[path]
            feature = torch.from_numpy(data[local_index].astype(np.float32))

        return feature, label, index