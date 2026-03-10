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
import random
import torchaudio
import numpy as np
from typing import List
from multiprocessing import Pool, cpu_count

from nanowakeword.utils.logger import print_warning, print_info
from torch_audiomentations import Compose, PitchShift, Gain, ApplyImpulseResponse


def _load_clip_parallel(args):
    """A generic and robust helper for loading any audio clip for multiprocessing."""
    clip_path, sr = args
    try:
        waveform, clip_sr = torchaudio.load(clip_path)
        if clip_sr != sr:
            resampler = torchaudio.transforms.Resample(orig_freq=clip_sr, new_freq=sr)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform.squeeze(0)
    except Exception:
        return None

def _mix_snr(fg_wf, bg_wf, start_index, snr_db):
    fg_len = fg_wf.shape[0]
    bg_len = bg_wf.shape[0]

    if start_index + fg_len > bg_len:
        start_index = bg_len - fg_len
    if fg_len > bg_len:
        fg_wf = fg_wf[:bg_len]
        fg_len = bg_len
        start_index = 0

    bg_segment = bg_wf[start_index : start_index + fg_len]
    
    eps = torch.finfo(torch.float32).eps
    fg_rms = torch.sqrt(torch.mean(fg_wf**2) + eps)
    bg_rms = torch.sqrt(torch.mean(bg_segment**2) + eps)

    MIN_BG_RMS = 0.005
    bg_rms = torch.clamp(bg_rms, min=MIN_BG_RMS)

    snr_linear = 10**(snr_db / 20.0)
    scale = snr_linear * bg_rms / fg_rms
    
    scaled_fg = fg_wf * scale
    scaled_fg_rms = torch.sqrt(torch.mean(scaled_fg**2) + eps)
    
    # 0.01 = -40 dBFS, clearly audible but not loud
    MIN_FG_RMS = 0.01
    if scaled_fg_rms < MIN_FG_RMS:
        scale = scale * (MIN_FG_RMS / scaled_fg_rms)
    
    mixed_bg = bg_wf.clone()
    mixed_bg[start_index : start_index + fg_len] += fg_wf * scale
    
    return mixed_bg

def augment_clips(
        clip_paths: List[str],
        total_length: int,
        sr: int = 16000,
        batch_size: int = 128,
        augmentation_settings: dict = None,
        background_clip_paths: List[str] = [],
        RIR_paths: List[str] = [],
        num_workers: int = 0
        ):
    """
    Applies batch-wise audio augmentation to a list of audio clips.

    This function loads audio files, resamples them to a fixed sample rate,
    converts them to mono, and enforces a fixed length through random cropping
    or zero-padding. It then applies a configurable sequence of audio
    augmentations including gain, background noise, room impulse response (RIR),
    pitch shifting, and colored noise. The function is implemented as a
    generator for memory-efficient large-scale data processing.

    Args:
        clip_paths (List[str]):
            Paths to input audio files.
        total_length (int):
            Target number of samples per audio clip.
        sr (int, optional):
            Target sample rate in Hz. Defaults to 16000.
        batch_size (int, optional):
            Number of audio clips processed per batch. Defaults to 128.
        augmentation_settings (dict, optional):
            Dictionary to override default augmentation parameters. Supported keys:
                - min_snr_in_db (float): Minimum SNR for background noise.
                - max_snr_in_db (float): Maximum SNR for background noise.
                - rir_prob (float): Probability of applying RIR convolution.
                - pitch_prob (float): Probability of applying pitch shifting.
                - min_pitch_semitones (float): Minimum pitch shift in semitones.
                - max_pitch_semitones (float): Maximum pitch shift in semitones.
                - gain_prob (float): Probability of applying gain adjustment.
                - min_gain_in_db (float): Minimum gain in decibels.
                - max_gain_in_db (float): Maximum gain in decibels.
                - min_volume_augmentation
                - max_volume_augmentation


        background_clip_paths (List[str], optional):
            Paths to background noise audio files.
        RIR_paths (List[str], optional):
            Paths to room impulse response (RIR) files.

    Returns:
        Generator[np.ndarray]:
            Yields batches of augmented audio with shape
            (batch_size, total_length) and dtype int16.    


    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cfg = {
        "rir_prob": 0.5, "gain_prob": 1.0, "pitch_prob": 0.5,
        "min_pitch_semitones": -2.0, "max_pitch_semitones": 2.0,
        "max_snr_in_db": 30.0, "min_snr_in_db": 5.0,
        "min_gain_in_db": -3.0, "max_gain_in_db": 3.0,
        "min_volume_augmentation": 0.5, "max_volume_augmentation": 1.0 # For realistic volume levels
    }
    if augmentation_settings:
        cfg.update(augmentation_settings)

    # Augmenters that are applied *after* the primary SNR mixing
    post_mixing_augmenter = Compose([
        Gain(min_gain_in_db=cfg["min_gain_in_db"], max_gain_in_db=cfg["max_gain_in_db"], p=cfg["gain_prob"], sample_rate=sr),
        PitchShift(min_transpose_semitones=cfg["min_pitch_semitones"], max_transpose_semitones=cfg["max_pitch_semitones"], p=cfg["pitch_prob"], sample_rate=sr)
    ])
    if RIR_paths:
        post_mixing_augmenter.transforms.append(
            ApplyImpulseResponse(ir_paths=RIR_paths, p=cfg["rir_prob"], sample_rate=sr, output_type="dict")
        )
    post_mixing_augmenter.to(device)

    random.shuffle(clip_paths)

    if num_workers == -1: worker_count = cpu_count()
    elif num_workers > 0: worker_count = min(num_workers, cpu_count())
    else: worker_count = 0

    if worker_count > 0:
        pool = Pool(processes=worker_count)

    # Main processing loop
    for i in range(0, len(clip_paths), batch_size):
        fg_batch_paths = clip_paths[i:i+batch_size]
        current_batch_size = len(fg_batch_paths)

        # Select a UNIQUE random background for EACH foreground clip
        if background_clip_paths:
            bg_batch_paths = random.choices(background_clip_paths, k=current_batch_size)
        else: # Handle cases with no background noise provided
            bg_batch_paths = [None] * current_batch_size

        # Load BOTH foregrounds and backgrounds in PARALLEL for max performance
        fg_args = [(path, sr) for path in fg_batch_paths]
        bg_args = [(path, sr) for path in bg_batch_paths]

        if worker_count > 0:
            fg_waveforms = pool.map(_load_clip_parallel, fg_args)
            bg_waveforms = pool.map(_load_clip_parallel, bg_args)
        else:
            fg_waveforms = [_load_clip_parallel(arg) for arg in fg_args]
            bg_waveforms = [_load_clip_parallel(arg) for arg in bg_args]
        
        # Core mixing loop (in-memory, fast)
        mixed_batch = []
        for fg_wf, bg_wf in zip(fg_waveforms, bg_waveforms):
            if fg_wf is None or len(fg_wf) == 0: continue

            # Prepare background: use silence if loading failed or wasn't provided
            if bg_wf is None or len(bg_wf) == 0:
                bg_wf = torch.zeros(total_length)
            else:
                # Ensure background is the correct length by tiling or snipping
                if len(bg_wf) < total_length: bg_wf = bg_wf.repeat(int(np.ceil(total_length / len(bg_wf))))
                if len(bg_wf) > total_length:
                    start = random.randint(0, len(bg_wf) - total_length)
                    bg_wf = bg_wf[start : start + total_length]
            
            # Prepare foreground: if it's too long, take a random snippet
            fg_len = fg_wf.shape[0]
            if fg_len > total_length:
                start_fg = random.randint(0, fg_len - total_length)
                fg_wf = fg_wf[start_fg : start_fg + total_length]
                fg_len = total_length
            
            # Mix at a random position with a random SNR
            start_index = random.randint(0, total_length - fg_len)
            snr_db = random.uniform(cfg["min_snr_in_db"], cfg["max_snr_in_db"])
            
            mixed_clip = _mix_snr(fg_wf, bg_wf, start_index, snr_db)
            mixed_batch.append(mixed_clip)

        if not mixed_batch: continue

        # Apply post-mixing augmentations on GPU
        batch_tensor = torch.stack(mixed_batch).unsqueeze(1).to(device)
        with torch.no_grad():
            result = post_mixing_augmenter(samples=batch_tensor, sample_rate=sr)
            try:
                augmented_tensor = result["samples"]
            except:
                augmented_tensor = result

        # Intelligent Volume Augmentation (replaces bad peak normalization)
        target_levels = torch.FloatTensor(augmented_tensor.shape[0]).uniform_(
            cfg["min_volume_augmentation"], cfg["max_volume_augmentation"]
        ).to(device)
        
        # Calculate current peak absolute values for each clip in the batch
        current_peaks = torch.max(torch.abs(augmented_tensor), dim=2, keepdim=True)[0]
        current_peaks[current_peaks < 1e-8] = 1.0 # Avoid division by zero for silent clips

        # Scale each clip to its new target level
        scaled_tensor = augmented_tensor * (target_levels.unsqueeze(1).unsqueeze(2) / current_peaks)

        # Final clipping to prevent any overshoot and convert to int16
        final_tensor = torch.clamp(scaled_tensor, -1.0, 1.0)
        output_batch = (final_tensor.cpu().numpy() * 32767).astype(np.int16)
        
        yield output_batch.squeeze(1)
        
    if worker_count > 0:
        pool.close()
        pool.join()