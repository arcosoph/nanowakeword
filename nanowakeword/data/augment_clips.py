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


def augment_clips(
        clip_paths: List[str],
        total_length: int,
        sr: int = 16000,
        batch_size: int = 128,
        augmentation_settings: dict = None,
        background_clip_paths: List[str] = [],
        RIR_paths: List[str] = []
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
        background_clip_paths (List[str], optional):
            Paths to background noise audio files.
        RIR_paths (List[str], optional):
            Paths to room impulse response (RIR) files.

    Returns:
        Generator[np.ndarray]:
            Yields batches of augmented audio with shape
            (batch_size, total_length) and dtype int16.
    """

    from torch_audiomentations import Compose, AddBackgroundNoise, ApplyImpulseResponse, PitchShift, Gain, AddColoredNoise

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cfg = {
        "min_snr_in_db": 3.0, "max_snr_in_db": 30.0,
        "rir_prob": 0.2,
        "pitch_prob": 0.3, "min_pitch_semitones": -2.0, "max_pitch_semitones": 2.0,
        "gain_prob": 1.0, "min_gain_in_db": -6.0, "max_gain_in_db": 6.0
    }
    if augmentation_settings:
        cfg.update(augmentation_settings)

    transforms = [
        Gain(min_gain_in_db=cfg["min_gain_in_db"], max_gain_in_db=cfg["max_gain_in_db"], p=cfg["gain_prob"], sample_rate=sr),
    ]

    if background_clip_paths: 
        transforms.append(
            AddBackgroundNoise(background_paths=background_clip_paths, min_snr_in_db=cfg["min_snr_in_db"], max_snr_in_db=cfg["max_snr_in_db"], p=0.8, sample_rate=sr)
        )

    if RIR_paths: 
        transforms.append(
            ApplyImpulseResponse(ir_paths=RIR_paths, p=cfg["rir_prob"], sample_rate=sr)
        )

    transforms.append(
        PitchShift(min_transpose_semitones=cfg["min_pitch_semitones"], max_transpose_semitones=cfg["max_pitch_semitones"], p=cfg["pitch_prob"], sample_rate=sr)
    )    
    
    transforms.append(
        AddColoredNoise(min_snr_in_db=20.0, max_snr_in_db=40.0, p=1.0, sample_rate=sr)
    )

    augmenter = Compose(transforms=transforms, output_type="dict")
    augmenter.to(device)

    # The list is being shuffled so that a new batch is created each time.
    random.shuffle(clip_paths)

    for i in range(0, len(clip_paths), batch_size):
        batch_paths = clip_paths[i:i+batch_size]
        
        batch_audio = []
        for clip_path in batch_paths:
            try:
                # Load the audio and trim it to the correct length
                waveform, clip_sr = torchaudio.load(clip_path)
                if clip_sr != sr:
                    waveform = torchaudio.transforms.Resample(orig_freq=clip_sr, new_freq=sr)(waveform)
                
                # Mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                current_len = waveform.shape[1]
                if current_len > total_length:
                    start = random.randint(0, current_len - total_length)
                    waveform = waveform[:, start:start+total_length]
                elif current_len < total_length:
                    padding_needed = total_length - current_len
                    start_pad = random.randint(0, padding_needed)
                    end_pad = padding_needed - start_pad
                    waveform = torch.nn.functional.pad(waveform, (start_pad, end_pad))
                
                batch_audio.append(waveform)

            except Exception as e:
                print(f"Warning: Skipping corrupted file {clip_path}: {e}")
                continue
        
        if not batch_audio:
            continue
        batch_tensor = torch.stack(batch_audio).to(device)
        
        with torch.no_grad():
            augmented_tensor = augmenter(samples=batch_tensor, sample_rate=sr)['samples']

        final_batch_np = augmented_tensor.cpu().numpy()
        
        # Normalization (peak normalization is safe here because it is done at the end of all augmentation)
        max_val = np.abs(final_batch_np).max(axis=-1, keepdims=True)
        max_val[max_val < 1e-8] = 1.0 
        
        # The output will be in the form (batch_size, samples)
        output_batch = (final_batch_np / max_val).squeeze(1) 
        
        yield (output_batch * 32767).astype(np.int16)

