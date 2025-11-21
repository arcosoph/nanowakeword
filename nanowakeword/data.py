# Copyright 2022 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#####################################################
# Modified by Muhammad Abid
#  For more information, visit the official repository:
#       https://github.com/arcosoph/nanowakeword
######################################################


import os
import re
import torch
import random
import logging
import mutagen
import acoustics
import itertools
import torchaudio
import numpy as np
import pronouncing
from tqdm import tqdm
from pathlib import Path
from functools import partial
from typing import List, Tuple
from numpy.lib.format import open_memmap
from multiprocessing.pool import ThreadPool
from scipy.signal import convolve
from torch_audiomentations import Compose, Gain, PitchShift, AddColoredNoise, BandStopFilter, ApplyImpulseResponse



# Load audio clips and structure into clips of the same length
def stack_clips(audio_data, clip_size=16000*2):
    """
    Takes an input list of 1D arrays (of different lengths), concatenates them together,
    and then extracts clips of a uniform size by dividing the combined array.

    Args:
        audio_data (List[ndarray]): A list of 1D numpy arrays to combine and stack
        clip_size (int): The desired total length of the uniform clip size (in samples)

    Returns:
        ndarray: A N by `clip_size` array with the audio data, converted to 16-bit PCM
    """

    # Combine all clips into single clip
    combined_data = np.hstack((audio_data))

    # Get chunks of the specified size
    new_examples = []
    for i in range(0, combined_data.shape[0], clip_size):
        chunk = combined_data[i:i+clip_size]
        if chunk.shape[0] != clip_size:
            chunk = np.hstack((chunk, np.zeros(clip_size - chunk.shape[0])))
        new_examples.append(chunk)

    return np.array(new_examples)


def load_audio_clips(files, clip_size=32000):
    """
    Takes the specified audio files and shapes them into an array of N by `clip_size`,
    where N is determined by the length of the audio files and `clip_size` at run time.

    Clips longer than `clip size` are truncated and extended into the N+1 row.
    Clips shorter than `clip_size` are combined with the previous or next clip
    (except for the last clip in `files`, which is ignored if it is too short.)

    Args:
        files (List[str]): A list of filepaths
        clip_size (int): The number of samples (of 16khz audio) for all of the rows in the array

    Returns:
        ndarray: A N by `clip_size` array with the audio data, converted to 16-bit PCM
    """

    # Load audio files
    audio_data = []
    for i in files:
        try:
            # audio_data.append(read_audio(i))
            audio_data.append(torchaudio.load(i)[0])
        except ValueError:
            continue

    # Get shape of output array
    N = sum([i.shape[0] for i in audio_data])//clip_size
    X = np.empty((N, clip_size))

    # Add audio data to rows
    previous_row_remainder = None
    cnt = 0
    for row in audio_data:
        row = np.hstack((previous_row_remainder, row))
        while row.shape[0] >= clip_size:
            X[cnt, :] = row[0:clip_size]
            row = row[clip_size:]
            cnt += 1

        previous_row_remainder = row if row.size > 0 else None

    # Convert to 16-bit PCM data
    X = (X*32767).astype(np.int16)

    return X


# Dato I/O utils


# Convert clips with sox
def _convert_clip(input_file, output_file, backend="ffmpeg"):
    if backend == "sox":
        cmd = f"sox \"{input_file}\" -G -r 16000 -c 1 -b 16 \"{output_file}\""
    elif backend == "ffmpeg":
        cmd = f"ffmpeg -y -i \"{input_file}\" -ar 16000 \"{output_file}\""
    os.system(cmd)
    return None


def convert_clips(input_files, output_files, sr=16000, ncpu=1, backend="ffmpeg"):
    """
    Converts files in parallel with multithreading using Sox or ffmpeg.

    Intended to only convert input audio files into single-channel, 16 khz clips.

    Args:
        input_files (List[str]): A list of paths to input files
        output_files (List[str]): A list of paths to output files, corresponding 1:1 to the input files
        sr (int): The output sample rate of the converted clip
        ncpu (int): The number of CPUs to use for the conversion
        backend (str): The utilty to use for conversion, "sox" or "ffmpeg"

    Returns:
        None
    """
    # Setup ThreadPool object
    pool = ThreadPool(processes=ncpu)

    # Set backend for conversion
    f = partial(_convert_clip, backend=backend)

    # Submit jobs
    pool.starmap(f, [(i, j) for i, j in zip(input_files, output_files)])


def filter_audio_paths(target_dirs, min_length_secs, max_length_secs, duration_method="size", glob_filter=None):
    """
    Gets the paths of wav files in flat target directories, automatically filtering
    out files below/above the specified length (in seconds). Assumes that all
    wav files are sampled at 16khz, are single channel, and have 16-bit PCM data.

    Uses `os.scandir` in Python for highly efficient file system exploration,
    and doesn't require loading the files into memory for length estimation.

    Args:
        target_dir (List[str]): The target directories containing the audio files
        min_length_secs (float): The minimum length in seconds (otherwise the clip is skipped)
        max_length_secs (float): The maximum length in seconds (otherwise the clip is skipped)
        duration_method (str): Whether to use the file size ('size'), or header information ('header')
                               to estimate the duration of the audio file. 'size' is generally
                               much faster, but assumes that all files in the target directory
                               are the same type, sample rate, and bitrate. If None, durations are not calculated.
        glob_filter (str): A pathlib glob filter string to select specific files within the target directory

    Returns:
        tuple: A list of strings corresponding to the paths of the wav files that met the length criteria,
               and a list of their durations (in seconds)
    """

    file_paths = []
    durations = []
    for target_dir in target_dirs:
        sizes = []
        dir_paths = []
        if glob_filter:
            dir_paths = [str(i) for i in Path(target_dir).glob(glob_filter)]
            file_paths.extend(dir_paths)
            sizes.extend([os.path.getsize(i) for i in dir_paths])
        else:
            for i in tqdm(os.scandir(target_dir)):
                dir_paths.append(i.path)
                file_paths.append(i.path)
                sizes.append(i.stat().st_size)

        if duration_method == "size":
            durations.extend(estimate_clip_duration(dir_paths, sizes))

        elif duration_method == "header":
            durations.extend([get_clip_duration(i) for i in tqdm(dir_paths)])

    if durations != []:
        filtered = [(i, j) for i, j in zip(file_paths, durations) if j >= min_length_secs and j <= max_length_secs]
        return [i[0] for i in filtered], [i[1] for i in filtered]
    else:
        return file_paths, []


def estimate_clip_duration(audio_files: list, sizes: list):
    """Estimates the duration of each audio file in a list.

    Assumes that all of the audio files have the same audio format,
    bit depth, and sample rate.

    Args:
        audio_file (str): A list of audio file paths
        sizes (int): The size of each audio file in bytes

    Returns:
        list: A list of durations (in seconds) for the audio files
    """

    # Determine file type by checking the first file
    details = torchaudio.info(audio_files[0])

    # Caculate any correction factors needed from the first file
    details = mutagen.File(audio_files[0])
    correction = 8*os.path.getsize(audio_files[0]) - details.info.bitrate*details.info.length

    # Estimate duration for all remaining clips from file size only
    durations = []
    for size in sizes:
        durations.append((size*8-correction)/details.info.bitrate)

    return durations


def estimate_mp3_duration(fpath):
    """Estimates the duration of an MP3 file from metadata and file-size.
    Is only accurate for 16000 khz sample rate audio with a relatively
    constant bit-rate.

    Args:
        fpath (str): The input path to the MP3 file

    Returns:
        float: The duration of the MP3 file in seconds
    """

    conversion_factors = {
        "16_khz_single_channel": 0.000333318208471784,
        "16_khz_stereo": 0.000333318208471784/2
    }

    duration_seconds = 0
    try:
        md = torchaudio.info(fpath)
    except RuntimeError:
        return duration_seconds

    nbytes = os.path.getsize(fpath)
    if md.num_channels == 1:
        if md.sample_rate == 16000:
            duration_seconds = nbytes*conversion_factors["16_khz_single_channel"]
    elif md.num_channels == 2:
        if md.sample_rate == 16000:
            duration_seconds = nbytes*conversion_factors["16_khz_stereo"]

    return duration_seconds


def get_clip_duration(clip):
    """Gets the duration of an audio clip in seconds from file header information"""
    try:
        metadata = torchaudio.info(clip)
    except RuntimeError:  # skip cases where file metadata can't be read
        return 0

    return metadata.num_frames/metadata.sample_rate


def get_wav_duration_from_filesize(size, nbytes=2):
    """
    Calculates the duration (in seconds) from a WAV file, assuming it contains 16 khz single-channel audio.
    The bit depth is user specified, and defaults to 2 for 16-bit PCM audio.

    Args:
        size (int): The file size in bytes
        nbytes (int): How many bytes for each data point in the audio (e.g., 16-bit is 2, 32-bit is 4, etc.)

    Returns:
        float: The duration of the audio file in seconds
    """
    return (size-44)/nbytes/16000


# Data augmentation utility function
def mix_clips_batch(
        foreground_clips: List[str],
        background_clips: List[str],
        combined_size: int,
        labels: List[int] = [],
        batch_size: int = 32,
        snr_low: float = 0,
        snr_high: float = 0,
        start_index: List[int] = [],
        foreground_durations: List[float] = [],
        foreground_truncate_strategy: str = "random",
        rirs: List[str] = [],
        rir_probability: int = 1,
        volume_augmentation: bool = True,
        generated_noise_augmentation: float = 0.0,
        shuffle: bool = True,
        return_sequence_labels: bool = False,
        return_background_clips: bool = False,
        return_background_clips_delay: Tuple[int, int] = (0, 0),
        seed: int = 0
        ):
    """
    Mixes foreground and background clips at a random SNR level in batches.

    Args:
        foreground_clips (List[str]): A list of paths to the foreground clips
        background_clips (List[str]): A list of paths to the background clips (randomly selected for each
                                      foreground clip)
        combined_size (int): The total length (in samples) of the combined clip. If needed, the background
                             clips are duplicated or truncated to reach this length.
        labels (List[int]): A list of integer labels corresponding 1:1 for the foreground clips. Will be updated
                            as needed with foreground clips to ensure that mixed clips retain the proper labels.
        batch_size (int): The batch size
        snr_low (float): The low SNR level of the mixing in db
        snr_high (float): The high snr level of the mixing in db
        start_index (List[int]): The starting position (in samples) for the foreground clip to start in
                                 the background clip. If the foreground clip is longer than `combined_size`
                                 when starting at this point, the foreground clip will be truncated
                                 according to the `foreground_truncate_strategy` argument.
        foreground_durations (List[float]): The desired duration of each foreground clip (in seconds)
        foreground_truncate_strategy (str): The method used to truncate the foreground clip, if needed based on the
                                            `start_index`, `foreground_durations`, and `combined_size` arguments.
                                            See the options in the `truncate_clip` method.
        rirs (List[str]): A list of paths to room impulse response functions (RIR) to convolve with the
                          clips to simulate different recording environments. Applies a single random selection from the
                          list RIR file to the entire batch. If empty (the default), nothing is done.
        rir_probability (float): The probability (between 0 and 1) that the batch will be convolved with a RIR file.
        volume_augmentation (bool): Whether to randomly apply volume augmentation to the clips in the batch.
                                    This simply scales the data of each clip such that the maximum value is is between
                                    0.02 and 1.0 (the floor shouldn't be zero as beyond a certain point the audio data
                                    is no longer valid).
        generated_noise_augmentation: The probability of further mixing the mixed clip with generated random noise.
                                      Will be either "white", "brown", "blue", "pink", or "violet" noise, mixed at a
                                      random SNR between `snr_low` and `snr_high`.
        return_background_clips (bool): Whether to return the segment of the background clip that was mixed with each
                                        foreground clip in the batch.
        return_background_clips_delay (Tuple(int)): The lower and upper bound of a random delay (in samples)
                                           to apply to the segment of each returned backgroud clip mixed
                                           with each foreground clip in the batch. This is primarily intended to
                                           simulate the drift between input and output channels
                                           in audio devices, which means that the mixed audio is never
                                           exactly aligned with the two source clips.
        shuffle (bool): Whether to shuffle the foreground clips before mixing (default: True)
        return_sequence_labels (bool): Whether to return sequence labels (i.e., frame-level labels) for each clip
                                       based on the start/end positions of the foreground clip.
        seed (int): A random seed

    Returns:
        generator: Returns a generator that yields batches of mixed foreground/background audio, labels, and the
                   background segments used for each audio clip (or None is the
                   `return_backgroun_clips` argument is False)
    """
    # Set random seed, if needed
    if seed:
        np.random.seed(seed)
        random.seed(seed)

    # Check and Set start indices, if needed
    if not start_index:
        start_index = [0]*batch_size
    else:
        if min(start_index) < 0:
            raise ValueError("Error! At least one value of the `start_index` argument is <0. Check your inputs.")

    # Make dummy labels
    if not labels:
        labels = [0]*len(foreground_clips)

    if shuffle:
        p = np.random.permutation(len(foreground_clips))
        foreground_clips = np.array(foreground_clips)[p].tolist()
        start_index = np.array(start_index)[p].tolist()
        labels = np.array(labels)[p].tolist()
        if foreground_durations:
            foreground_durations = np.array(foreground_durations)[p].tolist()

    for i in range(0, len(foreground_clips), batch_size):
        # Load foreground clips/start indices and truncate as needed
        sr = 16000
        start_index_batch = start_index[i:i+batch_size]
        
        foreground_clips_batch = [torchaudio.load(j)[0] for j in foreground_clips[i:i+batch_size]]

        foreground_clips_batch = [j[0] if len(j.shape) > 1 else j for j in foreground_clips_batch]

        if foreground_durations:
            foreground_clips_batch = [truncate_clip(j, int(k*sr), foreground_truncate_strategy)
                                      for j, k in zip(foreground_clips_batch, foreground_durations[i:i+batch_size])]
        labels_batch = np.array(labels[i:i+batch_size])

        # Load background clips and pad/truncate as needed
        background_clips_batch = [torchaudio.load(j)[0] for j in random.sample(background_clips, batch_size)]
        background_clips_batch = [j[0] if len(j.shape) > 1 else j for j in background_clips_batch]

        background_clips_batch_delayed = []
        delay = np.random.randint(return_background_clips_delay[0], return_background_clips_delay[1] + 1)
        for ndx, background_clip in enumerate(background_clips_batch):
            if background_clip.shape[0] < (combined_size + delay):
                repeated = background_clip.repeat(
                    np.ceil((combined_size + delay)/background_clip.shape[0]).astype(np.int32)
                )
                background_clips_batch[ndx] = repeated[0:combined_size]
                background_clips_batch_delayed.append(repeated[0+delay:combined_size + delay].clone())
            elif background_clip.shape[0] > (combined_size + delay):
                r = np.random.randint(0, max(1, background_clip.shape[0] - combined_size - delay))
                background_clips_batch[ndx] = background_clip[r:r + combined_size]
                background_clips_batch_delayed.append(background_clip[r+delay:r + combined_size + delay].clone())

        # Mix clips at snr levels
        snrs_db = np.random.uniform(snr_low, snr_high, batch_size)
        mixed_clips = []
        sequence_labels = []
        for fg, bg, snr, start in zip(foreground_clips_batch, background_clips_batch,
                                      snrs_db, start_index_batch):
            if bg.shape[0] != combined_size:
                raise ValueError(bg.shape)
            mixed_clip = mix_clip(fg, bg, snr, start)
            sequence_labels.append(get_frame_labels(combined_size, start, start+fg.shape[0]))

            if np.random.random() < generated_noise_augmentation:
                noise_color = ["white", "pink", "blue", "brown", "violet"]
                noise_clip = acoustics.generator.noise(combined_size, color=np.random.choice(noise_color))
                noise_clip = torch.from_numpy(noise_clip/noise_clip.max())
                mixed_clip = mix_clip(mixed_clip, noise_clip, np.random.choice(snrs_db), 0)

            mixed_clips.append(mixed_clip)

        mixed_clips_batch = torch.vstack(mixed_clips)
        sequence_labels_batch = torch.from_numpy(np.vstack(sequence_labels))

        
        # Apply reverberation to the batch (from a single RIR file)
        if rirs:
            if np.random.random() <= rir_probability:
                rir_waveform, sr = torchaudio.load(random.choice(rirs))
                if rir_waveform.shape[0] > 1:
                    rir_waveform = rir_waveform[random.randint(0, rir_waveform.shape[0]-1), :]

                rir_numpy = rir_waveform.numpy().flatten()
                reverbed_clips = []
    
                for clip_tensor in mixed_clips_batch:
                    
                    clip_numpy = clip_tensor.cpu().numpy()
                    
                    reverbed_clip = convolve(clip_numpy, rir_numpy, mode='same')
                    reverbed_clips.append(torch.from_numpy(reverbed_clip))

                mixed_clips_batch = torch.stack(reverbed_clips)

                abs_max, _ = torch.max(torch.abs(mixed_clips_batch), dim=1, keepdim=True)
                
                mixed_clips_batch = mixed_clips_batch / abs_max.clamp(min=1.0)


        # Apply volume augmentation
        if volume_augmentation:
            volume_levels = np.random.uniform(0.02, 1.0, mixed_clips_batch.shape[0])
            mixed_clips_batch = (volume_levels/mixed_clips_batch.max(dim=1)[0])[..., None]*mixed_clips_batch
        else:
            # Normalize clips only if max value is outside of [-1, 1]
            abs_max, _ = torch.max(
                torch.abs(mixed_clips_batch), dim=1, keepdim=True
            )
            mixed_clips_batch = mixed_clips_batch / abs_max.clamp(min=1.0)

        # Convert to 16-bit PCM audio
        mixed_clips_batch = (mixed_clips_batch.numpy()*32767).astype(np.int16)

        # Remove any clips that are silent (happens rarely when mixing/reverberating)
        error_index = torch.from_numpy(np.where(mixed_clips_batch.max(dim=1) != 0)[0])
        mixed_clips_batch = mixed_clips_batch[error_index]
        labels_batch = labels_batch[error_index]
        sequence_labels_batch = sequence_labels_batch[error_index]

        if not return_background_clips:
            yield mixed_clips_batch, labels_batch if not return_sequence_labels else sequence_labels_batch, None
        else:
            background_clips_batch_delayed = (torch.vstack(background_clips_batch_delayed).numpy()
                                              * 32767).astype(np.int16)[error_index]
            yield (mixed_clips_batch,
                   labels_batch if not return_sequence_labels else sequence_labels_batch,
                   background_clips_batch_delayed)


def get_frame_labels(combined_size, start, end, buffer=1):
    sequence_label = np.zeros(np.ceil((combined_size-12400)/1280).astype(int))
    frame_positions = np.arange(12400, combined_size, 1280)
    start_frame = np.argmin(abs(frame_positions - start))
    end_frame = np.argmin(abs(frame_positions - end))
    sequence_label[start_frame:start_frame+2] = 1
    sequence_label[end_frame-1:end_frame+1] = 1
    return sequence_label


def mix_clip(fg, bg, snr, start):
    fg_rms, bg_rms = fg.norm(p=2), bg.norm(p=2)
    snr = 10 ** (snr / 20)
    scale = snr * bg_rms / fg_rms
    bg[start:start + fg.shape[0]] = bg[start:start + fg.shape[0]] + scale*fg
    return bg / 2


def truncate_clip(x, max_size, method="truncate_start"):
    """
    Truncates and audio clip with the specified method

    Args:
        x (nd.array): An array of audio data
        max_size (int): The maximum size (in samples)
        method (str): Can be one of four options:
            - "truncate_start": Truncate the start of the clip
            - "truncate_end": Truncate the end of the clip
            - "truncate_both": Truncate both the start and end of the clip
            - "random": Randomly select a segment of the right size from the clip

    Returns:
        nd.array: The truncated audio data
    """
    if x.shape[0] > max_size:
        if method == "truncate_start":
            x = x[x.shape[0] - max_size:]
        if method == "truncate_end":
            x = x[0:max_size]
        if method == "truncate_both":
            n = int(np.ceil(x.shape[0] - max_size)/2)
            x = x[n:-n][0:max_size]
        if method == "random":
            rn = np.random.randint(0, x.shape[0] - max_size)
            x = x[rn:rn + max_size]

    return x


# Reverberation data augmentation function
def apply_reverb(x, rir_files):
    """
    Applies reverberation to the input audio clips

    Args:
        x (nd.array): A numpy array of shape (batch, audio_samples) containing the audio clips
        rir_files (Union[str, list]): Either a path to an RIR (room impulse response) file or a list
                                      of RIR files. If a list, one file will be randomly chosen
                                      to apply to `x`

    Returns:
        nd.array: The reverberated audio clips
    """

    if isinstance(rir_files, str):
        rir_waveform, sr = torchaudio.load(rir_files) 
    elif isinstance(rir_files, list):
        rir_waveform, sr = torchaudio.load(random.choice(rir_files))

    if rir_waveform.shape[0] > 1:
        rir_waveform = rir_waveform[random.randint(0, rir_waveform.shape[0]-1), :]
    
    rir_numpy = rir_waveform.numpy().flatten()
    reverbed_clips = []
    # Apply convolution to each clip of the input NumPy array 'x'
    for clip_numpy in x:
        reverbed_clip = convolve(clip_numpy, rir_numpy, mode='same')
        reverbed_clips.append(reverbed_clip)

    reverbed_numpy = np.stack(reverbed_clips)

    abs_max = np.max(np.abs(reverbed_numpy), axis=1, keepdims=True)
    
    reverbed_numpy = reverbed_numpy / np.maximum(abs_max, 1e-7)

    return reverbed_numpy


def _add_background_noise_manual(samples: torch.Tensor, background_paths: List[str], min_snr: float, max_snr: float, sr: int) -> torch.Tensor:
    """
    Optimized background noise injection.
    """
    device = samples.device
    batch_size = samples.shape[0]
    
    if not background_paths:
        return samples

    for i in range(batch_size):
        if random.random() < 0.5: 
            continue

        try:
            noise_path = random.choice(background_paths)
            noise_waveform, noise_sr = torchaudio.load(noise_path)
            
            if noise_sr != sr:
                noise_waveform = torchaudio.functional.resample(noise_waveform, noise_sr, sr)
            
            noise_waveform = noise_waveform[0].to(device)
            target_length = samples.shape[2] # [batch, channel, time]
            
            if len(noise_waveform) < target_length:
                repeats = (target_length // len(noise_waveform)) + 1
                noise_waveform = noise_waveform.repeat(repeats)
            
            start_index = random.randint(0, len(noise_waveform) - target_length)
            noise_snippet = noise_waveform[start_index : start_index + target_length]

            sample_rms = samples[i, 0, :].pow(2).mean().sqrt()
            noise_rms = noise_snippet.pow(2).mean().sqrt()
            
            snr_db = random.uniform(min_snr, max_snr)
            
            if noise_rms > 1e-9:
                desired_noise_rms = sample_rms / (10**(snr_db / 20))
                samples[i, 0, :] += noise_snippet * (desired_noise_rms / noise_rms)

        except Exception as e:
            continue

    return samples


def augment_clips(
        clip_paths: List[str],
        total_length: int,
        sr: int = 16000,
        batch_size: int = 128,
        augmentation_settings: dict = None,
        background_clip_paths: List[str] = [],
        RIR_paths: List[str] = [],
        min_snr_in_db: float = 5.0,  
        max_snr_in_db: float = 25.0,
        end_jitter_ms: int = 200
        ):
    """
    UPDATED: Uses 'per_example' mode for diverse augmentation within batches.
    """
    if augmentation_settings is None:
        augmentation_settings = {}
    
    default_probs = {
        "Gain": 1.0, 
        "PitchShift": 0.5,        
        "BandStopFilter": 0.2, 
        "ColoredNoise": 0.4,      
        "BackgroundNoise": 0.7,   
        "RIR": 0.5                
    }
    final_aug_probs = {**default_probs, **augmentation_settings}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    augmenter = Compose([
        Gain(min_gain_in_db=-12.0, max_gain_in_db=6.0, p=final_aug_probs.get("Gain", 0.0), p_mode="per_example"),
        PitchShift(min_transpose_semitones=-3.0, max_transpose_semitones=3.0, p=final_aug_probs.get("PitchShift", 0.0), sample_rate=sr, p_mode="per_example"),
        BandStopFilter(min_center_frequency=200, max_center_frequency=3500, p=final_aug_probs.get("BandStopFilter", 0.0), p_mode="per_example"),
        AddColoredNoise(min_snr_in_db=5.0, max_snr_in_db=25.0, p=final_aug_probs.get("ColoredNoise", 0.0), p_mode="per_example"),
        ApplyImpulseResponse(p=final_aug_probs.get("RIR", 0.0), ir_paths=RIR_paths, p_mode="per_example", compensate_for_propagation_delay=True)
    ])
    
    augmenter.to(device)

    for i in range(0, len(clip_paths), batch_size):
        batch_paths = clip_paths[i:i+batch_size]
        processed_clips = []
        
        for clip_path in batch_paths:
            try:
                # Torchaudio load is safer inside try-except
                waveform, sample_rate = torchaudio.load(clip_path)
                if sample_rate != sr:
                    waveform = torchaudio.functional.resample(waveform, sample_rate, sr)
                
                waveform = waveform[0] 

                # Smart Padding/Cutting
                if waveform.shape[0] > total_length:
                    start_index = random.randint(0, waveform.shape[0] - total_length)
                    processed_clip = waveform[start_index : start_index + total_length]
                else:
                    padding = total_length - waveform.shape[0]
                    offset = random.randint(0, padding) 
                    padded_clip = torch.zeros(total_length)
                    padded_clip[offset : offset + waveform.shape[0]] = waveform
                    processed_clip = padded_clip
                
                processed_clips.append(processed_clip)
            except Exception:
                continue
        
        if not processed_clips:
            continue

        # GPU Processing
        batch_tensor = torch.stack(processed_clips).unsqueeze(1).to(device) # [B, 1, T]
        
        # 1. GPU Augmentations
        augmented_batch = augmenter(samples=batch_tensor, sample_rate=sr)
        
        # 2. Manual Background Noise (Real Noise)
        if final_aug_probs.get("BackgroundNoise", 0.0) > 0:
            augmented_batch = _add_background_noise_manual(
                augmented_batch, background_clip_paths, min_snr_in_db, max_snr_in_db, sr
            )
        
        # 3. Normalize (Safety check against exploding gradients)
        max_val, _ = torch.max(torch.abs(augmented_batch), dim=2, keepdim=True)
        augmented_batch = augmented_batch / (max_val + 1e-8)

        # Yield as Int16 numpy array
        yield (augmented_batch.squeeze(1).cpu().numpy() * 32767).astype(np.int16)


def create_fixed_size_clip(x, n_samples, sr=16000, start=None, end_jitter=.200):
    """
    Create a fixed-length clip of the specified size by padding an input clip with zeros
    Optionally specify the start/end position of the input clip, or let it be chosen randomly.

    Args:
        x (ndarray): The input audio to pad to a fixed size
        n_samples (int): The total number of samples for the fixed length clip
        sr (int): The sample rate of the audio
        start (int): The start position of the clip in the fixed length output, in samples (default: None)
        end_jitter (float): The time (in seconds) from the end of the fixed length output
                            that the input clip should end, if `start` is None.

    Returns:
        ndarray: A new array of audio data of the specified length
    """
    dat = np.zeros(n_samples)
    end_jitter = int(np.random.uniform(0, end_jitter)*sr)
    if start is None:
        start = max(0, n_samples - (int(len(x))+end_jitter))

    if len(x) > n_samples:
        if np.random.random() >= 0.5:
            dat = x[0:n_samples].numpy()
        else:
            dat = x[-n_samples:].numpy()
    else:
        dat[start:start+len(x)] = x

    return dat


def mmap_batch_generator(data_files, n_per_class, data_transform_funcs, label_transform_funcs, triplet_mode=True):
    """
    SMART BATCH GENERATOR: Prioritizes 'negative_speech' in Triplet Mode.
    """
    memmaps = {name: np.load(path, mmap_mode='r') for name, path in data_files.items()}
    class_indices = {name: np.arange(len(arr)) for name, arr in memmaps.items()}
    
    positive_key = 'positive'
    
    neg_speech_key = 'adversarial_negative' 

    if neg_speech_key not in data_files:
        possible_keys = [k for k in data_files.keys() if 'negative' in k and k != 'pure_noise']
        neg_speech_key = possible_keys[0] if possible_keys else None

    pure_noise_key = 'pure_noise'
    
    all_negative_keys = [k for k in data_files.keys() if k != positive_key]

    while True:
        if not triplet_mode:
            # Standard Classification Mode (No changes needed mostly)
            batch_x, batch_y = [], []
            for name, n_samples in n_per_class.items():
                if n_samples > 0 and name in memmaps:
                    indices = np.random.choice(class_indices[name], n_samples, replace=True)
                    data = memmaps[name][indices]
                    if name in data_transform_funcs: data = data_transform_funcs[name](data)
                    batch_x.append(data)
                    if name in label_transform_funcs: batch_y.extend(label_transform_funcs[name](data))
            
            if batch_x:
                yield torch.from_numpy(np.vstack(batch_x)).float(), torch.from_numpy(np.array(batch_y)).float()

        else:
            # SMART TRIPLET MODE 
            n_pos = n_per_class.get(positive_key, 32) # Default 32 if missing
            
            # 1. Anchor & Positive (From Positive Class)
            anchor_idxs = np.random.choice(class_indices[positive_key], n_pos, replace=True)
            pos_idxs = np.random.choice(class_indices[positive_key], n_pos, replace=True)
            
            anchors = memmaps[positive_key][anchor_idxs]
            positives = memmaps[positive_key][pos_idxs]

            # 2. Smart Negative Selection (Hard Negative Priority)
            negatives = []
            
            n_hard_neg = int(n_pos * 0.7) 
            n_easy_neg = n_pos - n_hard_neg
            
            # Hard Negatives (Speech)
            if neg_speech_key and neg_speech_key in memmaps:
                hard_idxs = np.random.choice(class_indices[neg_speech_key], n_hard_neg, replace=True)
                negatives.append(memmaps[neg_speech_key][hard_idxs])
            else:
                # If there is no specific speech negative, take a random negative
                n_easy_neg += n_hard_neg 

            # Easy Negatives (Noise)
            if pure_noise_key in memmaps:
                easy_idxs = np.random.choice(class_indices[pure_noise_key], n_easy_neg, replace=True)
                negatives.append(memmaps[pure_noise_key][easy_idxs])
            elif all_negative_keys:
                # If there is no noise, then any other negative
                rnd_key = np.random.choice(all_negative_keys)
                easy_idxs = np.random.choice(class_indices[rnd_key], n_easy_neg, replace=True)
                negatives.append(memmaps[rnd_key][easy_idxs])

            if not negatives: continue # Safety
            
            negatives = np.vstack(negatives)
            
            # Shuffle negatives to mix hard and easy
            np.random.shuffle(negatives)

            # Apply Transforms
            if positive_key in data_transform_funcs:
                anchors = data_transform_funcs[positive_key](anchors)
                positives = data_transform_funcs[positive_key](positives)
            
            transform_key = neg_speech_key if neg_speech_key in data_transform_funcs else pure_noise_key
            if transform_key and transform_key in data_transform_funcs:
                negatives = data_transform_funcs[transform_key](negatives)

            # Labels
            labels_anchor = torch.ones(n_pos, 1)
            labels_neg = torch.zeros(n_pos, 1)
            
            yield (torch.from_numpy(anchors).float(),
                   torch.from_numpy(positives).float(),
                   torch.from_numpy(negatives).float(),
                   labels_anchor.float(),
                   labels_neg.float())


# Function to remove empty rows from the end of a mmap array
def trim_mmap(mmap_path):
    """
    Trims blank rows from the end of a mmaped numpy array by creates new mmap array without the blank rows.
    Note that a copy is created and disk usage will briefly double as the function runs.

    Args:
        mmap_path (str): The path to mmap array file to trim

    Returns:
        None
    """
    # Identify the last full row in the mmaped file
    mmap_file1 = np.load(mmap_path, mmap_mode='r')
    i = -1
    while np.all(mmap_file1[i, :, :] == 0):
        i -= 1

    N_new = mmap_file1.shape[0] + i + 1

    # Create new mmap_file and copy over data in batches
    output_file2 = mmap_path.strip(".npy") + "2.npy"
    mmap_file2 = open_memmap(output_file2, mode='w+', dtype=np.float32,
                             shape=(N_new, mmap_file1.shape[1], mmap_file1.shape[2]))

    for i in tqdm(range(0, mmap_file1.shape[0], 1024), total=mmap_file1.shape[0]//1024, desc="Trimming empty rows"):
        if i + 1024 > N_new:
            mmap_file2[i:N_new] = mmap_file1[i:N_new].copy()
            mmap_file2.flush()
        else:
            mmap_file2[i:i+1024] = mmap_file1[i:i+1024].copy()
            mmap_file2.flush()

    # Remove old mmaped file
    os.remove(mmap_path)

    # Rename new mmap file to match original
    os.rename(output_file2, mmap_path)


import torch
from phonemize.preprocessing.text import Preprocessor, LanguageTokenizer
from phonemize.preprocessing.text import SequenceTokenizer

# Add SequenceTokenizer to safe globals
torch.serialization.add_safe_globals([Preprocessor, LanguageTokenizer, SequenceTokenizer])



# Generate words that sound similar ("adversarial") to the input phrase using phoneme overlap
def generate_adversarial_texts(input_text: str, N: int, include_partial_phrase: float = 0, include_input_words: float = 0):
    """
    Generate adversarial words and phrases based on phoneme overlap.
    Currently only works for english texts.
    Note that homophones are excluded, as this wouldn't actually be an adversarial example for the input text.

    Args:
        input_text (str): The target text for adversarial phrases
        N (int): The total number of adversarial texts to return. Uses sampling,
                 so not all possible combinations will be included and some duplicates
                 may be present.
        include_partial_phrase (float): The probability of returning a number of words less than the input
                                        text (but always between 1 and the number of input words)
        include_input_words (float): The probability of including individual input words in the adversarial
                                     texts when the input text consists of multiple words. For example,
                                     if the `input_text` was "ok google", then setting this value > 0.0
                                     will allow for adversarial texts like "ok noodle", versus the word "ok"
                                     never being present in the adversarial texts.

    Returns:
        list: A list of strings corresponding to words and phrases that are phonetically similar (but not identical)
              to the input text.
    """
    # Get phonemes for english vowels (CMUDICT labels)
    vowel_phones = ["AA", "AE", "AH", "AO", "AW", "AX", "AXR", "AY", "EH", "ER", "EY", "IH", "IX", "IY", "OW", "OY", "UH", "UW", "UX"]

    word_phones = []
    input_text_phones = [pronouncing.phones_for_word(i) for i in input_text.split()]

    # Download phonemizer model for OOV words, if needed
    if [] in input_text_phones:
        phonemizer_mdl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "resources", "phonemizer_model", "phonemize_m1.pt")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(phonemizer_mdl_path), exist_ok=True)

        # Download if the file does not exist
        if not os.path.exists(phonemizer_mdl_path):
            import requests
            file_url = "https://github.com/arcosoph/phonemize/releases/download/v0.2.0/phonemize_m1.pt"
            logging.warning(f"Downloading phonemizer model from {file_url}...")
            r = requests.get(file_url, stream=True)
            with open(phonemizer_mdl_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=2048):
                    if chunk:
                        f.write(chunk)


        # Create phonemizer object
        from phonemize.phonemizer import Phonemizer
        phonemizer = Phonemizer.from_checkpoint(phonemizer_mdl_path)

    for phones, word in zip(input_text_phones, input_text.split()):
        if phones != []:
            word_phones.extend(phones)
        elif phones == []:
            logging.warning(f"The word '{word}' was not found in the pronunciation dictionary! "
                            "Using the Phonemize library to predict the phonemes.")
            phones = phonemizer(word, lang='en_us')
            logging.warning(f"Phones for '{word}': {phones}")
            word_phones.append(re.sub(r"[\]|\[]", "", re.sub(r"\]\[", " ", phones)))
        elif isinstance(phones[0], list):
            logging.warning(f"There are multiple pronunciations for the word '{word}'.")
            word_phones.append(phones[0])

    # add all possible lexical stresses to vowels
    word_phones = [re.sub('|'.join(vowel_phones), lambda x: str(x.group(0)) + '[0|1|2]', re.sub(r'\d+', '', i)) for i in word_phones]

    adversarial_phrases = []
    for phones, word in zip(word_phones, input_text.split()):
        query_exps = []
        phones = phones.split()
        adversarial_words = []
        if len(phones) <= 2:
            query_exps.append(" ".join(phones))
        else:
            query_exps.extend(phoneme_replacement(phones, max_replace=max(0, len(phones)-2), replace_char="(.){1,3}"))

        for query in query_exps:
            matches = pronouncing.search(query)
            matches_phones = [pronouncing.phones_for_word(i)[0] for i in matches]
            allowed_matches = [i for i, j in zip(matches, matches_phones) if j != phones]
            adversarial_words.extend([i for i in allowed_matches if word.lower() != i])

        if adversarial_words != []:
            adversarial_phrases.append(adversarial_words)

    # Build combinations for final output
    adversarial_texts = []
    for i in range(N):
        txts = []
        for j, k in zip(adversarial_phrases, input_text.split()):
            if np.random.random() > (1 - include_input_words):
                txts.append(k)
            else:
                txts.append(np.random.choice(j))

        if include_partial_phrase is not None and len(input_text.split()) > 1 and np.random.random() <= include_partial_phrase:
            n_words = np.random.randint(1, len(input_text.split())+1)
            adversarial_texts.append(" ".join(np.random.choice(txts, size=n_words, replace=False)))
        else:
            adversarial_texts.append(" ".join(txts))

    # Remove any exact matches to input phrase
    adversarial_texts = [i for i in adversarial_texts if i != input_text]

    return adversarial_texts


def phoneme_replacement(input_chars, max_replace, replace_char='"(.){1,3}"'):
    results = []
    chars = list(input_chars)

    # iterate over the number of characters to replace (1 to max_replace)
    for r in range(1, max_replace+1):
        # get all combinations for a fixed r
        comb = itertools.combinations(range(len(chars)), r)
        for indices in comb:
            chars_copy = chars.copy()
            for i in indices:
                chars_copy[i] = replace_char
            results.append(' '.join(chars_copy))

    return results
