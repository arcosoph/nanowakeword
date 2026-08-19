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

import os
import time
import scipy
import torch
import random
import torchaudio
import numpy as np
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

from nanowakeword.data.trim_mmap import trim_mmap
from nanowakeword.data.augment_clips import augment_clips
from nanowakeword.data.AudioFeatures import AudioFeatures
from nanowakeword.utils.logger import print_step_header, print_info, print_warning

SEED = 10

def set_seed(seed):
    """Set the random seed for reproducibility across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

def _worker_init():
    """Initialize random seeds for multiprocessing workers.

    Each worker gets a unique seed derived from its process ID and the current
    time so that augmentations differ across workers while remaining
    reproducible within the same worker.
    """
    seed = (os.getpid() * int(time.time() * 1000)) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_and_preprocess_clip(args):
    """Load a single audio clip, resample to target rate, convert to mono,
    and crop or pad to the specified length.

    Args:
        args: Tuple of (clip_path, total_length, sample_rate).

    Returns:
        Tensor of shape [1, total_length] or None if loading failed.
    """
    clip_path, total_length, sr = args
    try:
        waveform, clip_sr = torchaudio.load(clip_path)
        if clip_sr != sr:
            resampler = torchaudio.transforms.Resample(orig_freq=clip_sr, new_freq=sr)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        current_len = waveform.shape[1]
        if current_len > total_length:
            start = random.randint(0, current_len - total_length)
            waveform = waveform[:, start:start + total_length]
        elif current_len < total_length:
            padding_needed = total_length - current_len
            waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
        return waveform
    except Exception as e:
        print_warning(f"Error loading {clip_path}: {e}")
        return None


def _raw_audio_batch_generator(clip_paths, total_length, batch_size, sr=16000, num_workers=0):
    """Generator that yields batches of raw audio clips for E2E mode processing.

    Loads clips without augmentation (no background noise or RIR), applies
    random volume scaling (0.5x to 1.0x), and yields int16 audio batches.
    Supports both multiprocessing (via Pool) and single-process execution.

    Args:
        clip_paths: List of audio file paths to process.
        total_length: Target number of samples per clip (crop/pad to this length).
        batch_size: Number of clips per batch.
        sr: Target sample rate in Hz.
        num_workers: Number of worker processes (0 = single process, -1 = all cores).

    Yields:
        np.ndarray of shape [batch_size, total_length] with int16 dtype.
    """
    random.shuffle(clip_paths)
    worker_count = cpu_count() if num_workers == -1 else (min(num_workers, cpu_count()) if num_workers > 0 else 0)

    if worker_count > 0:
        with Pool(processes=worker_count, initializer=_worker_init) as pool:
            for i in range(0, len(clip_paths), batch_size):
                batch_paths = clip_paths[i:i + batch_size]
                args_for_pool = [(path, total_length, sr) for path in batch_paths]
                processed = pool.map(_load_and_preprocess_clip, args_for_pool)
                batch_audio = [wf for wf in processed if wf is not None]
                if not batch_audio: continue
                batch_tensor = torch.stack(batch_audio)
                # Random volume scaling: multiply each clip by a gain factor in [0.5, 1.0]
                volumes = torch.FloatTensor(batch_tensor.shape[0], 1, 1).uniform_(0.5, 1.0)
                batch_tensor = torch.clamp(batch_tensor * volumes, -1.0, 1.0)
                output_batch = (batch_tensor.cpu().numpy() * 32767).astype(np.int16).squeeze(1)
                yield output_batch
    else:
        for i in range(0, len(clip_paths), batch_size):
            batch_paths = clip_paths[i:i + batch_size]
            batch_audio = [wf for path in batch_paths if (wf := _load_and_preprocess_clip((path, total_length, sr))) is not None]
            if not batch_audio: continue
            batch_tensor = torch.stack(batch_audio)
            volumes = torch.FloatTensor(batch_tensor.shape[0], 1, 1).uniform_(0.5, 1.0)
            batch_tensor = torch.clamp(batch_tensor * volumes, -1.0, 1.0)
            output_batch = (batch_tensor.cpu().numpy() * 32767).astype(np.int16).squeeze(1)
            yield output_batch


def _determine_clip_length(config, target_sr):
    """Determine the fixed clip length (in samples) for audio processing.

    Priority:
    1. If `audio_processing.clip_length_samples` is set in config, use it directly.
    2. Otherwise, if autotuning is enabled, inspect positive clips to find the
       median duration, add a buffer, and snap to the nearest 1000-sample boundary.
    3. Fall back to `min_allowable_length`.

    Args:
        config: ConfigProxy or dict containing audio_processing settings.
        target_sr: Target sample rate in Hz.

    Returns:
        Integer number of samples for each clip.
    """
    audio_cfg = config.get("audio_processing", {})
    if (fixed_clip_length := audio_cfg.get("clip_length_samples")) is not None:
        print_info(f"Using user-defined clip duration: {fixed_clip_length} samples.")
        return fixed_clip_length

    autotune_cfg = audio_cfg.get("autotune_length", {})
    if autotune_cfg.get("enabled", True):
        print_info("Autotuning optimal clip duration...")
        num_to_inspect = autotune_cfg.get("num_samples_to_inspect", 50)
        buffer_ms = autotune_cfg.get("duration_buffer_ms", 750)
        min_length = autotune_cfg.get("min_allowable_length", 32000)
        snap_tolerance = autotune_cfg.get("snap_to_min_tolerance", 4000)

        positive_clips = [str(p) for p in Path(config["positive_data_path"]).glob("*.wav")]
        if not positive_clips:
            raise FileNotFoundError(f"No .wav files found for autotuning in: {config['positive_data_path']}")

        num_to_sample = min(num_to_inspect, len(positive_clips))
        sampled_clips = np.random.choice(positive_clips, num_to_sample, replace=False)

        # Measure raw duration of each sampled clip
        duration_in_samples = []
        for clip_path in sampled_clips:
            try:
                sample_rate, data = scipy.io.wavfile.read(clip_path)
                if sample_rate != target_sr:
                    print_warning(f"Clip '{os.path.basename(clip_path)}' has sample rate {sample_rate}Hz, expected {target_sr}Hz.")
                duration_in_samples.append(len(data))
            except Exception as e:
                print_warning(f"Could not read {os.path.basename(clip_path)} for autotuning: {e}")

        if not duration_in_samples:
            final_length = min_length
        else:
            # Use the median duration, round up to the nearest 1000 samples,
            # and add a buffer in milliseconds
            median_duration_samples = np.median(duration_in_samples)
            buffer_samples = int((buffer_ms / 1000) * target_sr)
            base_length = round(median_duration_samples / 1000) * 1000
            final_length = max(min_length, int(base_length + buffer_samples))
            # Snap to min_length if within tolerance
            if abs(final_length - min_length) <= snap_tolerance:
                final_length = min_length

        print_info(f"Clip duration autotuned to: {final_length} samples ({final_length / target_sr:.2f}s).")
        return final_length

    fallback_length = autotune_cfg.get("min_allowable_length", 32000)
    print_info(f"Autotuning disabled. Using fallback: {fallback_length} samples.")
    return fallback_length


def _collect_resource_paths(config):
    """Collect Room Impulse Response (RIR) and background noise file paths
    from the configured directories.

    RIR paths are gathered from each directory in `rir_paths`. Background
    paths are gathered from `background_paths`, with each directory's files
    duplicated according to its entry in `background_paths_duplication_rate`.

    Args:
        config: ConfigProxy or dict containing RIR and background path settings.

    Returns:
        Tuple of (rir_paths, background_paths), each a list of file path strings.
    """
    rir_paths = []
    for j in config.get("rir_paths", []):
        if not os.path.isdir(j): continue
        try:
            rir_paths.extend([entry.path for entry in os.scandir(j)])
        except OSError as e:
            print_warning(f"Error accessing RIR path {j}: {e}")

    background_paths = []
    bg_paths_config = config.get("background_paths", [])
    bg_rates_config = config.get("background_paths_duplication_rate", [])
    # Default duplication rate to 1 if not explicitly provided for each path
    bg_rates_config = bg_rates_config if len(bg_rates_config) == len(bg_paths_config) else [1] * len(bg_paths_config)

    for path, rate in zip(bg_paths_config, bg_rates_config):
        try:
            for entry in os.scandir(path):
                background_paths.extend([entry.path] * rate)
        except OSError as e:
            print_warning(f"Error accessing Background path {path}: {e}")

    return rir_paths, background_paths


def _prepare_audio_generator(job_name, recipe, config, rir_paths, background_paths, total_length, target_sr, is_e2e=False):
    """Set up the audio generation pipeline for a single recipe job.

    Collects input clips from the recipe's input directories, determines how
    many output clips to generate, merges augmentation settings (global and
    recipe-level), and constructs the appropriate audio generator.

    When augmentation is enabled, clips are processed through the full
    augmentation pipeline (background noise, RIR convolution, gain, pitch, etc.).
    When disabled, clips are loaded and volume-scaled without augmentation.

    Args:
        job_name: Identifier for the generation job (used in log messages).
        recipe: Dict describing input directories, augmentation rounds,
            output directory/filename, and augmentation settings.
        config: Full training config (used for augmentation settings and batch params).
        rir_paths: List of RIR file paths for convolution augmentation.
        background_paths: List of background noise file paths.
        total_length: Target clip length in samples.
        target_sr: Target sample rate in Hz.
        is_e2e: If True, generate a fixed number of augmented audio clips.
            If False, process each input clip once to produce embeddings.

    Returns:
        Tuple of (audio_generator, total_clips) where audio_generator yields
        batches of audio data. Returns (None, 0) if the job was skipped.
    """
    input_clips = [str(p) for d in recipe.get("input_audio_dirs", []) for p in Path(d).rglob("*.wav")]
    if not input_clips:
        print_warning(f"Skipping job '{job_name}': no .wav files found.")
        return None, 0

    aug_rounds = recipe.get("augmentation_rounds", 1)
    clips_to_generate = input_clips * aug_rounds

    if is_e2e:
        num_samples = recipe.get("num_samples", 0)
        if num_samples <= 0:
            print_warning(f"Skipping job '{job_name}': 'num_samples' is 0 or missing.")
            return None, 0
        total_clips = num_samples * aug_rounds
    else:
        total_clips = len(clips_to_generate)

    # Merge global and recipe-level augmentation settings
    # Recipe settings override global settings
    global_aug_proxy = config.get("augmentation_settings", {})
    recipe_aug_proxy = recipe.get("augmentation_settings", {})
    g_aug = global_aug_proxy.to_dict() if hasattr(global_aug_proxy, 'to_dict') else (global_aug_proxy or {})
    r_aug = recipe_aug_proxy.to_dict() if hasattr(recipe_aug_proxy, 'to_dict') else (recipe_aug_proxy or {})
    final_aug_settings = {**g_aug, **r_aug}

    use_augmentation = not (global_aug_proxy is False or recipe_aug_proxy is False)
    batch_size = config.get("augmentation_batch_size", 128)
    num_workers = config.get("feature_gen_num_workers", config.get("num_workers", 3))

    if use_augmentation:
        bg_paths = background_paths if recipe.get("use_background_noise", True) else []
        rir_paths_job = rir_paths if recipe.get("use_rir", False) else []
        audio_generator = augment_clips(
            clip_paths=clips_to_generate,
            total_length=total_length,
            batch_size=batch_size,
            background_clip_paths=bg_paths,
            RIR_paths=rir_paths_job,
            num_workers=num_workers,
            augmentation_settings=final_aug_settings,
        )
    else:
        audio_generator = _raw_audio_batch_generator(
            clip_paths=clips_to_generate,
            total_length=total_length,
            batch_size=batch_size,
            sr=target_sr,
            num_workers=num_workers,
        )

    return audio_generator, total_clips


def _save_wav_file(path, rate, data):
    """Write a single audio clip to disk as a WAV file."""
    scipy.io.wavfile.write(path, rate, data)


def _process_e2e_generation_job(job_name, recipe, config, rir_paths, background_paths, total_length, target_sr):
    """Execute a single E2E audio generation job.

    Runs the audio generator, writes augmented clips to the output directory
    as individual WAV files using a thread pool for concurrent disk I/O.

    Args:
        job_name: Identifier for the job.
        recipe: Dict with output_dir, file_prefix, num_samples, etc.
        config: Full training config.
        rir_paths: RIR file paths for augmentation.
        background_paths: Background noise file paths for augmentation.
        total_length: Target clip length in samples.
        target_sr: Target sample rate in Hz.
    """
    print_info(f"Running E2E Audio Generation: {job_name}")

    output_dir = recipe.get("output_dir")
    if not output_dir: return print_warning(f"Skipping job '{job_name}': 'output_dir' is missing.")

    audio_generator, total_clips = _prepare_audio_generator(
        job_name,
        recipe,
        config,
        rir_paths,
        background_paths,
        total_length,
        target_sr, is_e2e=True
    )

    if not audio_generator: return

    os.makedirs(output_dir, exist_ok=True)
    file_prefix = recipe.get("file_prefix", "clip")
    batch_size = config.get("augmentation_batch_size", 128)

    generated = 0
    pbar = tqdm(audio_generator, total=-(-total_clips // batch_size), desc=f"Augmenting {job_name}")

    # Use a thread pool for concurrent disk I/O to improve write throughput
    max_threads = min(32, (os.cpu_count() or 4) * 2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        for audio_batch in pbar:
            futures = []
            for i in range(audio_batch.shape[0]):
                if generated >= total_clips: break
                out_path = os.path.join(output_dir, f"{file_prefix}_{generated:06d}.wav")
                futures.append(executor.submit(_save_wav_file, out_path, target_sr, audio_batch[i]))
                generated += 1

            concurrent.futures.wait(futures)
            if generated >= total_clips: break

    print_info(f"Job '{job_name}' completed: {generated} clips saved to {output_dir}")


def _process_embedding_generation_job(
        job_name,
        overwrite,
        recipe,
        config,
        feature_save_dir,
        rir_paths,
        background_paths,
        total_length,
        target_sr
        ):
    """Execute a single embedding feature generation job (embedding mode).

    Runs the audio generator to produce augmented audio batches, computes
    acoustic embeddings for each batch, and writes them to a memory-mapped
    .npy file for efficient disk usage during training.

    Args:
        job_name: Identifier for the job.
        overwrite: If False and the output file already exists, skip this job.
        recipe: Dict with output_filename and other generation parameters.
        config: Full training config.
        feature_save_dir: Directory where the .npy file will be written.
        rir_paths: RIR file paths for augmentation.
        background_paths: Background noise file paths for augmentation.
        total_length: Target clip length in samples.
        target_sr: Target sample rate in Hz.
    """
    print_info(f"Running Feature Generation: {job_name}")

    output_filename = recipe.get("output_filename")
    if not output_filename: return print_warning(f"Skipping job '{job_name}': 'output_filename' is missing.")

    output_filepath = os.path.join(feature_save_dir, output_filename)
    if os.path.exists(output_filepath) and not overwrite:
        return print_warning(f"Feature file '{output_filename}' already exists. Skipping.")

    audio_generator, total_clips = _prepare_audio_generator(
        job_name,
        recipe,
        config,
        rir_paths,
        background_paths,
        total_length,
        target_sr,
        is_e2e=False
    )

    if not audio_generator: return

    batch_size = config.get("augmentation_batch_size", 128)
    # Allocate CPU cores for feature extraction (60% of available cores by default)
    n_cpus = max(1, int((os.cpu_count() or 2) * config.get("feature_gen_cpu_ratio", 0.6)))
    feature_extractor = AudioFeatures(device="gpu" if torch.cuda.is_available() else "cpu")

    # Determine output shape using a sample embedding
    sample_embedding_shape = feature_extractor.get_embedding_shape(total_length / target_sr)
    output_shape = (total_clips, *sample_embedding_shape)

    # Use a memory-mapped array to write features incrementally to disk
    fp = np.lib.format.open_memmap(output_filepath, mode='w+', dtype=np.float32, shape=output_shape)
    row_counter = 0
    pbar = tqdm(audio_generator, total=-(-total_clips // batch_size), desc=f"Processing {job_name}")

    for audio_batch in pbar:
        if row_counter >= total_clips: break
        features = feature_extractor.embed_clips(audio_batch, batch_size=len(audio_batch), ncpu=n_cpus)
        end_index = min(row_counter + features.shape[0], total_clips)
        fp[row_counter:end_index, :, :] = features[:end_index - row_counter]
        row_counter = end_index
        fp.flush()

    del fp
    # Trim the memmap to the actual number of rows written
    trim_mmap(output_filepath)
    print_info(f"Job '{job_name}' completed!")


def transform_clips(config, args, feature_save_dir):
    """Run the full audio transformation / feature extraction pipeline.

    Depending on the config mode, this either:
    - E2E mode: generates augmented audio clip WAV files, or
    - Embedding mode: computes acoustic embedding features into .npy files.

    Iterates over all jobs defined in the generation manifest and dispatches
    each to the appropriate processing function.

    Args:
        config: ConfigProxy containing training configuration.
        args: Parsed command-line arguments (for --transform_clips and --overwrite flags).
        feature_save_dir: Directory path where feature files are saved (embedding mode only).
    """
    if not (args.transform_clips or config.get("transform_clips", False)):
        return print_info("Transform/generation is disabled. Skipping.")

    mode = config.get("mode", "embedding")
    is_e2e = (mode == "e2e")

    generation_manifest = config.get("data_generation_manifest") or config.get("feature_generation_manifest")
    if not generation_manifest:
        msg = "'data_generation_manifest'" if is_e2e else "'feature_generation_manifest' (or data_generation_manifest)"
        return print_warning(f"{msg} not found. Skipping.")

    target_sr = config.get("sample_rate", 16000)
    rir_paths, background_paths = _collect_resource_paths(config)
    total_length = _determine_clip_length(config, target_sr)

    # Store the determined clip length in config for downstream use (training, export)
    config["total_length"] = total_length
    is_overwrite = config.get("overwrite", False) or args.overwrite

    if is_e2e:
        print_step_header("E2E: Augmenting and Saving Audio Files")
    else:
        print_step_header("Computing Acoustic Features from Audio Sources")

    for job_name, recipe in generation_manifest.items():
        if is_e2e:
            _process_e2e_generation_job(
                job_name,
                recipe,
                config,
                rir_paths,
                background_paths,
                total_length,
                target_sr
            )
        else:
            _process_embedding_generation_job(
                job_name,
                is_overwrite,
                recipe,
                config,
                feature_save_dir,
                rir_paths,
                background_paths,
                total_length,
                target_sr
            )

    print_info("All generation jobs finished.")
