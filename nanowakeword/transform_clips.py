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

# (✿◕‿◕✿)
import os
import scipy
import torch
import random
import torchaudio
import numpy as np
from tqdm import tqdm
from pathlib import Path

from nanowakeword.data.augment_clips import augment_clips
from nanowakeword.data.AudioFeatures import AudioFeatures
from nanowakeword.utils.logger import print_step_header, print_info, print_warning
from nanowakeword.data.trim_mmap import trim_mmap


SEED=10
def set_seed(seed):
    """
    This function sets the seed to make the training results reliable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

# In transform_clips.py

# Make sure these imports are at the top of your transform_clips.py file
from multiprocessing import Pool, cpu_count
import torchaudio # This might be needed if not already imported

# We need the same helper function here as well. 
# For better code organization, you could move this helper to a shared utility file,
# but for now, we can define it again here or assume it's available.
# Let's assume you've already added it to augment_clips.py as a top-level function.
# For this file to work standalone, let's define it.

def _load_and_preprocess_clip(args):
    """Helper function to load and preprocess a single audio clip. Runs in a separate process."""
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
            waveform = waveform[:, -total_length:]
        elif current_len < total_length:
            padding_needed = total_length - current_len
            waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
        
        return waveform
    except Exception as e:
        # print_warning(f"Skipping corrupted file {clip_path}: {e}")
        return None

def _raw_audio_batch_generator(clip_paths, total_length, batch_size, sr=16000, num_workers=0):
    """
    A high-performance generator that loads and pre-processes raw audio in batches
    using parallel processing.
    """
    random.shuffle(clip_paths)
    
    # Determine the actual number of workers to use
    if num_workers == -1:
        worker_count = cpu_count()
    elif num_workers > 0:
        worker_count = min(num_workers, cpu_count())
    else: # num_workers is 0 or invalid
        worker_count = 0
    
    # Logic for parallel processing
    if worker_count > 0:
        with Pool(processes=worker_count) as pool:
            for i in range(0, len(clip_paths), batch_size):
                batch_paths = clip_paths[i:i+batch_size]
                args_for_pool = [(path, total_length, sr) for path in batch_paths]
                
                processed_waveforms = pool.map(_load_and_preprocess_clip, args_for_pool)
                batch_audio = [wf for wf in processed_waveforms if wf is not None]

                if not batch_audio:
                    continue
                
                # Stack, normalize, and yield
                batch_tensor = torch.stack(batch_audio)
                final_batch_np = batch_tensor.cpu().numpy()
                max_val = np.abs(final_batch_np).max(axis=-1, keepdims=True)
                max_val[max_val < 1e-8] = 1.0 
                output_batch = (final_batch_np / max_val).squeeze(1) 
                yield (output_batch * 32767).astype(np.int16)
    else:
        # Fallback to serial processing (safe mode for num_workers=0)
        for i in range(0, len(clip_paths), batch_size):
            batch_paths = clip_paths[i:i+batch_size]
            batch_audio = []
            for path in batch_paths:
                waveform = _load_and_preprocess_clip((path, total_length, sr))
                if waveform is not None:
                    batch_audio.append(waveform)
            
            if not batch_audio:
                continue
            
            # Same processing logic
            batch_tensor = torch.stack(batch_audio)
            final_batch_np = batch_tensor.cpu().numpy()
            max_val = np.abs(final_batch_np).max(axis=-1, keepdims=True)
            max_val[max_val < 1e-8] = 1.0 
            output_batch = (final_batch_np / max_val).squeeze(1) 
            yield (output_batch * 32767).astype(np.int16)



def _determine_clip_length(config):
    """
    Determines the optimal training clip length in samples.

    It follows a priority system:
    1. A user-defined fixed length (`audio_processing.clip_length_samples`).
    2. An autotuned length based on the median duration of positive clips.
    3. A fallback length if autotuning is disabled.

    Args:
        config: The configuration proxy object.

    Returns:
        int: The determined clip length in samples.
    """
    audio_cfg = config.get("audio_processing", {})

    # User-defined fixed clip length
    if (fixed_clip_length := audio_cfg.get("clip_length_samples")) is not None:
        print_info(f"Using user-defined clip duration: {fixed_clip_length} samples.")
        return fixed_clip_length

    # Autotune process
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

        duration_in_samples = []
        for clip_path in sampled_clips:
            try:
                sample_rate, data = scipy.io.wavfile.read(clip_path)
                if sample_rate != 16000:
                    print_warning(f"Clip '{os.path.basename(clip_path)}' has sample rate {sample_rate}Hz, not 16kHz. This may affect duration calculation.")
                duration_in_samples.append(len(data))
            except Exception as e:
                print_warning(f"Could not read and process clip '{os.path.basename(clip_path)}': {e}")

        if not duration_in_samples:
            print_warning("Could not determine median duration. Using minimum allowable length as fallback.")
            final_length = min_length
        else:
            median_duration_samples = np.median(duration_in_samples)
            buffer_samples = int((buffer_ms / 1000) * 16000)
            base_length = round(median_duration_samples / 1000) * 1000
            calculated_length = int(base_length + buffer_samples)

            # Apply constraints
            final_length = max(min_length, calculated_length)
            if abs(final_length - min_length) <= snap_tolerance:
                final_length = min_length
        
        print_info(f"Optimal clip duration autotuned to: {final_length} samples ({final_length/16000:.2f} seconds).")
        return final_length

    # Fallback when autotune is disabled and no fixed length is set
    fallback_length = autotune_cfg.get("min_allowable_length", 32000)
    print_info(f"Autotuning is disabled. Using fallback clip duration: {fallback_length} samples.")
    return fallback_length


def _process_generation_job(job_name, overwrite, recipe, config, feature_save_dir, rir_paths, background_paths, total_length):
    """
    Processes a single feature generation job defined in the manifest.

    This includes gathering input files, setting up augmentation, generating
    features, and saving them to a memory-mapped file.

    Args:
        job_name (str): The name of the generation job.
        recipe (ConfigProxy): The configuration for this specific job.
        config (ConfigProxy): The global configuration object.
        feature_save_dir (str): Directory to save the generated feature file.
        rir_paths (list): A list of paths to impulse response files.
        background_paths (list): A list of paths to background noise files.
        total_length (int): The target length of audio clips in samples.
    """
    print_info(f"Running Generation: {job_name}")

    output_filename = recipe.get("output_filename")
    if not output_filename:
        print_warning(f"Skipping job '{job_name}' because 'output_filename' is missing.")
        return
        
    # is_overwrite = config.get("overwrite", False) or args.overwrite
    output_filepath = os.path.join(feature_save_dir, output_filename)
    if os.path.exists(output_filepath) and not overwrite:
        print_warning(f"Feature file '{output_filename}' already exists. Skipping generation. (Use --overwrite to force regeneration)")
        return

    input_clips = [str(p) for d in recipe.get("input_audio_dirs", []) for p in Path(d).rglob("*.wav")]
    if not input_clips:
        print_warning(f"Skipping job '{job_name}' as no .wav files were found in the specified directories.")
        return
    print_info(f"Found {len(input_clips)} source audio files.")

    global_aug_proxy = config.get("augmentation_settings", {})
    recipe_aug_proxy = recipe.get("augmentation_settings", {})
    global_aug_dict = global_aug_proxy.to_dict() if global_aug_proxy else {}
    recipe_aug_dict = recipe_aug_proxy.to_dict() if recipe_aug_proxy else {}
    final_aug_settings = {**global_aug_dict, **recipe_aug_dict}

    aug_rounds = recipe.get("augmentation_rounds", 1)
    clips_to_generate = input_clips * aug_rounds
    total_clips_to_generate = len(clips_to_generate)
    batch_size = config.get('augmentation_batch_size', 128)
    print_info(f"Augmentation rounds: {aug_rounds}. Total clips to generate: {total_clips_to_generate}")

    use_augmentation = not (global_aug_proxy is False or recipe_aug_proxy is False)
    

    num_workers = config.get("feature_gen_num_workers")
    if num_workers is None:
        num_workers = config.get("num_workers", 3)

    if use_augmentation:
        bg_paths_for_job = background_paths if recipe.get("use_background_noise", True) else []
        rir_paths_for_job = rir_paths if recipe.get("use_rir", True) else []
        audio_generator = augment_clips(
            clip_paths=clips_to_generate,
            total_length=total_length,
            batch_size=batch_size,
            background_clip_paths=bg_paths_for_job,
            RIR_paths=rir_paths_for_job,
            num_workers=num_workers,
            augmentation_settings=final_aug_settings
        )
    else:
        print_info("Augmentation disabled for this job. Using raw audio.")
        audio_generator = _raw_audio_batch_generator(
            clip_paths=clips_to_generate,
            total_length=total_length,
            batch_size=batch_size,
            num_workers=num_workers
        )

    n_cpus = max(1, int(os.cpu_count() * config.get("feature_gen_cpu_ratio", 0.6)))
    feature_extractor = AudioFeatures(device="gpu" if torch.cuda.is_available() else "cpu")
    sample_embedding_shape = feature_extractor.get_embedding_shape(total_length / 16000)
    output_shape = (total_clips_to_generate, *sample_embedding_shape)

    fp = np.lib.format.open_memmap(output_filepath, mode='w+', dtype=np.float32, shape=output_shape)
    
    row_counter = 0
    pbar_total = -(total_clips_to_generate // -batch_size)
    pbar = tqdm(audio_generator, total=pbar_total, desc=f"Processing {job_name}")

    for audio_batch in pbar:
        if row_counter >= total_clips_to_generate: break
        
        features = feature_extractor.embed_clips(audio_batch, batch_size=len(audio_batch), ncpu=n_cpus)
        
        end_index = min(row_counter + features.shape[0], total_clips_to_generate)
        fp[row_counter:end_index, :, :] = features[:end_index - row_counter]
        row_counter = end_index
        fp.flush()
    
    del fp
    trim_mmap(output_filepath)
    
    print_info(f"Job '{job_name}' completed successfully!")


def transform_clips(config, args, feature_save_dir):
    """
    Orchestrates the feature generation process based on the configuration.
    
    This function prepares necessary resources, determines the audio clip length,
    and then iterates through a manifest to generate feature sets for different
    types of audio data (e.g., positives, negatives, background noise).
    
    Args:
        config: The main configuration proxy object.
        args: Command-line arguments.
        feature_save_dir (str): The directory where feature files will be saved.
    """
    if not (args.transform_clips or config.get("transform_clips", False)):
        print_info("Feature generation is disabled via config/flag. Skipping.")
        return

    # 1. Prepare shared resources
    rir_paths = [i.path for j in config["rir_paths"] for i in os.scandir(j)]
    background_paths = []
    bg_paths_config = config.get("background_paths", [])
    bg_rates_config = config.get("background_paths_duplication_rate", [])
    if len(bg_rates_config) != len(bg_paths_config):
        bg_rates_config = [1] * len(bg_paths_config)

    for path, rate in zip(bg_paths_config, bg_rates_config):
        background_paths.extend([i.path for i in os.scandir(path)] * rate)

    # 2. Determine and set the clip length for this run
    config["total_length"] = _determine_clip_length(config)
    
    # 3. Process the feature generation manifest
    generation_manifest = config.get("feature_generation_manifest")
    if not generation_manifest:
        print_warning("'feature_generation_manifest' not found. Skipping feature generation.")
        return

    is_overwrite = config.get("overwrite", False) or args.overwrite

    print_step_header("Computing Acoustic Features from Audio Sources")
    for job_name, recipe in generation_manifest.items():
        _process_generation_job(
            job_name=job_name,
            overwrite=is_overwrite,
            recipe=recipe,
            config=config,
            feature_save_dir=feature_save_dir,
            rir_paths=rir_paths,
            background_paths=background_paths,
            total_length=config["total_length"]
        )

    print_info("All feature generation jobs finished.")