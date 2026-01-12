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
import numpy as np
from tqdm import tqdm
from pathlib import Path

from nanowakeword.data.augment_clips import augment_clips
from nanowakeword.data.AudioFeatures import AudioFeatures
from nanowakeword.utils.logger import print_step_header, print_info


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


def transform_clips(config, args, feature_save_dir):
        
    # Get paths for impulse response and background audio files
    rir_paths = [i.path for j in config["rir_paths"] for i in os.scandir(j)]
    background_paths = []
    if len(config["background_paths_duplication_rate"]) != len(config["background_paths"]):
        config["background_paths_duplication_rate"] = [1]*len(config["background_paths"])
    for background_path, duplication_rate in zip(config["background_paths"], config["background_paths_duplication_rate"]):
        background_paths.extend([i.path for i in os.scandir(background_path)]*duplication_rate)

    # Determine the optimal training clip length 
    # Get the audio config section using the proxy. This returns another ConfigProxy.
    audio_cfg = config.get("audio_processing", {})

    # Priority 1: Check if the user has provided a fixed clip length to override everything.
    # The access to 'clip_length_samples' is now automatically tracked by the proxy.
    fixed_clip_length = audio_cfg.get("clip_length_samples", None)

    if fixed_clip_length is not None:
        # If a fixed length is specified, use it directly and skip the autotune process.
        config["total_length"] = fixed_clip_length
        print_info(f"Using user-defined clip duration: {fixed_clip_length} samples.")

    else:
        # Priority 2: Proceed with the autotune process.
        # Get the autotune section. If it doesn't exist, use an empty dict as default.
        # This also returns a ConfigProxy.
        autotune_cfg = audio_cfg.get("autotune_length", {})
        
        # Autotune is enabled by default. Each .get() call from here is automatically
        # tracked with the full nested path (e.g., "audio.autotune_length.enabled").
        if autotune_cfg.get("enabled", True):
            print_info("Autotuning optimal clip duration...")

            # Get autotune parameters. The proxy handles defaults gracefully.
            num_to_inspect = autotune_cfg.get("num_samples_to_inspect", 50)
            buffer_ms = autotune_cfg.get("duration_buffer_ms", 750)
            min_length = autotune_cfg.get("min_allowable_length", 32000)
            snap_tolerance = autotune_cfg.get("snap_to_min_tolerance", 4000)
            
            # Sample clips and calculate median duration 
            positive_clips_path = Path(config["positive_data_path"])
            positive_clips = [str(p) for p in positive_clips_path.glob("*.wav")]
            
            if not positive_clips:
                raise FileNotFoundError(f"No .wav files found for autotuning in: {positive_clips_path}")
            
            num_to_sample = min(num_to_inspect, len(positive_clips))
            sampled_clips = np.random.choice(positive_clips, num_to_sample, replace=False)

            duration_in_samples = []
            for clip_path in sampled_clips:
                try:
                    sample_rate, data = scipy.io.wavfile.read(clip_path)
                    if sample_rate != 16000:
                        print_info(f"[WARNING] Clip '{os.path.basename(clip_path)}' has sample rate {sample_rate}Hz, not 16kHz. This may affect duration calculation.")
                    duration_in_samples.append(len(data))
                except Exception as e:
                    print_info(f"[WARNING] Could not read and process clip '{os.path.basename(clip_path)}': {e}")
            
            # Calculate the final length based on the sampled durations
            if not duration_in_samples:
                print_info("[WARNING] Could not determine median duration. Using minimum allowable length as fallback.")
                final_length = min_length
            else:
                median_duration_samples = np.median(duration_in_samples)
                buffer_samples = int((buffer_ms / 1000) * 16000)
                
                base_length = round(median_duration_samples / 1000) * 1000
                calculated_length = int(base_length + buffer_samples)

                # Apply constraints
                if calculated_length < min_length:
                    final_length = min_length
                elif abs(calculated_length - min_length) <= snap_tolerance:
                    final_length = min_length
                else:
                    final_length = calculated_length
            
            config["total_length"] = final_length
            print_info(f"Optimal clip duration autotuned to: {final_length} samples ({final_length/16000:.2f} seconds).")
        
        else:
            # Priority 3: Autotune is explicitly disabled, and no fixed length was given.
            fallback_length = autotune_cfg.get("min_allowable_length", 32000)
            config["total_length"] = fallback_length
            print_info(f"Autotuning is disabled. Using fallback clip duration: {fallback_length} samples.")

    ISoverwrite = config.get("overwrite", False)
    transform_data = config.get("transform_clips", False)

    if args.transform_clips is True or transform_data:

        generation_manifest = config.get("feature_generation_manifest")

        if not generation_manifest:
            print_info("[INFO] 'feature_generation_manifest' not found in config.yaml. Skipping custom feature generation.")
        else:
            # print_step_header("Activating Flexible Feature Generation Engine")
            print_step_header("Computing Acoustic Features from Audio Sources")

            for job_name, recipe in generation_manifest.items():
                print_info(f"Running Generation: {job_name}")

                output_filename = recipe.get("output_filename")
                if not output_filename:
                    print_info(f"[WARNING] Skipping job '{job_name}' because 'output_filename' is missing.")
                    continue

                output_filepath = os.path.join(feature_save_dir, output_filename)

                if os.path.exists(output_filepath) and not (args.overwrite or ISoverwrite):
                    print_info(f"[INFO] Feature file '{output_filename}' already exists. Skipping generation. (Use --overwrite to force regeneration)")
                    continue

                input_audio_dirs = recipe.get("input_audio_dirs", [])
                if not input_audio_dirs:
                    print_info(f"[WARNING] Skipping job '{job_name}' because 'input_audio_dirs' is empty or missing.")
                    continue

                input_clips = []
                for d in input_audio_dirs:
                    input_clips.extend([str(p) for p in Path(d).rglob("*.wav")])
                
                if not input_clips:
                    print_info(f"[WARNING] Skipping job '{job_name}' as no .wav files were found in the specified directories.")
                    continue
                
                print_info(f"Found {len(input_clips)} source audio files.")

                
                global_aug_proxy = config.get("augmentation_settings", {})
                recipe_aug_proxy = recipe.get("augmentation_settings", {})

                # Converting ConfigProxy objects to plain dictionaries
                # The .to_dict attribute holds the actual dictionary inside the ConfigProxy.
                global_aug_dict = global_aug_proxy.to_dict() if global_aug_proxy else {}
                recipe_aug_dict = recipe_aug_proxy.to_dict() if recipe_aug_proxy else {}

                final_aug_settings = {**global_aug_dict, **recipe_aug_dict}          
                
                use_bg = recipe.get("use_background_noise", True)
                use_rir = recipe.get("use_rir", True)

                bg_paths_for_job = background_paths if use_bg else []
                rir_paths_for_job = rir_paths if use_rir else []

                aug_rounds = recipe.get("augmentation_rounds", 1)
                clips_to_generate = input_clips * aug_rounds
                total_clips_to_generate = len(clips_to_generate)
                
                print_info(f"Augmentation rounds: {aug_rounds}. Total clips to generate: {total_clips_to_generate}")

                audio_generator = augment_clips(
                    clip_paths=clips_to_generate,
                    total_length=config["total_length"],
                    batch_size=config["augmentation_batch_size"],
                    background_clip_paths=bg_paths_for_job,
                    RIR_paths=rir_paths_for_job,
                    augmentation_settings=final_aug_settings
                )
                
                # print(f"Computing features'{output_filename}'...")
                n_cpus = os.cpu_count()
                cpu_usage_ratio = config.get("feature_gen_cpu_ratio", 0.6)
                n_cpus = max(1, int(n_cpus * cpu_usage_ratio))
                    
                feature_extractor = AudioFeatures(device="gpu" if torch.cuda.is_available() else "cpu")
                sample_embedding_shape = feature_extractor.get_embedding_shape(config["total_length"] / 16000)
                output_shape = (total_clips_to_generate, sample_embedding_shape[0], sample_embedding_shape[1])
                
                fp = np.lib.format.open_memmap(output_filepath, mode='w+', dtype=np.float32, shape=output_shape)
                
                row_counter = 0
                batch_size = config.get('augmentation_batch_size', 128)
                pbar = tqdm(audio_generator, total=-(total_clips_to_generate // -batch_size), desc=f"{job_name}")

                for audio_batch in pbar:
                    if row_counter >= total_clips_to_generate: break
                    features = feature_extractor.embed_clips(audio_batch, batch_size=len(audio_batch), ncpu=n_cpus)
                    end_index = min(row_counter + features.shape[0], total_clips_to_generate)
                    fp[row_counter:end_index, :, :] = features[:end_index - row_counter]
                    row_counter = end_index
                    fp.flush()
                
                del fp
                from nanowakeword.data.trim_mmap import trim_mmap
                trim_mmap(output_filepath)
                
                print_info(f"{job_name} Completed Successfully!")
            
            print_info("Flexible Feature Generation Finished")

    else:
        print_info("Feature generation is disabled as 'transform_clips' is false and '--transform_clips' flag is not set.")
