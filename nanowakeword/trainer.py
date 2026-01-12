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
import sys
import time
import yaml
import json
import torch
import random
import hashlib
import logging
import warnings
import argparse
import matplotlib
import numpy as np
import collections
import collections.abc

from nanowakeword._config.config_generator import ConfigGenerator
from nanowakeword._config.ConfigProxy import ConfigProxy

from torch.utils.data import DataLoader

from nanowakeword.data.data_sampler import DynamicClassAwareSampler, HardnessCurriculumDataset

from nanowakeword.utils.audio_preprocess import verify_and_process_directory

from nanowakeword._export.auto_gen_name import auto_gen_name as atoGeNm
from nanowakeword.utils.audio_analyzer import DatasetAnalyzer
from nanowakeword.utils.DynamicTable import DynamicTable
from nanowakeword.utils.journal import update_training_journal
from nanowakeword.utils.logger import print_banner, print_step_header, print_info, print_table
from nanowakeword.modules.model import Model

matplotlib.use('Agg')

# To make the terminal look clean
warnings.filterwarnings("ignore")
logging.getLogger("torchaudio").setLevel(logging.ERROR)

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

def deep_merge(d1, d2):
    """
    Recursively merges d2 into d1. If a key exists in both and the values are
    dictionaries, it merges them recursively. Otherwise, the value from d2
    overwrites the value from d1.
    """
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, collections.abc.Mapping):
            d1[k] = deep_merge(d1[k], v)
        else:
            d1[k] = v
    return d1


def collate_fn_with_indices(batch):
    """
    Custom collate function that handles features, labels, and indices.
    """
    features = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    indices = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return features, labels, indices


def train(cli_args=None):
    
    parser = argparse.ArgumentParser(
        description="NanoWakeWord: The Intelligent Wake Word Training Framework.",
        formatter_class=argparse.RawTextHelpFormatter # For better help message formatting
    )

    # Configuration 
    parser.add_argument(
            "-c", "--config_path",
            help="Path to the training configuration YAML file. (Required)",
            type=str,
            required=True,
            metavar="PATH"
        )

    # Pipeline Stages (Primary Actions)
    parser.add_argument(
        "-G", "--generate_clips",
        help="Activates the 'Generation' stage to synthesize audio clips.",
        action="store_true"
    )
    parser.add_argument(
        "-t", "--transform_clips",
        help="Activates the preparatory 'transform' stage (augmentation and feature extraction).",
        action="store_true"
    )
    parser.add_argument(
        "-T", "--train_model",
        help="Activates the final 'Training' stage to build the model.",
        action="store_true"
    )

    # Modifiers 
    parser.add_argument(
        "-f", "--force-verify",
        help="Forces re-verification of all data directories, ignoring the cache.",
        action="store_true"
    )
    parser.add_argument(
        "--overwrite", # NO SHORTHAND BY DESIGN FOR SAFETY
        help="Forces regeneration of feature files, overwriting any existing ones. Use with caution.",
        action="store_true"
    )

    parser.add_argument(
        "--resume",
        help="Path to the project directory to resume training from. (e.g., --resume ./trained_models/my_wakeword_v1)",
        type=str,
        default=None,
        metavar="PATH"
    )
    
    args = parser.parse_args(cli_args)


#=====
    print_banner()

    user_config = yaml.load(open(args.config_path, 'r', encoding='utf-8').read(), yaml.Loader)
# #=====

    # Define a stable cache directory based on the user's output_dir
    #    This ensures the path is known and available from the very beginning.
    output_dir_from_config = user_config.get("output_dir", "./trained_models")
    VERIFICATION_CACHE_DIR = os.path.join(output_dir_from_config, ".cache", "verification_receipts")
    os.makedirs(VERIFICATION_CACHE_DIR, exist_ok=True)

    # Define these file names here to avoid magic strings
    VERIFICATION_RECEIPT_FILENAME_TEMPLATE = "{hash}.json" # We'll use a hash now

    def get_directory_state(path):
        """Returns the current state of a directory (number of files and total size)."""
        file_count = 0
        total_size = 0
        # More robustly check for various audio extensions
        audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
        
        try:
            for entry in os.scandir(path):
                if entry.is_file() and os.path.splitext(entry.name)[1].lower() in audio_extensions:
                    file_count += 1
                    total_size += entry.stat().st_size
        except FileNotFoundError:
            # If the directory doesn't exist, its state is empty
            return {"file_count": 0, "total_size": 0}
                
        return {"file_count": file_count, "total_size": total_size}

    def smart_verify(path, force=False):
        """
        Smartly verifies a directory using a centralized cache in the project's output directory.
        This version is robust and will work correctly.
        """
        if not path: # Handle cases where a path might be None or empty in the config
            return

        # 1. Create a stable and unique hash for the directory path
        path_hash = hashlib.md5(path.encode('utf-8')).hexdigest()
        receipt_filename = VERIFICATION_RECEIPT_FILENAME_TEMPLATE.format(hash=path_hash)
        receipt_path = os.path.join(VERIFICATION_CACHE_DIR, receipt_filename)

        # 2. Check the verification receipt
        if not force and os.path.exists(receipt_path):
            try:
                with open(receipt_path, 'r') as f:
                    saved_state = json.load(f)
                
                current_state = get_directory_state(path)
                
                # If the state is identical, we can safely skip
                if saved_state == current_state:
                    print_info(f"'{os.path.basename(path)}' already verified. Skipping.")
                    return 
                else:
                    print_info(f"Data in '{os.path.basename(path)}' has changed. Re-verifying...")

            except (json.JSONDecodeError, KeyError) as e:
                print_info(f"Could not read or parse receipt for '{os.path.basename(path)}'. Re-verifying... Error: {e}")
        
        # 3. Perform the actual verification and processing
        # This part runs if verification is forced, receipt doesn't exist, or state has changed.
        try:
            verify_and_process_directory(path)
        
            # 4. Write the new state to the centralized cache
            current_state = get_directory_state(path)
            with open(receipt_path, 'w') as f:
                json.dump(current_state, f, indent=4)

        except FileNotFoundError:
            logging.warning(f"Directory not found, skipping preprocessing: {path}")
        except Exception as e:
            # Catch other potential errors during verification or writing the receipt
            print_info(f"Warning: An unexpected error occurred for '{os.path.basename(path)}'. Error: {e}")


    print_step_header("Verifying and Preprocessing Data Directories")
    
    data_paths_to_process = [
        user_config.get("positive_data_path"),
        user_config.get("negative_data_path")
    ]

    data_paths_to_process.extend(user_config.get("background_paths", []))
    data_paths_to_process.extend(user_config.get("rir_paths", []))


    unique_paths = set(p for p in data_paths_to_process if p)
    
    ISforce_verify= user_config.get("force_verify", False)
    if args.force_verify or ISforce_verify:
        print_info("User has forced re-verification of all data directories.")
    
    for path in unique_paths:
        smart_verify(path, force=args.force_verify or ISforce_verify)
        
    print_info("Data verification and preprocessing complete.\n")
   
    # Hardware-Only Configuration (Express Pass) 
    print_info("Determining hardware-specific configurations...")
    try:
        generator1 = ConfigGenerator() 
        intelligent_config1 = generator1.generate()

        final_config1 = intelligent_config1.copy()
        final_config1.update(user_config)

        base_config = final_config1

    except Exception as e:
        print_info(f"Could not generate intelligent hardware config due to an error: {e}. Proceeding with user config only.")
        base_config = user_config.copy() 


    ISgenaret_data = base_config.get("generate_clips", False)
    if args.generate_clips or ISgenaret_data:
        from nanowakeword.generate_clips import generate_clips
        # gen_sample = generate_clips(base_config)
        generate_clips(base_config)


    print_step_header("Activating Intelligent Configuration Engine")
    try:
        analyzer = DatasetAnalyzer(
            positive_path=user_config["positive_data_path"],
            negative_path=user_config["negative_data_path"],
            noise_path=user_config.get("background_paths", []), 
            rir_path=user_config["rir_paths"][0]
        ) 
        dataset_stats = analyzer.analyze()
        print_table(dataset_stats, "Dataset Statistics")

        generator = ConfigGenerator(dataset_stats)
        intelligent_config = generator.generate()

    except KeyError as e:
        print_info(f"ERROR: Missing essential path in config file for auto-config: {e}")
        exit()
    except Exception as e:
        print_info(f"Could not generate intelligent config due to an error: {e}. Proceeding with user config only.")
        intelligent_config = {} 

    final_config = intelligent_config.copy()
    
    # Deep merge the user's configuration on top of it.
    # This will correctly merge nested dictionaries like 'augmentation_settings'.
    final_config = deep_merge(final_config, user_config)
    
    # Now, `final_config` contains the correct, merged values.
    config_proxy = ConfigProxy(final_config)
    config = config_proxy

    show_table_flag = config.get("show_training_summary", True)
    dynamic_table = DynamicTable(config_proxy, title="Effective Training Configuration", enabled=show_table_flag)

    # Define and Create Professional Output Directory Structure  
    # project_dir = os.path.join(os.path.abspath(base_config["output_dir"]), base_config.get("model_name", AGMINTVP))
    project_dir = os.path.join(
        os.path.abspath(base_config["output_dir"]),
        base_config.get(
            "model_name",
            atoGeNm(model_type=config.get("model_type", "dnn"))
        )
    )

    feature_save_dir = os.path.join(project_dir, "features")
    artifacts_dir = os.path.join(project_dir, "training_artifacts")
    model_save_dir = os.path.join(project_dir, "model")

    for path in [project_dir, feature_save_dir, artifacts_dir, model_save_dir]:
        os.makedirs(path, exist_ok=True)

    print_info(f"Project assets will be saved in: {project_dir}")


    # ISoverwrite = config.get("overwrite", False)
    transform_data = config.get("transform_clips", False)

    if args.transform_clips is True or transform_data:
        from nanowakeword.transform_clips import transform_clips
        transform_clips(
                    config=config,
                    args=args,
                    feature_save_dir=feature_save_dir
        )


    should_train = config.get("train_model", False)
    if args.train_model is True or should_train:
        training_start_time = time.time()

        # Get the feature manifest from the config
        manifest = config.get("feature_manifest", {})

        # Pass the full manifest dictionary to the dataset
        dataset = HardnessCurriculumDataset(
            feature_manifests=manifest
        )

        if len(dataset) == 0:
            raise ValueError("CRITICAL: Dataset is empty. Check your feature file paths in the manifest.")

        # Get the batch composition from the config
        composition_cfg = config.get("batch_composition")

        if not composition_cfg:
            print_info("'batch_composition' not found in config. Generating a default balanced composition.")
            
            # Get the total batch size
            total_batch_size = config.get('batch_size', 128)
            
            # Determine which categories are present in the manifest
            has_targets = bool(manifest.get("targets"))
            has_negatives = bool(manifest.get("negatives"))
            has_backgrounds = bool(manifest.get("backgrounds"))
            
            num_categories_present = sum([has_targets, has_negatives, has_backgrounds])
            
            if num_categories_present == 0:
                raise ValueError("CRITICAL: feature_manifest is empty. Cannot create a default batch composition.")

            # A good default ratio: 25% targets, 50% negatives, 25% backgrounds
            # If a category is missing, its share is distributed among the others.
            
            composition_cfg = {}
            
            # Calculate default quotas
            if num_categories_present == 1:
                # If only one category exists, it gets the full batch size
                if has_targets: composition_cfg['targets'] = total_batch_size
                if has_negatives: composition_cfg['negatives'] = total_batch_size
                if has_backgrounds: composition_cfg['backgrounds'] = total_batch_size
            elif num_categories_present == 2:
                # Distribute among two, e.g., 50/50 or 33/67
                if has_targets and has_negatives:
                    composition_cfg['targets'] = total_batch_size // 3
                    composition_cfg['negatives'] = total_batch_size - composition_cfg['targets']
                elif has_targets and has_backgrounds:
                    composition_cfg['targets'] = total_batch_size // 2
                    composition_cfg['backgrounds'] = total_batch_size - composition_cfg['targets']
                elif has_negatives and has_backgrounds:
                    composition_cfg['negatives'] = total_batch_size // 2
                    composition_cfg['backgrounds'] = total_batch_size - composition_cfg['negatives']
            else: # All three are present
                composition_cfg['targets'] = total_batch_size // 4       # 25%
                composition_cfg['backgrounds'] = total_batch_size // 4  # 25%
                composition_cfg['negatives'] = total_batch_size - (composition_cfg['targets'] + composition_cfg['backgrounds']) # Remaining 50%

            print_info(f"Using default composition: {composition_cfg}")


        sampler = DynamicClassAwareSampler(
            dataset=dataset,
            batch_composition=composition_cfg,
            feature_manifests=manifest
        )

        # Create DataLoader with our custom sampler and collate_fn
        num_workers = config.get("num_workers", 2)
        X_train = DataLoader(
            dataset,
            batch_sampler=sampler, # Use batch_sampler for samplers that yield full batches
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False,
            persistent_workers=True if num_workers > 0 else False,
            collate_fn=collate_fn_with_indices # Use our custom collate function
        )
        
        # Shape deteced
        try:
            sample_feature, _, _ = dataset[0] 
            input_shape = sample_feature.shape
            seconds_per_example = (1280 * input_shape[0]) / 16000 
            print_info(f"Input Shape Detected: {input_shape} ({seconds_per_example:.2f}s context)")
        except Exception as e:
            print_info(f"[ERROR] Data integrity check failed: {e}")
            sys.exit(1)
        
        # MODEL INITIALIZATION
        print_info("Initializing Neural Architecture...")
        
        nww = Model(
            n_classes=1, 
            input_shape=input_shape,
            config=config,
            model_name=config.get("model_name", atoGeNm(config.get("model_type", "dnn"))),
            model_type=config.get("model_type", "dnn"),
            layer_dim=config["layer_size"],
            n_blocks=config["n_blocks"],
            dropout_prob=config.get("dropout_prob", 0.5),
            seconds_per_example=seconds_per_example
        )

        from nanowakeword.train.train_model import Trainer
        trainer_instance = Trainer(model=nww, config=config)
        # nww.setup_optimizer_and_scheduler(config=config)
        
        # EXECUTE TRAINING
        print_step_header("Training is progress")

        best_model = trainer_instance.auto_train(
            # model=nww,
            X_train=X_train,
            steps=config.get("steps", 15000),
            debug_path=artifacts_dir,
            table_updater=dynamic_table,
            resume_from_dir=args.resume 
        )

        nww.plot_history(artifacts_dir)
        
        training_end_time = time.time()
        training_duration_minutes = (training_end_time - training_start_time) / 60

        from nanowakeword._export.onnx import export_onnx_model
        export_onnx_model(
            model=best_model, 
            input_shape=input_shape, 
            config=config, 
            model_name=config.get("model_name", atoGeNm(config.get("model_type", "dnn"))), 
            output_dir=model_save_dir
        )

        from nanowakeword._export.pytorch import export_pytorch_model
        export_pytorch_model(
            model=best_model,
            model_name=config.get("model_name", atoGeNm(config.get("model_type", "dnn"))),
            output_dir=model_save_dir
        )

        if config.get("enable_journaling", True):
            final_metrics = {}
            if nww.history.get("final_report"):
                report = nww.history["final_report"]
                final_metrics["Stable Loss"] = report.get("Average Stable Loss", "N/A")
                final_metrics["Avg. Pos Conf"] = report.get("Avg. Positive Score (Logit)", "N/A")
                final_metrics["Avg. Neg Conf"] = report.get("Avg. Negative Score (Logit)", "N/A")

            final_metrics["Train Time"] = f"{training_duration_minutes:.1f}"
            base_output_directory = os.path.abspath(base_config["output_dir"])

            update_training_journal(
                base_output_dir=base_output_directory,
                model_name=config.get("model_name", atoGeNm(config.get("model_type", "dnn"))),
                metrics=final_metrics,
                current_config=config.report()
            )

if __name__ == '__main__':
    train()