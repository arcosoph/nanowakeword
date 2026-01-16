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
import torch

from nanowakeword.data.generator.adversarial_texts import adversarial_texts
from nanowakeword.data.generator.generate_samples import generate_samples

from nanowakeword.utils.logger import print_step_header, print_info



def generate_clips(base_config):
    print_step_header("Activating Synthetic Data Generation Engine")

    # Acquire the Target Phrase
    target_phrase = base_config.get("target_phrase")
    if not target_phrase:
        print_info("\n[CONFIGURATION NOTICE]: 'target_phrase' is not set in your config file. This is required to generate audio samples.")
        try:
            user_input = input(">>> Please enter the target phrase to proceed: ").strip()
            if not user_input:
                print_info("\n[ABORT] A target phrase is mandatory for generation. Exiting.")
                sys.exit(1)
            target_phrase = [user_input]
            print_info(f"Using runtime target phrase: '{user_input}'")
        except (KeyboardInterrupt, EOFError):
            print_info("\n\nOperation cancelled by user.")
            sys.exit()

    # 1. Retrieve Sample Counts (Handle missing values safely)
    raw_pos_samples = base_config.get('generate_positive_samples')
    raw_neg_samples = base_config.get('generate_negative_samples')

    tts_settings = base_config.get("tts_settings", {})

    # Convert to integers if present, else default to 0
    n_pos_train = int(raw_pos_samples) if raw_pos_samples is not None else 0

    # 2. Configure Negative Data Generation Strategy
    enable_auto_adversarial = base_config.get("adversarial_text_generation", True)
    custom_negatives = base_config.get("custom_negative_phrases", [])
    repeats_per_phrase = int(base_config.get("custom_negative_per_phrase", 50))

    include_partial_phrase= base_config.get("include_partial_phrase", 0.09)
    include_input_words= base_config.get("include_input_words", 0.2)
    multi_word_prob = base_config.get("multi_word_prob", 0.9)
    max_multi_word_len= base_config.get("max_multi_word_len", 2)

    final_negative_texts = []

    # Custom Negative Phrases 
    if custom_negatives:
        print_info(f"Processing {len(custom_negatives)} custom negative phrases.")
        print_info(f"Generating {repeats_per_phrase} copies for EACH custom phrase.")

        # Expand the list: Each custom phrase is repeated 'repeats_per_phrase' times
        for phrase in custom_negatives:
            final_negative_texts.extend([phrase] * repeats_per_phrase)
        
        print_info(f"Total custom samples prepared: {len(final_negative_texts)}")

    # Gap Filling with Auto-Adversarial Data 
    if raw_neg_samples is not None:
        # Scenario: User provided a specific target total (e.g., 600)
        target_total_neg = int(raw_neg_samples)
        current_count = len(final_negative_texts)
        gap = max(0, target_total_neg - current_count)

        if gap > 0:
            if enable_auto_adversarial:
                print_info(f"Target negative samples: {target_total_neg}. Current custom samples: {current_count}.")
                print_info(f"Generating {gap} auto-adversarial phrases to fill the gap.")
                
                # Generate phonetically similar words to fill the remaining count
                auto_adversarial = adversarial_texts(
                                                                target_phrase[0], 
                                                                N=gap, 
                                                                include_input_words=include_input_words, 
                                                                include_partial_phrase=include_partial_phrase, 
                                                                multi_word_prob=multi_word_prob,
                                                                max_multi_word_len=max_multi_word_len)
                
                final_negative_texts.extend(auto_adversarial)
            else:
                print_info(f"Target is {target_total_neg}, but auto-adversarial generation is DISABLED.")
                print_info(f"Proceeding with only {current_count} custom samples.")
        else:
            if current_count > target_total_neg:
                print_info(f"Note: Custom samples ({current_count}) exceed the target ({target_total_neg}). Keeping all custom samples.")

    # Update final count
    final_neg_count = len(final_negative_texts)

    # Construct Unified Generation Plan
    generation_plan = {}

    if n_pos_train > 0:
        generation_plan["Positive_Train"] = {
            "count": n_pos_train,
            "texts": target_phrase,
            "output_dir": base_config["positive_data_path"],
            "prefix": "pos" 
        }
    
    if final_neg_count > 0:
        generation_plan["Adversarial_Train"] = {
            "count": final_neg_count,
            "texts": final_negative_texts,
            "output_dir": base_config["negative_data_path"],
            "prefix": "neg" 
        }

    # Execute Generation Engine
    if generation_plan:
        print_info(f"Initiating data generation pipeline for phrase: '{target_phrase[0]}'")
        
        for task_name, params in generation_plan.items():
            if params["count"] > 0 and params["texts"]:
                print_info(f"Executing task '{task_name}': {params['count']} clips -> '{params['output_dir']}'")
                os.makedirs(params["output_dir"], exist_ok=True)
                
                generate_samples(
                    text=params["texts"],
                    max_samples=params["count"],
                    output_dir=params["output_dir"],
                    file_prefix=params.get("prefix", "sample"),
                    **tts_settings
                )
                
                # Clear GPU cache after each heavy task to prevent fragmentation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print_info("Synthetic data generation process finished successfully.\n")


