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

# (✿◕‿◕✿) - Your wish for even more power is granted!

import os
import torch

# New imports for the Phoneme Adversarial Generator
from nanowakeword.data.generator.adversarial_texts import (
    adversarial_texts,
    get_phonemizer_model,
    PhonemeAdversarialGenerator,
    collapse_repeated_letters
)
from nanowakeword.data.generator.generate_samples import generate_samples
from nanowakeword.utils.logger import print_step_header, print_info, print_warning, print_error

def generate_clips(base_config):
    


    """Activates the flexible, task-based synthetic data generation engine.

    This function serves as the central orchestrator for creating synthetic audio
    clips. It operates based on a list of "generation tasks" defined in the
    main configuration file under the `data_generation_tasks` key. This
    task-based approach grants the user fine-grained control over the entire
    data generation process, allowing for the creation of multiple, diverse
    datasets (e.g., positive, negative, validation) in a single run.

    Each task is an independent job that specifies what text to synthesize, how
    many samples to create, where to save them, and what Text-To-Speech (TTS)
    settings to use. This modularity empowers users to build complex and robust
    datasets tailored to their specific needs.

    The primary workflow is as follows:
    1.  Loads the list of tasks from the configuration.
    2.  Pre-loads any globally required models (like the phonemizer) for efficiency.
    3.  Iterates through each enabled task.
    4.  For each task, it determines the text source and generates the list of
        phrases to be synthesized.
    5.  It then calls the `generate_samples` utility to create the audio files.
    6.  Clears the GPU cache after heavy tasks to maintain performance.

    Configuration Schema (`data_generation_tasks`):
        The `data_generation_tasks` key in your config file should be a list of
        dictionaries, where each dictionary represents a single task.

        Task Keys:
            name (str): A descriptive name for the task (e.g., "Positive Wake Words").
            enabled (bool): If `False`, this task will be skipped. Defaults to `True`.
            output_dir (str): The path to the directory where audio clips will be saved.
            num_samples (int): The total number of audio clips to generate for this task.
            file_prefix (str): A prefix for the generated audio filenames (e.g., "pos_").
            tts_settings (dict, optional): Task-specific TTS settings that override
                                           the global `tts_settings`.
            text_source (dict): A dictionary defining the source of the text to be
                                synthesized. This is the core of the task's logic.

    The `text_source` Dictionary:
        This dictionary must contain a `type` key, which determines how the text
        is generated. Supported types are:

        1.  `type: "fixed_phrase"`
            Generates audio for a single, repeated phrase. Ideal for positive
            wake word samples.
            - `phrase` (str, optional): The exact phrase to use. If not provided,
              it falls back to the global `target_phrase`.

        2.  `type: "from_list"`
            Generates audio from a user-provided list of phrases. Perfect for
            curated lists of negative samples.
            - `phrases` (list[str]): A list of custom text phrases.
            - `repeat_each` (int, optional): How many times to repeat each phrase
              in the list. Defaults to 1.

        3.  `type: "auto_adversarial"`
            Generates phonetically similar but common English words/phrases.
            Excellent for creating a robust set of negative samples that challenge
            the model with real-world, confusable words.
            - `base_phrase` (str, optional): The phrase to generate variations
              from. Falls back to the global `target_phrase`.
            - Supports other keys like `include_partial_phrase`, `max_multi_word_len`, etc.

        4.  `type: "phoneme_adversarial"`
            Generates nonsensical but phonetically very similar text by manipulating
            the phonemes of a base phrase. This creates extremely challenging
            negative samples to drastically reduce false activations.
            - `base_phrase` (str, optional): The phrase to generate variations
              from. Falls back to the global `target_phrase`.
            - `min_distance` (float, optional): Controls how different the generated
              phoneme strings are from the original. Defaults to 0.35.

    Example Usage (in a .yaml config file):
        ```yaml
        target_phrase: "hey nano"
        phonemizer_model_path: "path/to/your/phonemizer.pt"

        data_generation_tasks:
          - name: "Positive Wake Words"
            enabled: true
            output_dir: "dataset/positive"
            num_samples: 1000
            text_source:
              type: "fixed_phrase"
              # Uses the global "hey nano" target_phrase

          - name: "Phoneme-Based Hard Negatives"
            enabled: true
            output_dir: "dataset/negative"
            num_samples: 1500
            file_prefix: "neg_phoneme"
            text_source:
              type: "phoneme_adversarial"
              min_distance: 0.4
        ```

    Args:
        base_config (dict): The main configuration dictionary loaded from the
            project's config file. It is expected to contain keys like
            `data_generation_tasks`, `tts_settings`, etc.

    Side Effects:
        - Creates directories specified in `output_dir` for each task.
        - Writes `.wav` audio files into these directories.
        - Prints progress and status information to the console.
    """


    print_step_header("Activating Synthetic Data Generation Engine")

    generation_tasks = base_config.get("data_generation_tasks")
    if not generation_tasks or not isinstance(generation_tasks, list):
        print_info("No 'data_generation_tasks' found in the configuration. Skipping generation.")
        return

    global_tts_settings = base_config.get("tts_settings", {})
    global_target_phrase = base_config.get("target_phrase", None)

    #  Pre-load phonemizer model if any task requires it (for efficiency) 
    phonemizer_model = None
    if any(task.get("text_source", {}).get("type") == "phoneme_adversarial" for task in generation_tasks):
        phonemizer_path = os.path.join("NwwResourcesModel", "phonemize_model", "phonemize_m1.pt")
        print_info(f"Loading phonemizer model from: {phonemizer_path}")
        phonemizer_model = get_phonemizer_model(phonemizer_path)
        print_info("Phonemizer model loaded successfully.")

    print_info(f"Found {len(generation_tasks)} generation tasks defined in the configuration.")

    # Execute Generation Engine 
    for i, task in enumerate(generation_tasks):
    # for task_name, task_params in generation_tasks_dict.items():
        task_name = task.get("name", f"Unnamed Task {i+1}")
        
        if not task.get("enabled", True):
            print_info(f"\n Skipping Task: '{task_name}' (disabled)")
            continue

        print_info(f"Executing Task: '{task_name}'")

        output_dir = task.get("output_dir")
        num_samples = int(task.get("num_samples", 0))
        text_source_config = task.get("text_source")

        if not all([output_dir, num_samples > 0, text_source_config]):
            print_warning(f"Task '{task_name}' is misconfigured. Skipping.")
            continue

        # Prepare the List of Texts for Generation 
        final_texts = []
        source_type = text_source_config.get("type")
        
        if not source_type:
            source_type = 'fixed_phrase'

        if source_type == "fixed_phrase":
            phrase = text_source_config.get("phrase", global_target_phrase)
            if not phrase:
                print_error(f"Task '{task_name}' needs a 'phrase'. Skipping.")
                continue
            final_texts = [phrase]
            print_info(f"Source: Fixed phrase -> '{phrase}'")

        elif source_type == "from_list":
            phrases = text_source_config.get("phrases", [])
            repeats = int(text_source_config.get("repeat_each", 1))
            if not phrases:
                print_warning(f"Task '{task_name}' has an empty 'phrases' list. Skipping.")
                continue
            for p in phrases:
                final_texts.extend([p] * repeats)
            print_info(f"Source: Custom list of {len(phrases)} phrases, repeated {repeats} time(s) each.")

        elif source_type == "auto_adversarial":
            base_phrase = text_source_config.get("base_phrase", global_target_phrase)
            if not base_phrase:
                print_error(f"Task '{task_name}' needs a 'base_phrase'. Skipping.")
                continue
            print_info(f"Source: Auto-generating {num_samples} word-based adversarial phrases from '{base_phrase}'.")
            adv_params = {k: text_source_config.get(k) for k in ["include_input_words", "include_partial_phrase", "multi_word_prob", "max_multi_word_len"] if text_source_config.get(k) is not None}
            final_texts = adversarial_texts(base_phrase, N=num_samples, **adv_params)

        elif source_type == "phoneme_adversarial":
            if not phonemizer_model:
                print_error(f"Phonemizer model not loaded, cannot execute task '{task_name}'. Skipping.")
                continue
            
            base_phrase = text_source_config.get("base_phrase", global_target_phrase)
            if not base_phrase:
                print_error(f"Task '{task_name}' needs a 'base_phrase' for phoneme generation. Skipping.")
                continue

            min_distance = float(text_source_config.get("min_distance", 0.35))
            print_info(f"Source: Generating {num_samples} phoneme-based adversarial texts from '{base_phrase}'.")
            print_info(f"Using minimum phoneme distance: {min_distance}")

            generator = PhonemeAdversarialGenerator(phonemizer_model, min_distance=min_distance)
            variants = generator.generate(base_phrase, num_samples)
            final_texts = [collapse_repeated_letters(v) for v in variants]
            
        else:
            print_warning(f"Unknown text_source type: '{source_type}' in task '{task_name}'. Skipping.")
            continue

        if not final_texts:
            print_warning(f"No texts were generated for task '{task_name}'. Skipping.")
            continue

        # Configure TTS & Run Generation 
        task_tts_settings = global_tts_settings.copy()
        task_tts_settings.update(task.get("tts_settings", {}))
        
        print_info(f"Generating {num_samples} clips -> '{output_dir}'")
        os.makedirs(output_dir, exist_ok=True)
        
        generate_samples(
            text=final_texts,
            max_samples=num_samples,
            output_dir=output_dir,
            file_prefix=task.get("file_prefix", "sample"),
            **task_tts_settings
        )
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print_info("GPU cache cleared.")

    print_info("Synthetic Data Generation Process Finished Successfully")