# NanoWakeWord
# Copyright 2025 Arcosoph
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Project Repository: https://github.com/arcosoph/nanowakeword
#
# This software is provided "AS IS", without warranties or conditions of any kind.
# See the License for the specific language governing permissions and limitations.

import argparse
import logging
import os
import sys
import wave
import traceback
import random
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import numpy as np
import scipy.signal as sps

try:
    from nanowakeword import PROJECT_ROOT
except ImportError:
    logging.warning("Could not import PROJECT_ROOT, attempting to determine path manually.")
    try:
        PROJECT_ROOT = Path(__file__).resolve().parent.parent
    except NameError:
        PROJECT_ROOT = Path('.').resolve()

try:
    from piper.voice import PiperVoice, SynthesisConfig
except ImportError:
    print("CRITICAL ERROR: 'piper-tts' is not installed or not in the Python path.")
    print("Please install it using: pip install piper-tts")
    sys.exit(1)

try:
    from nanowakeword.utils.download_file import download_file
except ImportError:
    print("Could not import download_file. Please ensure it exists in nanowakeword/utils/")
    def download_file(url, target_directory):
        print(f"ERROR: download_file utility not found. Cannot download model from {url}")
        sys.exit(1)

_LOGGER = logging.getLogger("generate_samples")

_LOADED_VOICES = None

def _get_or_load_voices():
    """
    Internal function to load voices only once.
    It checks a global cache before loading from disk.
    """
    global _LOADED_VOICES

    if _LOADED_VOICES is not None:
        _LOGGER.debug("Using cached TTS voices.")
        return _LOADED_VOICES

    models_dir = PROJECT_ROOT / "resources" / "tts_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    existing_model_paths = list(models_dir.glob("*.onnx"))

    if len(existing_model_paths) < 3:
        prompt = (
            f"\nINFO: Found {len(existing_model_paths)} TTS voice model(s). For best results, 3+ voices are recommended.\n"
            "Would you like to download 3 default models for improved variety?\n\n"
            "Enter your choice: [y]es / [n]o / [c]ustom URL > "
        )
        choice = input(prompt).lower().strip()

        if choice == 'y':
            # List of default models
            voices_to_download = [
                {
                    "name": "en_US-ryan-low",
                    "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/low/en_US-ryan-low.onnx",
                    "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/low/en_US-ryan-low.onnx.json"
                },
                {
                    "name": "en_US-ljspeech-medium",
                    "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx",
                    "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx.json"
                },
                {
                    "name": "en_GB-alan-low",
                    "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/low/en_GB-alan-low.onnx",
                    "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/low/en_GB-alan-low.onnx.json"
                }
            ]
            
        
            existing_model_names = {p.stem for p in existing_model_paths}
            for voice_info in voices_to_download:
                if voice_info['name'] not in existing_model_names:
                    _LOGGER.info(f"Downloading model: {voice_info['name']}...")
                    download_file(voice_info['onnx_url'], target_directory=models_dir.as_posix())
                    download_file(voice_info['json_url'], target_directory=models_dir.as_posix())
                    _LOGGER.info(f"'{voice_info['name']}' download complete.")

        elif choice == 'c':
            _LOGGER.info("Please provide the URLs for your custom Piper model.")
            onnx_url = input("Enter the direct URL for the .onnx file: ").strip()
            json_url = input("Enter the direct URL for the .onnx.json file: ").strip()

            if onnx_url.startswith("http") and json_url.startswith("http"):
                _LOGGER.info("Downloading custom model...")
                download_file(onnx_url, target_directory=models_dir.as_posix())
                download_file(json_url, target_directory=models_dir.as_posix())
                _LOGGER.info("Custom model downloaded successfully.")
            else:
                _LOGGER.error("Invalid URLs provided. Skipping download.")

    
    final_model_paths = list(models_dir.glob("*.onnx"))

    
    if not final_model_paths:
        _LOGGER.error("FATAL: No TTS models found. Cannot proceed with audio generation.")
        _LOGGER.info("Please add at least one Piper model to the 'resources/tts_models' directory or re-run and choose to download.")
        return 

    _LOGGER.info(f"Found {len(final_model_paths)} TTS models in total. Pre-loading them...")
    try:
        voices = [PiperVoice.load(model_path) for model_path in tqdm(final_model_paths, desc="Loading voices")]

        _LOADED_VOICES = voices
        print()
        return _LOADED_VOICES

    except Exception as e:
        _LOGGER.error(f"Failed to load one or more TTS models: {e}")
        _LOGGER.error(traceback.format_exc())
        return


def generate_samples(
    text,
    output_dir,
    max_samples,
    file_names=None,
    **kwargs, 
):
    """
    Generates diverse audio samples by randomly selecting from multiple pre-defined TTS voices.
    """

    voices = _get_or_load_voices()

    if not voices:
        _LOGGER.error("Audio generation skipped as no TTS models could be loaded.")
        return

    os.makedirs(output_dir, exist_ok=True)

    if isinstance(text, str): text = [text]
    if not text:
        _LOGGER.warning("Input text list is empty. Nothing to generate.")
        return

    num_repeats = (max_samples // len(text)) + 1
    text_prompts = text * num_repeats
    text_prompts = text_prompts[:max_samples]

    file_map = [(prompt, f"sample_{i}_{hash(prompt) % 10000}.wav") for i, prompt in enumerate(text_prompts)]

    TARGET_SAMPLE_RATE = 16000
    _LOGGER.info(f"Generating {len(file_map)} samples...")

    for index, (text_prompt, out_file) in tqdm(enumerate(file_map), total=len(file_map), desc="Generating Audio"):
        try:
            out_path = os.path.join(output_dir, out_file)

            voice = random.choice(voices)

            synthesis_config = SynthesisConfig(
                length_scale=random.uniform(0.9, 1.1),
                noise_scale=random.uniform(0.667, 0.8),
                noise_w_scale=random.uniform(0.8, 1.0)
            )

            audio_generator = voice.synthesize(text_prompt, syn_config=synthesis_config)
            audio_bytes = b"".join([chunk.audio_int16_bytes for chunk in audio_generator])

            if not audio_bytes: continue

            source_sample_rate = voice.config.sample_rate
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

            if source_sample_rate != TARGET_SAMPLE_RATE:
                num_samples = int(len(audio_array) * TARGET_SAMPLE_RATE / source_sample_rate)
                final_audio_array = sps.resample(audio_array, num_samples).astype(np.int16)
            else:
                final_audio_array = audio_array

            with wave.open(out_path, "wb") as audio_file:
                audio_file.setnchannels(1)
                audio_file.setsampwidth(2)
                audio_file.setframerate(TARGET_SAMPLE_RATE)
                audio_file.writeframes(final_audio_array.tobytes())

        except Exception as e:
            _LOGGER.error(f"An unexpected error occurred: {e}")
            _LOGGER.error(traceback.format_exc())
            continue

    _LOGGER.info("Sample generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate audio samples using a Piper TTS model."
    )
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output_dir", required=True, help="Directory to save audio files")
    parser.add_argument("--max_samples", type=int, default=1, help="Number of samples to generate")

    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')

    generate_samples(
        text=args.text,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )