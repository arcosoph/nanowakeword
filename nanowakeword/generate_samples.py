# Copyright 2025 Arcosoph. All rights reserved.
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



import argparse
import logging
import os
import sys
from pathlib import Path
import wave
import traceback
from tqdm import tqdm
import os
from nanowakeword.utils.audio_processing import download_file
import numpy as np
import scipy.signal as sps

try:
    from nanowakeword import PROJECT_ROOT
except ImportError:

    logging.warning("Could not import PROJECT_ROOT, attempting to determine path manually.")
    try:
        PROJECT_ROOT = Path(__file__).resolve().parent.parent
    except NameError: # __file__ is not defined in interactive shells
        PROJECT_ROOT = Path('.').resolve()





try:
    from piper.voice import PiperVoice
except ImportError:
    print("CRITICAL ERROR: 'piper-python' is not installed or not in the Python path.")
    print("Please install it using: pip install piper-lang")
    sys.exit(1)

_LOGGER = logging.getLogger("generate_samples")

def generate_samples(
    text,
    output_dir,
    max_samples,
    batch_size=16,
    noise_scales=[0.667],
    noise_scale_ws=[0.8],
    length_scales=[1.0],
    file_names=None,
    model_path=None, 
    **kwargs,
):
    """
    Uses a specific Piper TTS model to generate audio samples from text.
    This version includes robust debugging, introspection, and configuration handling.
    """

    import os
    from pathlib import Path
    from nanowakeword.utils.download_file import download_file

    # Model path
    model_path = PROJECT_ROOT / "resources" / "tts_models" / "en_US-ryan-low.onnx"

    # Folder create korbo jodi na thake
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Hugging Face ONNX URL
    onnx_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/low/en_US-ryan-low.onnx"
    json_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/low/en_US-ryan-low.onnx.json"

    # File check + download
    if not model_path.exists():
        print("Model not found. Downloading...")
        download_file(onnx_url, target_directory=model_path.parent.as_posix())
        download_file(json_url, target_directory=model_path.parent.as_posix())
        print("Download complete.")
    


    if not model_path or not os.path.exists(model_path):
        _LOGGER.error(f"Piper voice model not found at the specified path: {model_path}")
        return
 
    _LOGGER.info(f"Loading Piper model from {model_path}")
    try:
        voice = PiperVoice.load(model_path)
    except Exception as e:
        _LOGGER.error(f"Failed to load Piper model: {e}")
        _LOGGER.error(traceback.format_exc())
        return
    
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(text, str):
        text = [text]

    if not text:
        _LOGGER.warning("Input text list is empty. Nothing to generate.")
        return

    num_repeats = (max_samples // len(text)) + 1
    text_prompts = text * num_repeats
    text_prompts = text_prompts[:max_samples]
    
    if file_names and len(file_names) == len(text_prompts):
        file_map = list(zip(text_prompts, file_names))
    else:
        file_map = [ (prompt, f"sample_{i}_{hash(prompt) % 10000}.wav") for i, prompt in enumerate(text_prompts) ]

    TARGET_SAMPLE_RATE = 16000
    _LOGGER.info(f"Generating {len(file_map)} samples...")

    for index, (text_prompt, out_file) in enumerate(file_map):
        try:
            out_path = os.path.join(output_dir, out_file)
            
            # audio_generator = voice.synthesize(text_prompt, output_format="int16")
            audio_generator = voice.synthesize(text_prompt)

            
            all_audio_bytes = []
            for audio_chunk in audio_generator:
                if hasattr(audio_chunk, 'audio_int16_bytes') and audio_chunk.audio_int16_bytes:
                    all_audio_bytes.append(audio_chunk.audio_int16_bytes)
                elif isinstance(audio_chunk, bytes):
                    all_audio_bytes.append(audio_chunk)
            
            audio_bytes = b"".join(all_audio_bytes)
            
            if not audio_bytes:
                _LOGGER.warning(f"No audio data was generated for text: '{text_prompt}'. Skipping file creation.")
                continue

            source_sample_rate = voice.config.sample_rate

            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

            if source_sample_rate != TARGET_SAMPLE_RATE:
                _LOGGER.debug(f"Resampling from {source_sample_rate} Hz to {TARGET_SAMPLE_RATE} Hz...")
                num_samples = int(len(audio_array) * TARGET_SAMPLE_RATE / source_sample_rate)
                resampled_audio = sps.resample(audio_array, num_samples)
                
                
                final_audio_array = resampled_audio.astype(np.int16)
            else:
                final_audio_array = audio_array

    
            with wave.open(out_path, "wb") as audio_file:
                audio_file.setnchannels(1)  # Mono
                audio_file.setsampwidth(2) # 16-bit
                audio_file.setframerate(TARGET_SAMPLE_RATE) # 16000 Hz
                audio_file.writeframes(final_audio_array.tobytes())

        except Exception as e:
            _LOGGER.error(f"An unexpected error occurred during generation for '{text_prompt}': {e}")
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
    
   
    parser.add_argument(
        "--model",
        default=None,
        help="Optional: Path to the .onnx voice model. If not provided, the default model will be used."
    )
    
    args = parser.parse_args()
        
    logging.basicConfig(level=logging.INFO)
    

    generate_samples(
        text=args.text,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        model_path=args.model
    )
