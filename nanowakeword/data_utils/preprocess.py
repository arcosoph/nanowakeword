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



import os
import torch
import torchaudio
from tqdm import tqdm
import logging
import tempfile
import shutil
import warnings 

logging.basicConfig(level=logging.INFO)

TARGET_SR = 16000
TARGET_CHANNELS = 1
TARGET_BITS_PER_SAMPLE = 16
TARGET_ENCODING = "PCM_S"


def needs_conversion(file_path):
    """
    Checks if an audio file needs to be converted to the standard format.
    Returns True if conversion is needed, False otherwise.
    """
    try:
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            info = torchaudio.info(file_path)
        
        if not file_path.lower().endswith('.wav'):
            return True

        if info.sample_rate != TARGET_SR:
            return True
        if info.num_channels != TARGET_CHANNELS:
            return True
        if hasattr(info, 'bits_per_sample') and info.bits_per_sample != TARGET_BITS_PER_SAMPLE:
            return True
            
        return False
    except Exception as e:
        logging.warning(f"Could not read info for {file_path}, skipping. Error: {e}")
        return False 

def process_and_convert_audio(file_path):
    """
    Converts a single audio file to the standard format and overwrites it.
    Returns True if a conversion was made, False otherwise.
    """
    if not needs_conversion(file_path):
        return False 

    try:
        waveform, sr = torchaudio.load(file_path)

        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
            waveform = resampler(waveform)
       
        if waveform.shape[0] > TARGET_CHANNELS:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)

        torchaudio.save(
            temp_path, waveform, sample_rate=TARGET_SR,
            encoding=TARGET_ENCODING, bits_per_sample=TARGET_BITS_PER_SAMPLE
        )
        
        shutil.move(temp_path, file_path)
        return True 

    except Exception as e:
        logging.error(f"Failed to convert {file_path}. Error: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return False
         

def verify_and_process_directory(dir_path):
    """
    Verifies all audio files in a directory and converts them if necessary.
    """
    if not os.path.isdir(dir_path):
        logging.warning(f"Directory not found, skipping preprocessing: {dir_path}")
        return

    logging.info(f"Verifying and preprocessing audio files in: {dir_path}")
    
    # all_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]
    all_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.aiff', '.alac', '.opus', '.pcm'))]
    converted_count = 0
    for file_path in tqdm(all_files, desc=f"Processing {os.path.basename(dir_path)}"):
        if os.path.isfile(file_path):
            if process_and_convert_audio(file_path):
                converted_count += 1
    logging.info(f"Finished processing directory: {dir_path}. Converted {converted_count} files.")