# batch_audio_converter.py
# -----------------------------------------
# Description:
# This script batch-processes all audio files in a folder, converting them to:
# - 16 kHz sample rate
# - Mono channel
# - Exactly 1 second duration (trimmed or padded)
# - Microsoft WAV PCM 16-bit format
# Supported input formats: WAV, MP3, FLAC, OGG, M4A, AAC
# Converted files are saved in a separate output folder.
# -----------------------------------------

import os
import soundfile as sf
import scipy.signal
import numpy as np

# --- Configuration ---
INPUT_FOLDER = "audio"              # Folder to read original audio files
OUTPUT_FOLDER = "batch_audio_preprocess_output"   # Folder to save converted audio files
TARGET_SAMPLE_RATE = 16000
TARGET_DURATION_SEC = 1
TARGET_NUM_SAMPLES = TARGET_SAMPLE_RATE * TARGET_DURATION_SEC
# ---------------------

def convert_audio_file(input_path, output_path):
    """Convert a single audio file to the target format."""
    # Load audio
    data, samplerate = sf.read(input_path)

    # Convert stereo to mono if necessary
    if len(data.shape) == 2:
        data = data.mean(axis=1)

    # Resample if needed
    if samplerate != TARGET_SAMPLE_RATE:
        number_of_samples = round(len(data) * float(TARGET_SAMPLE_RATE) / samplerate)
        data = scipy.signal.resample(data, number_of_samples)

    # Trim or pad to exactly 1 second
    if len(data) > TARGET_NUM_SAMPLES:
        data = data[:TARGET_NUM_SAMPLES]
    elif len(data) < TARGET_NUM_SAMPLES:
        padding = np.zeros(TARGET_NUM_SAMPLES - len(data))
        data = np.concatenate((data, padding))

    # Save in Microsoft WAV PCM 16-bit format
    sf.write(output_path, data, TARGET_SAMPLE_RATE, subtype='PCM_16')

def batch_convert(input_dir, output_dir):
    """Batch-convert all supported audio files in a folder."""
    os.makedirs(output_dir, exist_ok=True)
    supported_exts = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(supported_exts):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".wav")
                print(f"🔄 Converting: {file}")
                try:
                    convert_audio_file(input_path, output_path)
                    print(f"✅ Done: {file}")
                except Exception as e:
                    print(f"❌ Error: {file} – {e}")

if __name__ == "__main__":
    batch_convert(INPUT_FOLDER, OUTPUT_FOLDER)
