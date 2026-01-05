# ==============================================================================
#           NanoWakeWord Model Evaluation Script 
# Corrected to select the FIRST 'n' samples instead of random ones
# for consistent and fair model comparison.
# ==============================================================================

import os
import sys
import numpy as np
import scipy.io.wavfile
from glob import glob
from tqdm import tqdm

# Import the interpreter class from the library
from nanowakeword.nanointerpreter import NanoInterpreter

# ==============================================================================
#                           CONFIGURATION
# ==============================================================================
MODEL_PATH = r"trained_models/arcosoph_B_v3/model/arcosoph_B_v3.onnx"
POSITIVE_DIR = r"data/generated_positive"
NEGATIVE_SPEECH_DIR = r"data/generated_negative"
NEGATIVE_NOISE_DIR = r"data/background_noise"
THRESHOLD = 0.80
CHUNK_SIZE = 1280

# Set a number to limit how many files are tested from the START of each folder.
# Set to None to test ALL files.
MAX_SAMPLES_PER_FOLDER = 900 # ▽▽▽ CHANGE THIS VALUE OR SET TO None ▽▽▽

# ==============================================================================
#                           HELPER FUNCTIONS
# ==============================================================================

def get_limited_files(folder_path, max_samples):
    """
    Gets a list of .wav files from a folder, optionally limiting the count.
    It sorts the files alphabetically and takes the first 'max_samples' files
    to ensure consistency across different test runs.
    """
    if not os.path.isdir(folder_path):
        print(f"\nWarning: Directory not found: {folder_path}")
        return []
        
    all_files = glob(os.path.join(folder_path, "*.wav"))
    if not all_files:
        print(f"\nWarning: No .wav files found in {folder_path}")
        return []

    # Sort the files alphabetically to ensure a consistent order
    all_files.sort()
    
    if max_samples is not None:
        print(f"(Selecting the first {min(max_samples, len(all_files))} samples from {len(all_files)} total)")
        return all_files[:max_samples] # Take a slice from the beginning
    else:
        return all_files

def load_wav_file(path):
    try:
        sample_rate, data = scipy.io.wavfile.read(path)
        if data.dtype != np.int16:
            raise TypeError(f"Unsupported audio format: {data.dtype}")
        
        if sample_rate != 16000:
            num_samples = int(len(data) * 16000 / sample_rate)
            data = np.interp(
                np.linspace(0, len(data) - 1, num_samples),
                np.arange(len(data)),
                data
            ).astype(np.int16)

        if len(data.shape) > 1 and data.shape[1] > 1:
            data = np.mean(data, axis=1).astype(np.int16)
        
        return data
    except Exception as e:
        print(f"\nWarning: Could not load or process file '{os.path.basename(path)}'. Error: {e}")
        return None

def process_file(interpreter, audio_data, key):
    if audio_data is None:
        return 0.0
    interpreter.reset()
    max_score = 0.0
    for i in range(0, len(audio_data), CHUNK_SIZE):
        chunk = audio_data[i:i + CHUNK_SIZE]
        if len(chunk) < CHUNK_SIZE:
            padding = np.zeros(CHUNK_SIZE - len(chunk), dtype=np.int16)
            chunk = np.concatenate((chunk, padding))
        score = interpreter.predict(chunk).get(key, 0.0)
        if score > max_score:
            max_score = score
    return max_score
    
# ==============================================================================
#                               MAIN SCRIPT
# ==============================================================================

if __name__ == "__main__":

    print("-" * 60)
    print("Initializing NanoWakeWord Model Evaluator...")
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Error: Model not found at '{MODEL_PATH}'")
    try:
        interpreter = NanoInterpreter.load_model(MODEL_PATH)
        key = list(interpreter.models.keys())[0]
        print(f"Model '{os.path.basename(MODEL_PATH)}' loaded.")
        print(f"Wakeword to detect: '{key}'")
        print(f"Detection Threshold: {THRESHOLD}")
        print(f"Max Samples per Folder: {'All' if MAX_SAMPLES_PER_FOLDER is None else MAX_SAMPLES_PER_FOLDER} (from the beginning)")
        print("-" * 60)
    except Exception as e:
        sys.exit(f"Error initializing interpreter: {e}")

    print("\n>>> STEP 1: Evaluating POSITIVE samples (checking for Misses)...")
    positive_files = get_limited_files(POSITIVE_DIR, MAX_SAMPLES_PER_FOLDER)
    misses = 0
    if positive_files:
        for file_path in tqdm(positive_files, desc="Positive"):
            audio = load_wav_file(file_path)
            score = process_file(interpreter, audio, key)
            if score < THRESHOLD:
                misses += 1
    
    print("\n>>> STEP 2: Evaluating NEGATIVE samples (checking for False Alarms)...")
    negative_speech_files = get_limited_files(NEGATIVE_SPEECH_DIR, MAX_SAMPLES_PER_FOLDER)
    negative_noise_files = get_limited_files(NEGATIVE_NOISE_DIR, MAX_SAMPLES_PER_FOLDER)
    all_negative_files = negative_speech_files + negative_noise_files
    
    false_alarms = 0
    if all_negative_files:
        for file_path in tqdm(all_negative_files, desc="Negative"):
            audio = load_wav_file(file_path)
            score = process_file(interpreter, audio, key)
            if score > THRESHOLD:
                false_alarms += 1

    print("\n" + "=" * 60)
    print("                  EVALUATION COMPLETE - FINAL REPORT")
    print("=" * 60)
    print(f"Model Tested: {os.path.basename(MODEL_PATH)}")
    print("-" * 60)
    
    total_positives = len(positive_files)
    miss_rate = (misses / total_positives) * 100 if total_positives > 0 else 0
    print(f"Positive Samples (Misses):")
    print(f"  - Total Files Tested: {total_positives}")
    print(f"  - Files Missed:       {misses} (Score < {THRESHOLD})")
    print(f"  - Success Rate:       {100 - miss_rate:.2f}%")
    
    print("-" * 60)

    total_negatives = len(all_negative_files)
    fa_rate = (false_alarms / total_negatives) * 100 if total_negatives > 0 else 0
    print(f"Negative Samples (False Alarms):")
    print(f"  - Total Files Tested: {total_negatives}")
    print(f"  - False Alarms:       {false_alarms} (Score > {THRESHOLD})")
    print(f"  - Correct Rejection Rate: {100 - fa_rate:.2f}%")
    
    print("=" * 60)