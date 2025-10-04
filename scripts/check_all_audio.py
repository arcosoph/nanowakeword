
# check_all_audio.py
# -----------------------------------------
# Description:
# This script verifies all audio files in the training dataset.
# It checks if each .wav file:
# 1. Has the correct sample rate (16000 Hz)
# 2. Has the correct number of channels (Mono)
# If any file is incorrect, it lists them for correction.
# Supports multiple folders: Positive, Negative, RIR (Reverb), Noise
# -----------------------------------------

import soundfile as sf
import os
import glob

# --- Configuration ---
TRAINING_DATA_DIR = "training_data"
CORRECT_SAMPLE_RATE = 16000
CORRECT_CHANNELS = 1
problematic_files = []
# --------------------

def check_audio_files(folder_path, folder_name):
    """Checks all .wav files in the given folder for correct format."""
    wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
    if not wav_files:
        print(f"\n- ⚠️ WARNING: No .wav files found in '{folder_name}' folder ('{folder_path}')")
        return
    print(f"\n--- Checking files in: {folder_name} folder ---")
    for file_path in wav_files:
        try:
            info = sf.info(file_path)
            is_correct = True
            print(f"\nFile: {os.path.basename(file_path)}")
            
            # Check sample rate
            if info.samplerate != CORRECT_SAMPLE_RATE:
                print(f"  ❌ Incorrect Sample Rate: {info.samplerate} Hz (Should be {CORRECT_SAMPLE_RATE} Hz)")
                is_correct = False
            else:
                print(f"  ✅ Correct Sample Rate: {info.samplerate} Hz")
            
            # Check channels (Mono)
            if info.channels != CORRECT_CHANNELS:
                print(f"  ❌ Incorrect Channels: {info.channels} (Should be {CORRECT_CHANNELS} for Mono)")
                is_correct = False
            else:
                print(f"  ✅ Correct Channels: {info.channels} (Mono)")
            
            if not is_correct:
                problematic_files.append(file_path)
        
        except Exception as e:
            print(f"\n❌ ERROR: Could not read file '{os.path.basename(file_path)}'. Reason: {e}")
            problematic_files.append(file_path)

# --- Main Program ---
print("Starting audio file verification for ALL training data...")

check_audio_files(os.path.join(TRAINING_DATA_DIR, "positive"), "Positive")
check_audio_files(os.path.join(TRAINING_DATA_DIR, "negative"), "Negative")
# Uncomment if you have these folders
check_audio_files(os.path.join(TRAINING_DATA_DIR, "rir"), "RIR (Reverb)")
check_audio_files(os.path.join(TRAINING_DATA_DIR, "noise"), "Noise")

# --- Final Report ---
print("\n\n--- FINAL REPORT ---")
if not problematic_files:
    print("✅✅✅ Congratulations! All audio files (Positive, Negative, RIR, Noise) are in the correct format.")
    print("Your data is ready for training!")
else:
    print(f"❌ Found {len(problematic_files)} problematic file(s).")
    print("Please fix the following files by converting them to 16000 Hz, Mono, 16-bit WAV using Audacity:")
    for file in problematic_files:
        print(f"  - {file}")
print("--------------------")
