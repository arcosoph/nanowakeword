"""
audio_analyzer.py

This is an advanced audio quality checking tool.
It not only checks for sample rate and mono channel,
but also verifies:

✅ Minimum & maximum duration
✅ Acceptable volume (RMS) level
✅ Signal-to-Noise Ratio (SNR) to detect noisy files

This helps you ensure your dataset is clean and balanced 
before training a wake-word detection model.
"""

import soundfile as sf
import numpy as np
import os
import glob

# --- CONFIGURATION ---
TRAINING_DATA_DIR = "training_data"
CORRECT_SAMPLE_RATE = 16000
CORRECT_CHANNELS = 1

# Acceptable ranges
MIN_DURATION_S = 0.5   # Minimum duration (seconds)
MAX_DURATION_S = 10.0  # Maximum duration (seconds)
MIN_RMS_LEVEL = 0.01   # Min acceptable volume (too quiet if below)
MAX_RMS_LEVEL = 0.5    # Max acceptable volume (too loud if above)
MIN_SNR_DB = 15        # Minimum acceptable SNR (below means noisy)


def analyze_audio_file(file_path):
    """Analyze a single audio file and report problems if found."""
    try:
        audio, samplerate = sf.read(file_path)
        info = sf.info(file_path)
        
        problems = []

        # 1. Technical format checks
        if info.samplerate != CORRECT_SAMPLE_RATE:
            problems.append(f"Incorrect Sample Rate: {info.samplerate} Hz (Expected {CORRECT_SAMPLE_RATE})")
        if info.channels != CORRECT_CHANNELS:
            if len(audio.shape) > 1 and audio.shape[1] == 2:
                audio = audio.mean(axis=1)  # Convert stereo to mono for analysis
            else:
                problems.append(f"Incorrect Channels: {info.channels} (Expected {CORRECT_CHANNELS})")

        # 2. Duration check
        duration = len(audio) / samplerate
        if not (MIN_DURATION_S <= duration <= MAX_DURATION_S):
            problems.append(f"Unusual Duration: {duration:.2f}s (Expected between {MIN_DURATION_S}s and {MAX_DURATION_S}s)")

        # 3. RMS (volume) check
        rms = np.sqrt(np.mean(audio**2))
        if not (MIN_RMS_LEVEL <= rms <= MAX_RMS_LEVEL):
            problems.append(f"Unusual Volume (RMS): {rms:.3f} (Expected between {MIN_RMS_LEVEL} and {MAX_RMS_LEVEL})")

        # 4. SNR check (simplified)
        signal_power = np.mean(np.sort(audio**2)[-len(audio)//10:])
        noise_power = np.mean(np.sort(audio**2)[:len(audio)//5])

        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            if snr < MIN_SNR_DB:
                problems.append(f"LOW SNR: {snr:.1f} dB (Potential high noise, Expected > {MIN_SNR_DB} dB)")
        else:
            snr = float('inf')  # Perfectly clean audio

        # Print results
        print(f"\n📄 Analyzing: {os.path.basename(file_path)}")
        if not problems:
            print("  ✅ All checks passed!")
        else:
            for p in problems:
                print(f"  ❌ {p}")
        
        return problems

    except Exception as e:
        print(f"\n❌ ERROR reading file '{os.path.basename(file_path)}': {e}")
        return [f"Could not read file: {e}"]


def main():
    print("🚀 Starting Advanced Audio Analysis...")
    all_problem_files = {}

    for folder_type in ["positive", "negative"]:
        folder_path = os.path.join(TRAINING_DATA_DIR, folder_type)
        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))

        if not wav_files:
            print(f"\n⚠️ WARNING: No .wav files found in '{folder_path}'")
            continue

        print(f"\n\n--- Checking folder: {folder_path} ---")
        for file_path in wav_files:
            problems = analyze_audio_file(file_path)
            if problems:
                all_problem_files[file_path] = problems

    # Final Report
    print("\n\n" + "="*20 + " FINAL REPORT " + "="*20)
    if not all_problem_files:
        print("🎉🎉🎉 Congratulations! All audio files passed the advanced checks.")
    else:
        print(f"🕵️ Found {len(all_problem_files)} files with potential issues.")
        print("Please review these files in Audacity. Files with 'LOW SNR' might have too much background noise.")
        for file, problems in all_problem_files.items():
            print(f"\nFile: {file}")
            for p in problems:
                print(f"  - {p}")
    print("="*54)


if __name__ == "__main__":
    main()
