# audio_investigator.py
# -----------------------------------------
# Description:
# This script automatically investigates all 'positive' audio files in the training dataset.
# It checks each file for:
# 1. Correct sample rate and number of channels (Mono)
# 2. File duration (too short or too long)
# 3. Maximum volume (too quiet)
# 4. Excessive leading or trailing silence
# Any problematic files are moved to a quarantine folder for manual review or correction.
# -----------------------------------------

import os
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

# --- Control Panel ---
POSITIVE_DATA_FOLDER = os.path.join("training_data", "positive")
QUARANTINE_FOLDER = os.path.join("training_data", "_quarantine")
MAX_LEADING_SILENCE_MS = 500   # 0.5 seconds
MAX_TRAILING_SILENCE_MS = 500  # 0.5 seconds
MIN_LOUDNESS_DBFS = -35.0      # Minimum acceptable volume (-dBFS)
MIN_DURATION_MS = 500           # 0.5 seconds
MAX_DURATION_MS = 4000          # 4 seconds
REQUIRED_SAMPLE_RATE = 16000
REQUIRED_CHANNELS = 1           # Mono
# -----------------------------------------

def investigate_audio_files():
    """
    Investigates audio files and moves problematic files to the quarantine folder.
    """
    if not os.path.exists(POSITIVE_DATA_FOLDER):
        print(f"ERROR: '{POSITIVE_DATA_FOLDER}' folder not found.")
        return

    if not os.path.exists(QUARANTINE_FOLDER):
        print(f"INFO: Creating '{QUARANTINE_FOLDER}' folder for problematic files.")
        os.makedirs(QUARANTINE_FOLDER)

    print("\n🕵️ Starting audio investigation...")
    print("=========================================")

    total_files = 0
    problematic_files = 0

    for filename in os.listdir(POSITIVE_DATA_FOLDER):
        if not filename.lower().endswith(".wav"):
            continue

        total_files += 1
        filepath = os.path.join(POSITIVE_DATA_FOLDER, filename)
        issues_found = []

        try:
            audio = AudioSegment.from_wav(filepath)

            # --- Checks ---
            # 1. Format (Sample Rate & Channels)
            if audio.frame_rate != REQUIRED_SAMPLE_RATE:
                issues_found.append(f"Incorrect sample rate ({audio.frame_rate}Hz, required {REQUIRED_SAMPLE_RATE}Hz)")
            if audio.channels != REQUIRED_CHANNELS:
                issues_found.append(f"Incorrect channels ({audio.channels}, required {REQUIRED_CHANNELS} [Mono])")

            # 2. Duration
            if len(audio) < MIN_DURATION_MS:
                issues_found.append(f"Abnormally short file ({len(audio)}ms, minimum {MIN_DURATION_MS}ms)")
            if len(audio) > MAX_DURATION_MS:
                issues_found.append(f"Abnormally long file ({len(audio)}ms, maximum {MAX_DURATION_MS}ms)")

            # 3. Low volume
            if audio.max_dBFS < MIN_LOUDNESS_DBFS:
                issues_found.append(f"Very low volume (Max: {audio.max_dBFS:.2f} dBFS, minimum required {MIN_LOUDNESS_DBFS:.2f} dBFS)")

            # 4. Excessive silence
            leading_silence = detect_leading_silence(audio, silence_threshold=-40.0)
            if leading_silence > MAX_LEADING_SILENCE_MS:
                issues_found.append(f"Excessive leading silence ({leading_silence}ms, maximum {MAX_LEADING_SILENCE_MS}ms)")

            trailing_silence = detect_leading_silence(audio.reverse(), silence_threshold=-40.0)
            if trailing_silence > MAX_TRAILING_SILENCE_MS:
                issues_found.append(f"Excessive trailing silence ({trailing_silence}ms, maximum {MAX_TRAILING_SILENCE_MS}ms)")

            # --- Decision ---
            if issues_found:
                problematic_files += 1
                print(f"\n🚨 Problematic file detected: {filename}")
                for issue in issues_found:
                    print(f"   - Reason: {issue}")
                
                # Move file to quarantine folder
                destination_path = os.path.join(QUARANTINE_FOLDER, filename)
                os.rename(filepath, destination_path)
                print(f"   -> Moved to '{QUARANTINE_FOLDER}'.")

        except Exception as e:
            problematic_files += 1
            print(f"\n🚨 Could not process file: {filename}")
            print(f"   - Reason: {e}")
            destination_path = os.path.join(QUARANTINE_FOLDER, filename)
            os.rename(filepath, destination_path)
            print(f"   -> Moved to '{QUARANTINE_FOLDER}'.")

    print("\n=========================================")
    print("✅ Investigation complete!")
    print(f"Total files checked: {total_files}")
    print(f"Problematic files found: {problematic_files}")
    if problematic_files > 0:
        print(f"All problematic files have been moved to '{QUARANTINE_FOLDER}'.")
        print("You can now review or fix them.")
    else:
        print("🎉 No issues detected in your dataset!")

if __name__ == "__main__":
    # User warning
    print("!!! WARNING !!!")
    print(f"This script will investigate your '{POSITIVE_DATA_FOLDER}' folder.")
    print(f"Any problematic files will be moved to '{QUARANTINE_FOLDER}'.")
    print("It is recommended to keep a backup of your data before proceeding.")

    # Ask for user confirmation
    choice = input("\nDo you want to start the investigation? (yes/no): ").lower()
    if choice == 'yes':
        investigate_audio_files()
    else:
        print("Investigation cancelled.")
