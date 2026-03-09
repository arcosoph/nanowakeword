import os
from pydub import AudioSegment

# ====== CONFIG ======
INPUT_FOLDER = "clips"
OUTPUT_FOLDER = "final_clips"
LOG_FILE = "process_log.txt"
TARGET_DURATION_MS = 5000  # 5 seconds
# ====================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

file_counter = 1
total_created_files = 0

log_lines = []

def get_new_filename(counter):
    return f"nanowakeword_bg_noice_{counter:02d}.wav"

wav_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".wav")]

for wav_file in wav_files:
    file_path = os.path.join(INPUT_FOLDER, wav_file)
    audio = AudioSegment.from_wav(file_path)

    duration_ms = len(audio)

    if duration_ms < TARGET_DURATION_MS:
        log_lines.append(f"SKIPPED (too short): {wav_file} ({duration_ms/1000:.2f}s)")
        continue

    num_segments = duration_ms // TARGET_DURATION_MS

    for i in range(num_segments):
        start = i * TARGET_DURATION_MS
        end = start + TARGET_DURATION_MS
        segment = audio[start:end]

        new_filename = get_new_filename(file_counter)
        output_path = os.path.join(OUTPUT_FOLDER, new_filename)

        segment.export(output_path, format="wav")

        log_lines.append(
            f"CREATED: {new_filename} from {wav_file} "
            f"(segment {i+1}/{num_segments})"
        )

        file_counter += 1
        total_created_files += 1

# Save log file
with open(LOG_FILE, "w") as log_file:
    for line in log_lines:
        log_file.write(line + "\n")

print("====================================")
print(f"Total output files created: {total_created_files}")
print("All files saved in:", OUTPUT_FOLDER)
print("Log saved in:", LOG_FILE)
print("====================================")