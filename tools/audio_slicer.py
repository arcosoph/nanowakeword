

from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import sys

# --- Configuration (Change these as needed) ---
MIN_SILENCE_LEN = 700      # Minimum silence length (ms) to split audio (0.7 seconds)
SILENCE_THRESH = -40       # Silence threshold in dBFS (more negative = more sensitive)
KEEP_SILENCE = 100         # Extra audio to keep at the start and end of each chunk (ms)
MAX_SEGMENT_MS = 10 * 60 * 1000  # 10 minutes in milliseconds
DEFAULT_INPUT_FILE = "D:\MOIVES\Elemental (2023) Dual ORG 1080p.mkv"  # Default input file
# ---------------------------------------------

def slice_audio(input_file):
    """Split an audio file into chunks based on silence."""
    
    # Create output folder if it doesn't exist
    output_folder = "audio_slicer_output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print(f"📄 Loading input file: {input_file}")
    
    try:
        sound = AudioSegment.from_file(input_file)
        print("🎧 Audio loaded successfully.")
    except Exception as e:
        print(f"❌ Error: Could not load '{input_file}'.")
        print(f"   Reason: {e}")
        return

    # --- Check audio length ---
    if len(sound) > MAX_SEGMENT_MS:
        start_ms = (len(sound) // 2) - (MAX_SEGMENT_MS // 2)  # middle 10 min
        end_ms = start_ms + MAX_SEGMENT_MS
        print(f"⏱ Audio is longer than 10 minutes. Using middle 10 min segment ({start_ms/1000:.0f}s to {end_ms/1000:.0f}s)")
        sound = sound[start_ms:end_ms]

    print("🤫 Detecting silence and splitting audio... (This may take a moment)")
    
    # Split audio based on silence
    chunks = split_on_silence(
        sound,
        min_silence_len=MIN_SILENCE_LEN,
        silence_thresh=SILENCE_THRESH,
        keep_silence=KEEP_SILENCE
    )

    if not chunks:
        print("⚠️ No speech segments detected.")
        print("   Possible reasons: SILENCE_THRESH may be too low (try -50 or -60) or the file has no sound.")
        return

    print(f"🎉 Found {len(chunks)} speech segments! Saving chunks now...")
    
    # Save each chunk as a separate file
    for i, chunk in enumerate(chunks):
        output_filename = os.path.join(output_folder, f"arco_abc1_001{i+1:03d}.wav")
        chunk.export(output_filename)
        print(f"   ✅ Saved: {output_filename}")
        
    print(f"\n✨ Done! Your files are saved in '{output_folder}'.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = DEFAULT_INPUT_FILE
        
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found.")
        print(f"   Usage: python audio_slicer.py your_audio_file.wav")
    else:
        slice_audio(input_file)
