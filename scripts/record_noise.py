# record_noise.py
# -----------------------------------------
# Description:
# This script records 5 samples of your room's normal background noise.
# These recordings are intended as "negative samples" for training a wake-word detection model.
# Please stay completely silent during each recording to capture only ambient noise.
# -----------------------------------------

import soundfile as sf
import sounddevice as sd
import os
import time

# --- Configuration ---
SAVE_DIR = os.path.join("training_data", "negative")
SAMPLE_RATE = 16000
DURATION = 2  # seconds

print("We will now record 5 samples of your room's normal background noise.")
print("Please remain completely silent during each recording.")

for i in range(5):
    input(f"\n🔴 Press Enter to start recording {i+1}/5...")
    print("🎧 Recording in progress...")
    
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    
    filename = f"real2_background_noise_{i+1:02d}.wav"
    sf.write(os.path.join(SAVE_DIR, filename), audio, SAMPLE_RATE)
    
    print(f"✅ Saved: {filename}")

print("\n🎉🎉🎉 Background noise recording complete!")
