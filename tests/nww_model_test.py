"""
To run this code, type "python -m tests.nww_model_test" in the terminal.

⚠️ IMPORTANT: The value of DETECTION_THRESHOLD is specific to your trained model.
You must experiment to find the right sensitivity — try different values 
like 0.9, 0.5, or 0.05 until detection feels accurate and consistent for your environment.

"""

# import nanowakeword
from nanowakeword.model import Model
import pyaudio
import numpy as np
import os
import sys
import time

# --- Configuration ---
MODEL_PATH = os.path.join("trained_models","my_manual_model_v1.onnx")

DETECTION_THRESHOLD = 0.09  # ⚠️ IMPORTANT: The value of DETECTION_THRESHOLD is specific to your trained model.
                    # You must experiment to find the right sensitivity — try different values like 0.9, 0.5, or 0.05
                    # until detection feels accurate and consistent for your environment.

COOLDOWN_SECONDS = 2    

# --- Main Program ---
nwwModel = None
stream = None
pa = None
last_detection_time = 0

print(f"Checking your custom model path: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"Fatal error: Model file '{MODEL_PATH}' not found.")
    sys.exit(1)

try:
    print("Model file found. Loading now...")
    
    # --- Main change here ---
    nwwModel = Model(wakeword_models=[MODEL_PATH])  
    # -----------------------

    model_key = list(nwwModel.models.keys())[0]
    
    print("Starting microphone...")
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1280)

    print(f"System ready! Listening for '{model_key}'...")
    print(f"(Detection threshold set to {DETECTION_THRESHOLD})")
    print(f"(Cooldown period {COOLDOWN_SECONDS} seconds)")
    print("(Press Ctrl+C to exit)")

    while True:
        data = np.frombuffer(stream.read(1280, exception_on_overflow=False), dtype=np.int16)
        prediction = nwwModel.predict(data)
        score = prediction.get(model_key, 0)
        
        current_time = time.time()
        
        if score > DETECTION_THRESHOLD and (current_time - last_detection_time) > COOLDOWN_SECONDS:
            print(f"Wake word '{model_key}' detected! (Score: {score:.2f})")
            last_detection_time = current_time
        else:
            print(f"[{model_key}] Score: {score:.3f}", end='\r')

except (KeyboardInterrupt, SystemExit):
    print("Program is being stopped by the user...")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    if stream and stream.is_active():
        print("🎙️ Closing microphone stream...")
        stream.stop_stream()
        stream.close()
    if pa:
        print("🎧 Releasing PyAudio resources...")
        pa.terminate()
    print("Goodbye!")

