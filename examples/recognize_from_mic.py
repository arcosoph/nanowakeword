# simple

import pyaudio
import numpy as np
import os
import sys
import time
# Import the interpreter class from the library
from nanowakeword import NanoInterpreter 

#  Simple Configuration 
MODEL_PATH = r"model/path/your.onnx"
THRESHOLD = 0.95  # A simple threshold for detection | ⚠️⚠️ This may need to be changed (eg, 0.999, 0.80) 
COOLDOWN = 1     # A simple cooldown managed outside the interpreter
# If you want, you can use more advanced methods like VAD or PATIENCE_FRAMES.

# Initialization 
if not os.path.exists(MODEL_PATH):
    sys.exit(f"Error: Model not found at '{MODEL_PATH}'")
try:
    print(" Initializing NanoInterpreter (Simple Mode)...")
    
    # Load the model with NO advanced features.
    interpreter = NanoInterpreter.load_model(MODEL_PATH)
    
    key = list(interpreter.models.keys())[0]
    print(f" Interpreter ready. Listening for '{key}'...")

    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1280)

    last_detection_time = 0
    
    # Main Loop 
    while True:
        audio_chunk = np.frombuffer(stream.read(1280, exception_on_overflow=False), dtype=np.int16)
        
        # Call predict with NO advanced parameters.
        score = interpreter.predict(audio_chunk).get(key, 0.0)

        # The detection logic is simple and external.
        current_time = time.time()
        if score > THRESHOLD and (current_time - last_detection_time > COOLDOWN):
            print(f"Detected '{key}'! (Score: {score:.5f})")
            last_detection_time = current_time
            interpreter.reset()
        else:
            print(f"Score: {score:.5f}", end='\r', flush=True)

except KeyboardInterrupt:
    print("")

