# simple

import pyaudio
import numpy as np
import os
import sys
import time
# Import the interpreter class from the library
from nanowakeword.nanointerpreter import NanoInterpreter 
#  Simple Configuration 
# MODEL_PATH = r"model/path/your.onnx"
MODEL_PATH = r"trained_models/arcosoph_A_v1/model/arcosoph_A_v1.onnx"
THRESHOLD = 0.9  # A simple threshold for detection | ⚠️⚠️ This may need to be changed (eg, 0.999, 0.80) 
COOLDOWN = 2     # A simple cooldown managed outside the interpreter
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
            print(f"Detected '{key}'! (Score: {score:.3f})")
            last_detection_time = current_time
            interpreter.reset()
        else:
            print(f"Score: {score:.3f}", end='\r', flush=True)

except KeyboardInterrupt:
    print("")



#⚠️ The commented code below is for advanced users.

# import pyaudio
# import numpy as np
# import os
# import time
# import sys
# from nanowakeword.nanointerpreter import NanoInterpreter 

# # --- 1. PROFESSIONAL CONFIGURATION ---
# # All settings are at the top for easy modification.

# MODEL_PATH = r"trained_models\hey_computer_v1\model\hey_computer_v1.onnx" # Change this to your model's path

# # --- Core Detection Parameters ---
# # The score the model's raw output must exceed to be considered.
# DETECTION_THRESHOLD = 0.8   #⚠️ This may need to be changed (eg, 0.999, 0.80) 
# # How many consecutive frames the score must stay above the threshold.
# # 3 frames = 3 * 80ms = 240ms of continuous confidence.
# PATIENCE_FRAMES = 3 # or 1, 2

# # --- Advanced Features: The "Magic" ---
# # Set to True to enable a real-time noise filter. Requires `pip install noisereduce`.
# ENABLE_NOISE_REDUCTION = False
# # Set to a value > 0 (e.g., 0.5) to only run the model when speech is detected.
# VAD_THRESHOLD = 0.2

# # --- 2. INITIALIZATION ---

# if not os.path.exists(MODEL_PATH):
#     sys.exit(f"❌ Error: Model file not found at '{MODEL_PATH}'")

# try:
#     print("🚀 Initializing NanoInterpreter with ADVANCED features...")
    
#     # Load the model with all the power-ups enabled!
#     interpreter = NanoInterpreter.load_model(
#         MODEL_PATH,
#         enable_noise_reduction=ENABLE_NOISE_REDUCTION,
#         vad_threshold=VAD_THRESHOLD
#     )
    
#     # Get the model's output key automatically
#     key = list(interpreter.models.keys())[0]
    
#     print("✅ Advanced Interpreter ready.")
#     print(f"   - Listening for: '{key}'")
#     print(f"   - Noise Reduction: {'🟢 ON' if ENABLE_NOISE_REDUCTION else '🔴 OFF'}")
#     print(f"   - VAD Filter:    {'🟢 ON (Threshold: ' + str(VAD_THRESHOLD) + ')' if VAD_THRESHOLD > 0 else '🔴 OFF'}")
#     print(f"   - Patience:        {PATIENCE_FRAMES} frames")
#     print("-" * 50)


#     pa = pyaudio.PyAudio()
#     stream = pa.open(
#         format=pyaudio.paInt16, 
#         channels=1, 
#         rate=16000, 
#         input=True, 
#         frames_per_buffer=1280
#     )

#     # --- 3. MAIN LISTENING LOOP ---
#     while True:
#         # Read a chunk of audio from the microphone
#         audio_chunk = np.frombuffer(stream.read(1280, exception_on_overflow=False), dtype=np.int16)
        
#         # Let the interpreter do all the hard work.
#         # The return value is the final, actionable decision.
#         final_decision = interpreter.predict(
#             audio_chunk,
#             patience={key: PATIENCE_FRAMES},
#             threshold={key: DETECTION_THRESHOLD}
#         )
#         final_score = final_decision.get(key, 0.0)

#         # --- The logic is now crystal clear ---
#         if final_score > 0:
#             # The interpreter has already confirmed this is a valid detection
#             # by applying VAD, threshold, and patience checks.
#             print(f"\n🎯🎯🎯 DETECTED '{key}'! (Final Score: {final_score:.2f}) 🎯🎯🎯")
#             interpreter.reset() # Reset state for the next detection
#             time.sleep(1.0) # A small cooldown for user experience
        
#         # --- Display the "thought process" using the transparent attributes ---
#         raw_score = interpreter.raw_scores.get(key, 0.0)
        
#         # Determine the status based on the difference between raw and final score
#         status = "🤫 Silence"
#         if raw_score > 0.1:
#             if final_score == 0.0:
#                 if interpreter.vad_threshold > 0 and raw_score > 0.1:
#                      status = "🤔 Analyzing (VAD Filter Active)"
#                 elif raw_score > DETECTION_THRESHOLD:
#                      status = "🤔 Analyzing (Patience Filter Active)"
#                 else:
#                      status = "👂 Listening"
#             else:
#                 status = "✅ Confident"

#         print(
#             f"[{status.ljust(32)}] "
#             f"Raw Score: {raw_score:.3f} | "
#             f"Final Score: {final_score:.3f}", 
#             end='\r', 
#             flush=True
#         )

# except KeyboardInterrupt:
#     print("\n🛑 Stopping listener...")
# except Exception as e:
#     print(f"\n❌ An unexpected error occurred: {e}")
# finally:
#     # Cleanly close the audio stream and PyAudio instance
#     if 'stream' in locals() and stream.is_active():
#         stream.stop_stream()
#         stream.close()
#     if 'pa' in locals():
#         pa.terminate()
#     print("✅ Cleanup complete. Exiting.")

