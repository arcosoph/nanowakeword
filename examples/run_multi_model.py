
import pyaudio
import numpy as np
import time
import os
import sys

from nanowakeword.interpreter.nanointerpreter import NanoInterpreter


# =======================
# CONFIG
# =======================
MODEL_PATHS = [
    r"model/path/your_model_1.onnx",
    r"model/path/your_model_2.onnx",
    r"model/path/your_others_model.onnx",
    # others...
]

THRESHOLD = 0.9
COOLDOWN = 1.0  # seconds


# =======================
# CHECK MODELS
# =======================
for path in MODEL_PATHS:
    if not os.path.exists(path):
        sys.exit(f"❌ Model not found: {path}")


# =======================
# LOAD INTERPRETER
# =======================
print("🔄 Loading models...")

interpreter = NanoInterpreter.load_model(MODEL_PATHS)

print("✅ Models loaded:")
for name in interpreter.models.keys():
    print(f"   - {name}")

# =======================
# AUDIO SETUP
# =======================
pa = pyaudio.PyAudio()

stream = pa.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=1280
)

print("\n🎤 Listening for wake words...\n")

last_trigger_time = 0


# =======================
# MAIN LOOP
# =======================
try:
    while True:
        audio_chunk = np.frombuffer(
            stream.read(1280, exception_on_overflow=False),
            dtype=np.int16
        )

        # Run inference for ALL models
        scores = interpreter.predict(audio_chunk)

        current_time = time.time()

        # Print live scores
        score_text = " | ".join([f"{k}:{v:.2f}" for k, v in scores.items()])
        print(f"\r{score_text}", end="")

        # Check best model
        best_model = max(scores, key=scores.get)
        best_score = scores[best_model]

        # Trigger condition
        if best_score > THRESHOLD and (current_time - last_trigger_time > COOLDOWN):
            print(f"\nDETECTED: {best_model} (Score: {best_score:.2f})")
            last_trigger_time = current_time
            interpreter.reset()

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")

finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()






### 😄💖 MORE ADVANCE

# import pyaudio
# import numpy as np
# import time
# import os
# import sys

# from nanowakeword.interpreter.nanointerpreter import NanoInterpreter


# # =======================
# # CONFIG
# # =======================
# MODEL_PATHS = [
#     r"model/path/your_model_1.onnx",
#     r"model/path/your_model_2.onnx",
#     r"model/path/your_others_model.onnx",
#     # others...
# ]

# THRESHOLD = 0.9
# COOLDOWN = 1.0


# # =======================
# # LOAD MODELS
# # =======================
# for path in MODEL_PATHS:
#     if not os.path.exists(path):
#         sys.exit(f"❌ Model not found: {path}")

# print("🔄 Loading models...")

# interpreter = NanoInterpreter.load_model(MODEL_PATHS)

# print("✅ Loaded Models:")
# for name in interpreter.models.keys():
#     print(f"   - {name}")


# # =======================
# # AUDIO SETUP
# # =======================
# pa = pyaudio.PyAudio()

# stream = pa.open(
#     format=pyaudio.paInt16,
#     channels=1,
#     rate=16000,
#     input=True,
#     frames_per_buffer=1280
# )

# print("\n🎤 Listening...\n")

# last_trigger_time = 0


# # =======================
# # MAIN LOOP
# # =======================
# try:
#     while True:
#         start_time = time.time()

#         audio_chunk = np.frombuffer(
#             stream.read(1280, exception_on_overflow=False),
#             dtype=np.int16
#         )

#         # Predict
#         scores = interpreter.predict(audio_chunk)

#         current_time = time.time()

#         # latency (prediction time)
#         latency_ms = (current_time - start_time) * 1000

#         # sort by score (high → low)
#         sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

#         # display all scores
#         display = " | ".join([f"{k}:{v:.2f}" for k, v in sorted_scores])
#         print(f"\r{display}  |  ⏱ {latency_ms:.1f}ms", end="")

#         # detect ALL above threshold
#         triggered = False

#         for name, score in sorted_scores:
#             if score > THRESHOLD:
#                 print(f"\nDETECTED: {name} (Score: {score:.2f}, ⏱ {latency_ms:.1f}ms)")
#                 triggered = True

#         # reset only once per trigger batch
#         if triggered:
#             last_trigger_time = current_time
#             interpreter.reset()

# except KeyboardInterrupt:
#     print("\nStopped")

# finally:
#     stream.stop_stream()
#     stream.close()
#     pa.terminate()

