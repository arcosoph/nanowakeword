"""
To run this code, type "python -m tests.nww_model_test" in the terminal.

⚠️ IMPORTANT: The value of DETECTION_THRESHOLD is specific to your trained model.
You must experiment to find the right sensitivity — try different values 
like 0.9, 0.5, or 0.05 until detection feels accurate and consistent for your environment.

"""

# Example: Minimal NanoWakeWord Listener
from nanowakeword.model import Model
import pyaudio, numpy as np, os, sys, time

# MODEL_PATH = "nanowakeword-lstm-base.onnx"
MODEL_PATH = r"trained_models\hey_computer_v1.onnx"
THRESHOLD = 0.96
COOLDOWN = 2

if not os.path.exists(MODEL_PATH):
    sys.exit(f"Model not found: {MODEL_PATH}")

model = Model(wakeword_models=[MODEL_PATH])
key = list(model.models.keys())[0]

pa = pyaudio.PyAudio()
stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1280)

last_detect = 0
try:
    while True:
        data = np.frombuffer(stream.read(1280, exception_on_overflow=False), dtype=np.int16)
        score = model.predict(data).get(key, 0)
        t = time.time()
        if score > THRESHOLD and t - last_detect > COOLDOWN:
            print(f"🎯 Detected '{key}'! Score: {score:.2f}")
            last_detect = t
        else:
            print(f"[{key}] Score: {score:.3f}", end='\r')
except KeyboardInterrupt:
    pass
