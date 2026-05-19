import pyaudio
import numpy as np
from nanowakeword import NanoInterpreter # Import the interpreter from the library

# Load model
interpreter = NanoInterpreter.load_model(
    r"model/path/your.onnx" # Your Model Path
)

# Setup microphone
pa = pyaudio.PyAudio()

stream = pa.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=1280
)

print("Listening...")

while True:
    # Read audio from mic
    audio_chunk = np.frombuffer(
        stream.read(1280, exception_on_overflow=False),
        dtype=np.int16
    )

    # Run prediction
    result = interpreter.predict(audio_chunk)

    # Detection
    if result.score > 0.95:
        print("Detected!")

        # Optional
        interpreter.reset()