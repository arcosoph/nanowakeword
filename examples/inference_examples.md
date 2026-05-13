# NanoInterpreter Usage Guide

The interpreter's only job is to tell you **detected** or **not detected**.
Feed it audio chunks, read the score, decide what to do. That's it.

---

## Deployment modes at a glance

| Mode | Edge runs | Server runs | Best for |
|---|---|---|---|
| Fully local | mel + embedding + model | ✗ | Common use |
| Local cascade | mel + embedding + gate + verifier | ✗ | Any device with a lite model |
| Gate + remote verifier | mel + embedding + gate | verifier only | Medium-power edge |
| Gate + remote full pipeline | gate only | mel + embedding + verifier | Low-power edge (Pi Zero, MCU) |
| Fully remote | nothing | mel + embedding + verifier | Ultra-minimal edge |

---

## 1. Fully local

Everything runs on the device. The simplest setup.

```python
from nanowakeword import NanoInterpreter

interpreter = NanoInterpreter.load_model("my_model.onnx")

def on_audio_frame(audio_chunk):
    result = interpreter.predict(audio_chunk)
    if result.score > 0.95:
        print("Detected!")
        interpreter.reset()
```

---

## 2. Local cascade - 2-stage, both models on device

The lite model runs every chunk. The full model only runs when the lite model fires.
Saves CPU on always-on systems. The lite model is generated automatically during training.

```python
# Auto-discovers my_model_lite.onnx in the same folder
interpreter = NanoInterpreter.load_model("my_model.onnx", cascade=True)

# Or provide the lite model path explicitly
interpreter = NanoInterpreter.load_model(
    model="my_model.onnx",
    gate_model="my_model_lite.onnx",
    gate_threshold=0.25,   # lite model sensitivity lower = fewer missed detections
)
```

---

## 3. Gate local, verifier remote

The lite model runs on the edge device. When it fires, pre-computed features are sent
to the server where the full model runs. mel + embedding still run on the edge.

**Start the server** (on any machine):
```bash
nanowakeword --model my_model.onnx
```

**Edge device:**
```python
interpreter = NanoInterpreter.load_model(
    model="my_model_lite.onnx",
    remote_verifier="ws://192.168.1.100:8765",
    gate_threshold=0.25,
)
```

---

## 4. Gate local, full pipeline remote

The lite model runs on the edge. When it fires, raw audio is sent to the server
which handles mel + embedding + full model. The edge device does minimal work.

**Start the server:**
```bash
nanowakeword --model my_model.onnx --pipeline full
```

**Edge device:**
```python
interpreter = NanoInterpreter.load_model(
    model="my_model_lite.onnx",
    remote_verifier="ws://192.168.1.100:8765",
    remote_pipeline="full",
    gate_threshold=0.25,
)
```

---

## 5. Fully remote - nothing runs on the edge

The edge device streams raw audio directly to the server.
The server handles the complete pipeline. No models downloaded on the edge at all.
Use this on ultra-low-power devices or when you want zero local inference.

**Start the server:**
```bash
nanowakeword --model my_model.onnx --pipeline full
```

**Edge device:**
```python
interpreter = NanoInterpreter.load_model(
    remote_verifier="ws://192.168.1.100:8765",
    remote_pipeline="full",
)

def on_audio_frame(audio_chunk):
    result = interpreter.predict(audio_chunk)
    if result.score > 0.95:
        print("Detected!")
        interpreter.reset()
```

---

## Optional features (work with any mode)

**VAD** ignore chunks with no speech, reduces false activations in noisy environments:
```python
interpreter = NanoInterpreter.load_model("my_model.onnx", vad_threshold=0.5)
```

**Noise reduction:**
```python
interpreter = NanoInterpreter.load_model("my_model.onnx", enable_noise_reduction=True)
```

**Multiple wake words** load several models, each gets its own score:
```python
interpreter = NanoInterpreter.load_model(["wake.onnx", "stop.onnx"])

result = interpreter.predict(audio)
if result.get("wake") > 0.95:
    print("Detected!")
elif result.get("stop") > 0.95:
    print("Detected Stop!")
```

---

## Quick reference

```python
result = interpreter.predict(audio_chunk)

result.score          # primary model score (0.0 – 1.0)
result.gate_score     # gate score in cascade mode, else 0.0
result.get("name")    # score for a specific model by name
result["name"]        # same, subscript style

interpreter.score           # same as result.score, persists between calls
interpreter.model_name      # name of the primary model
interpreter.gate_name       # name of the gate model, or None
interpreter.is_cascade      # True when cascade is active
interpreter.raw_scores      # pre-filter scores dict
interpreter.info            # full state dict for logging/debugging
interpreter.reset()         # clear buffers after a detection
interpreter.detected(0.95)  # boolean threshold check

# Built-in mic loop for testing and demos only
interpreter.listen(on_detection=lambda name, score: print("Detected!"), threshold=0.95)
interpreter.listen(threshold=0.95, blocking=False)  # non-blocking
interpreter.stop()                                   # stop non-blocking loop
```

---

## Why keep a local gate model?

If the server handles everything, why run anything locally?

- **Privacy** audio only leaves the device when the gate fires. Without a local gate,
  every second of audio is streamed to the server continuously.
- **Cost** streaming 24/7 from many devices is expensive. The gate filters ~99% of audio
  before it hits the network.
- **Latency** a network round-trip takes 50–200ms. The local gate responds in microseconds.
- **Offline resilience** the gate works without internet. The server is only needed for
  final confirmation.

This is the same reason Google Assistant and Alexa keep a tiny model on the device.
