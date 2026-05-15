# NanoInterpreter Usage Guide

The interpreter's job is simple: feed it audio chunks and check if the wake word is detected. That's it.

---

## Deployment Modes at a Glance

| Mode | Edge runs | Server runs | Best for |
|---|---|---|---|
| **Fully local** | mel + embedding + model | ✗ | Common use |
| **Local cascade** | mel + embedding + gate + verifier | ✗ | Any device with a lite model |
| **Gate + remote verifier** | mel + embedding + gate | verifier only | Medium-power edge |
| **Gate + remote full pipeline** | gate only | mel + embedding + verifier | Low-power edge (Pi Zero, MCU) |
| **Fully remote** | ✗ | mel + embedding + verifier | Ultra-minimal edge |

---

## 1. Fully Local

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

## 2. Local Cascade (2-Stage)

The lite model runs every chunk. The full model only runs when the lite model fires, saving CPU.

```python
# Auto-discovers my_model_lite.onnx in the same folder
interpreter = NanoInterpreter.load_model("my_model.onnx", cascade=True)

# Or provide the lite model path explicitly
interpreter = NanoInterpreter.load_model(
    model="my_model.onnx",
    gate_model="my_model_lite.onnx",
    gate_threshold=0.25,  # Lower = fewer missed detections
)
```

---

## 3. Gate Local, Verifier Remote

The lite model runs on the edge. When it fires, pre-computed features are sent to the server.

**Start the server:**
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

## 4. Gate Local, Full Pipeline Remote

The lite model runs on the edge. When it fires, raw audio is sent to the server which handles the complete pipeline.

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

## 5. Fully Remote

The edge streams raw audio to the server. No models are downloaded on the edge.

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

## Security Configuration

Both client and server support authentication and encryption:

### Server Security

```bash
nanowakeword --model my_model.onnx --pipeline full \
    --api-key my-secret-key \
    --ssl-certfile server.crt \
    --ssl-keyfile server.key
```

Available options:
- `--api-key` - API keys for client authentication (repeat for multiple keys)
- `--enable-tokens` - Allow clients to exchange API keys for short-lived tokens
- `--token-ttl` - Token lifetime in seconds (default: 3600)
- `--rate-limit` - Max requests per IP per window (0 = disabled)
- `--rate-window` - Rate limit window in seconds (default: 60)
- `--ip-allowlist` - Allow only specific IPs/CIDR ranges
- `--ssl-certfile`/`--ssl-keyfile` - WSS/TLS certificate
- `--ssl-ca-certs` - CA bundle for mutual TLS
- `--max-connections` - Maximum simultaneous clients
- `--ban-duration` - Ban time after rate limit breach (default: 300)

### Client Security

```python
interpreter = NanoInterpreter.load_model(
    model="my_model_lite.onnx",
    remote_verifier="wss://192.168.1.100:8765",  # wss:// for TLS
    remote_pipeline="full",
    gate_threshold=0.25,
    # ༼ つ ◕_◕ ༽つ Either one will do👇
    remote_api_key="my-secret-key-xyz",           # X-API-Key header
    remote_token="my-jwt-token",                  # X-Token header (alternative)
    remote_ssl_ca_certs="ca-bundle.pem",          # Server cert verification
    remote_ssl_certfile="client.crt",             # mTLS client cert
    remote_ssl_keyfile="client.key",              # mTLS client key
)
```

---

## Optional Features (Work with Any Mode)

**VAD** ignores chunks with no speech, reducing false activations:
```python
interpreter = NanoInterpreter.load_model("my_model.onnx", vad_threshold=0.5)
```

**Noise reduction:**
```python
interpreter = NanoInterpreter.load_model("my_model.onnx", enable_noise_reduction=True)
```

**Multiple wake words:**
```python
interpreter = NanoInterpreter.load_model(["wake.onnx", "stop.onnx"])

result = interpreter.predict(audio)
if result.get("wake") > 0.95:
    print("Detected 'wake'!")
elif result.get("stop") > 0.95:
    print("Detected 'stop'!")
```

---

## Quick Reference

```python
result = interpreter.predict(audio_chunk)

result.score          # Primary model score (0.0 – 1.0)
result.gate_score     # Gate score in cascade mode, else 0.0
result.get("name")    # Score for a specific model
result["name"]        # Same, subscript style

interpreter.score           # Same as result.score
interpreter.model_name      # Name of the primary model
interpreter.gate_name       # Name of the gate model, or None
interpreter.is_cascade      # True when cascade is active
interpreter.raw_scores      # Pre-filter scores dict
interpreter.info            # Full state dict for logging

interpreter.reset()         # Clear buffers after detection
interpreter.detected(0.95)  # Boolean threshold check

# Mic loop for testing (not for production)
interpreter.listen(threshold=0.95)
interpreter.listen(threshold=0.95, blocking=False)
interpreter.stop()
```

---

## Why Keep a Local Gate Model?

If the server handles everything, why run anything locally?

- **Privacy** - Audio only leaves the device when the gate fires. Without it, every second is streamed continuously.
- **Cost** - Streaming 24/7 from many devices is expensive. The gate filters ~99% of audio before hitting the network.
- **Latency** - Network round-trip takes 50–200ms. Local gate responds in microseconds.
- **Offline resilience** - Gate works without internet. Server only confirms detections.

This is why Google Assistant and Alexa keep a tiny model on-device.