# ==============================================================================
#  NanoWakeWord: Lightweight, Intelligent Wake Word Detection
#  Copyright 2025 Arcosoph. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Project: https://github.com/arcosoph/nanowakeword
# ==============================================================================

"""
RemoteVerifier - WebSocket server that hosts wake word model inference remotely.

Supports three pipeline modes so users can choose exactly what runs where:

  "verifier_only"  (default)
      Client sends pre-computed feature tensor [1, T, 96].
      Server runs only the wake word model.
      mel + embedding run on the edge device.
      Lowest server load, lowest bandwidth.

  "embedding"
      Client sends raw mel-spectrogram frames.
      Server runs embedding_model + wake word model.
      mel runs on the edge device.

  "full"
      Client sends raw int16 PCM audio.
      Server runs mel + embedding + wake word model.
      Edge device runs nothing except the gate (if any).
      Best for very low-power edge devices.

Usage - start the server
------------------------
    # Default (verifier_only - expects pre-computed features from edge)
    python -m nanowakeword.interpreter.remote_verifier --model my_model.onnx

    # Full pipeline - edge just streams raw audio
    python -m nanowakeword.interpreter.remote_verifier \\
        --model my_model.onnx --pipeline full

    # From Python
    from nanowakeword.interpreter.remote_verifier import serve
    serve("my_model.onnx", pipeline="full", host="0.0.0.0", port=8765)

    # Secure server example
    from nanowakeword.interpreter import build_security
    security = build_security(
        api_keys=["my-secret-key"],
        ssl_certfile="server.crt",
        ssl_keyfile="server.key",
    )
    serve("my_model.onnx", security=security)

Usage - edge device
-------------------
    from nanowakeword import NanoInterpreter

    # Gate local, verifier remote (default - verifier_only)
    interpreter = NanoInterpreter.load_model(
        model="my_model_lite.onnx",
        remote_verifier="ws://192.168.1.100:8765",
        gate_threshold=0.25,
    )

    # Gate local, full pipeline remote (edge sends raw audio)
    interpreter = NanoInterpreter.load_model(
        model="my_model_lite.onnx",
        remote_verifier="ws://192.168.1.100:8765",
        remote_pipeline="full",
        gate_threshold=0.25,
    )

    # No local model at all - everything on server
    interpreter = NanoInterpreter.load_model(
        remote_verifier="ws://192.168.1.100:8765",
        remote_pipeline="full",
    )

Wire protocol
-------------
Each message starts with a 1-byte pipeline tag:

    0x01  verifier_only  - float32 feature tensor
                           header: 3 × int32 (batch, time, feat)
                           body:   batch*time*feat × float32

    0x02  embedding      - float32 mel-spectrogram frames
                           header: 3 × int32 (batch, frames, mel_bins)
                           body:   batch*frames*mel_bins × float32

    0x03  full           - int16 PCM audio
                           header: 1 × int32 (n_samples)
                           body:   n_samples × int16

Response (all modes): JSON  {"score": <float>}
"""

from __future__ import annotations

import json
import struct
import logging
import argparse
import numpy as np
from typing import Optional, Union

from nanowakeword.interpreter.server_security import (
    SecurityConfig,
    SecurityManager,
    build_security,
    is_token_request,
    decode_token_request,
    encode_token_response,
    encode_error_response,
)

logger = logging.getLogger(__name__)

# Pipeline mode constants - shared between server and client
PIPELINE_VERIFIER_ONLY = "verifier_only"
PIPELINE_EMBEDDING      = "embedding"
PIPELINE_FULL           = "full"

_VALID_PIPELINES = {PIPELINE_VERIFIER_ONLY, PIPELINE_EMBEDDING, PIPELINE_FULL}

# Wire protocol tags
_TAG_FEATURES   = 0x01   # pre-computed feature tensor
_TAG_MEL        = 0x02   # mel-spectrogram frames
_TAG_AUDIO      = 0x03   # raw PCM audio


# Encoding helpers (used by _RemoteSession on the client side)

def encode_features(features: np.ndarray) -> bytes:
    """Encode a float32 feature tensor [B, T, F] to wire format."""
    b, t, f = features.shape
    header  = struct.pack("<Biii", _TAG_FEATURES, b, t, f)
    return header + features.astype(np.float32).tobytes()


def encode_audio(audio: np.ndarray) -> bytes:
    """Encode int16 PCM audio to wire format."""
    n      = len(audio)
    header = struct.pack("<Bi", _TAG_AUDIO, n)
    return header + audio.astype(np.int16).tobytes()


# Server

def serve(
    model_path: str,
    pipeline: str = PIPELINE_VERIFIER_ONLY,
    host: str = "0.0.0.0",
    port: int = 8765,
    log_level: str = "INFO",
    security: Optional[Union[SecurityConfig, SecurityManager]] = None,
) -> None:
    """
    Start the RemoteVerifier WebSocket server.

    Blocks until interrupted (Ctrl+C).

    Args:
        model_path:  Path to the wake word .onnx model to host.
        pipeline:    What the server handles:
                       "verifier_only" - expects pre-computed features (default)
                       "embedding"     - expects mel frames, runs embedding+verifier
                       "full"          - expects raw audio, runs full pipeline
        host:        Bind address. "0.0.0.0" accepts all connections.
        port:        Port number. Default 8765.
        log_level:   Logging verbosity.
        security:    Optional SecurityConfig or SecurityManager for API key,
                     token, TLS, rate limiting, and allowlist support.
    """
    if pipeline not in _VALID_PIPELINES:
        raise ValueError(
            f"Invalid pipeline '{pipeline}'. "
            f"Choose from: {sorted(_VALID_PIPELINES)}"
        )

    security_manager: Optional[SecurityManager] = None
    if security is not None:
        if isinstance(security, SecurityConfig):
            security_manager = SecurityManager(security)
        elif isinstance(security, SecurityManager):
            security_manager = security
        else:
            raise TypeError(
                "security must be a SecurityConfig or SecurityManager instance"
            )

    try:
        import asyncio
        import websockets
    except ImportError:
        raise ImportError(
            "websockets is required for RemoteVerifier. "
            "Install it with: pip install websockets"
        )

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime is required. pip install onnxruntime")

    # Load wake word model
    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 2
    sess_opts.intra_op_num_threads = 2
    ww_session = ort.InferenceSession(
        model_path,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    ww_inputs    = ww_session.get_inputs()
    ww_input_name = ww_inputs[0].name
    n_frames      = ww_inputs[0].shape[1]
    model_name    = model_path.replace("\\", "/").split("/")[-1].replace(".onnx", "")

    logger.info(f"Wake word model: '{model_name}'  input={ww_inputs[0].shape}")
    logger.info(f"Pipeline mode:   '{pipeline}'")
    if security_manager is not None:
        logger.info(f"Security:        {security_manager.config.summary()}")

    # Load mel + embedding models only when the server needs them 
    # This is the key point: mel/embedding are only downloaded/loaded on the
    # server when pipeline != "verifier_only". On the edge device, AudioFeatures
    # handles them - but only when the edge device actually needs them too.
    mel_session       = None
    embedding_session = None

    if pipeline in (PIPELINE_EMBEDDING, PIPELINE_FULL):
        # Import AudioFeatures just to resolve model paths via the registry 
        # but we create our own ONNX sessions here for async-safe usage.
        from nanowakeword.interpreter.models import models as model_registry

        mel_path       = model_registry.melspectrogram_onnx
        embedding_path = model_registry.embedding_model_onnx

        mel_session = ort.InferenceSession(
            mel_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        embedding_session = ort.InferenceSession(
            embedding_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        logger.info("Mel-spectrogram and embedding models loaded on server.")

    # Per-connection state factory
    # For "full" pipeline, each client connection needs its own streaming state
    # (mel buffer, raw audio buffer). We use a lightweight stateful object
    # instead of AudioFeatures to avoid re-downloading models per connection.

    class _StreamingState:
        """Minimal streaming state for one client connection."""
        def __init__(self):
            from collections import deque
            self.raw_buffer        = deque(maxlen=16000 * 10)
            self.mel_buffer        = np.ones((76, 32), dtype=np.float32)
            self.mel_max_len       = 10 * 97
            self.accumulated       = 0
            self.remainder         = np.empty(0)
            self.feature_buffer    = self._warm_up()

        def _warm_up(self):
            """Pre-fill feature buffer with random data (same as AudioFeatures)."""
            dummy = np.random.randint(-1000, 1000, 16000 * 4).astype(np.int16)
            return self._get_embeddings(dummy)

        def _mel(self, audio: np.ndarray) -> np.ndarray:
            x = audio.astype(np.float32)[None, :]
            out = mel_session.run(None, {"input": x})
            spec = np.squeeze(out[0])
            return spec / 10 + 2  # same transform as AudioFeatures

        def _embed(self, mel_window: np.ndarray) -> np.ndarray:
            batch = mel_window[None, :, :, None].astype(np.float32)
            return embedding_session.run(None, {"input_1": batch})[0].squeeze()

        def _get_embeddings(self, audio: np.ndarray) -> np.ndarray:
            spec    = self._mel(audio)
            windows = []
            for i in range(0, spec.shape[0], 8):
                w = spec[i: i + 76]
                if w.shape[0] == 76:
                    windows.append(w)
            if not windows:
                return np.zeros((1, 96), dtype=np.float32)
            batch = np.array(windows)[:, :, :, None].astype(np.float32)
            return embedding_session.run(None, {"input_1": batch})[0].squeeze(axis=-1) \
                if batch.ndim == 4 else np.zeros((1, 96), dtype=np.float32)

        def process(self, audio: np.ndarray) -> Optional[np.ndarray]:
            """
            Feed an audio chunk. Returns feature tensor [1, n_frames, 96]
            when enough data is ready, else None.
            """
            if self.remainder.shape[0]:
                audio = np.concatenate([self.remainder, audio])
                self.remainder = np.empty(0)

            if self.accumulated + audio.shape[0] >= 1280:
                rem = (self.accumulated + audio.shape[0]) % 1280
                if rem:
                    self.raw_buffer.extend(audio[:-rem].tolist())
                    self.accumulated += len(audio) - rem
                    self.remainder = audio[-rem:]
                else:
                    self.raw_buffer.extend(audio.tolist())
                    self.accumulated += audio.shape[0]
            else:
                self.accumulated += audio.shape[0]
                self.raw_buffer.extend(audio.tolist())
                return None

            if self.accumulated >= 1280 and self.accumulated % 1280 == 0:
                n = self.accumulated
                chunk = np.array(list(self.raw_buffer)[-n - 160 * 3:], dtype=np.int16)
                new_mel = self._mel(chunk)
                self.mel_buffer = np.vstack([self.mel_buffer, new_mel])
                if self.mel_buffer.shape[0] > self.mel_max_len:
                    self.mel_buffer = self.mel_buffer[-self.mel_max_len:]

                for i in np.arange(n // 1280 - 1, -1, -1):
                    ndx = -8 * i
                    ndx = ndx if ndx != 0 else len(self.mel_buffer)
                    w = self.mel_buffer[-76 + ndx: ndx].astype(np.float32)[None, :, :, None]
                    if w.shape[1] == 76:
                        emb = embedding_session.run(None, {"input_1": w})[0].squeeze()
                        self.feature_buffer = np.vstack([self.feature_buffer, emb])

                self.accumulated = 0

                if self.feature_buffer.shape[0] > 120:
                    self.feature_buffer = self.feature_buffer[-120:]

                if self.feature_buffer.shape[0] >= n_frames:
                    return self.feature_buffer[-n_frames:][None].astype(np.float32)

            return None

    # WebSocket handler 

    async def handle_client(websocket):
        client_addr = websocket.remote_address
        client_ip = client_addr[0] if isinstance(client_addr, tuple) else str(client_addr)
        logger.info(f"Client connected: {client_addr}  pipeline='{pipeline}'")

        # Per-connection streaming state (only needed for full pipeline)
        state = _StreamingState() if pipeline == PIPELINE_FULL else None
        connected = False

        try:
            if security_manager is not None:
                allowed, reason = security_manager.check_handshake(websocket)
                if not allowed:
                    logger.warning(f"Rejected connection from {client_ip}: {reason}")
                    await websocket.close(code=1008, reason=reason)
                    return
                security_manager.on_connect()
                connected = True

            async for message in websocket:
                if not isinstance(message, bytes) or len(message) < 1:
                    continue

                if security_manager is not None and not security_manager.record_request(client_ip):
                    await websocket.close(code=1008, reason="rate limit exceeded")
                    return

                if security_manager is not None and security_manager.config.enable_tokens and is_token_request(message):
                    api_key = decode_token_request(message)
                    if security_manager.verify_api_key(api_key):
                        token = security_manager.issue_token()
                        await websocket.send(encode_token_response(token))
                    else:
                        await websocket.send(encode_error_response("invalid API key"))
                        await websocket.close(code=1008, reason="invalid API key")
                    continue

                tag   = message[0]
                score = 0.0

                # verifier_only: receive pre-computed features
                if tag == _TAG_FEATURES:
                    b, t, f = struct.unpack("<iii", message[1:13])
                    n_bytes  = b * t * f * 4
                    features = np.frombuffer(
                        message[13: 13 + n_bytes], dtype=np.float32
                    ).reshape(b, t, f)
                    out   = ww_session.run(None, {ww_input_name: features})
                    score = float(out[0].item())

                # full: receive raw audio, run full pipeline
                elif tag == _TAG_AUDIO and pipeline == PIPELINE_FULL:
                    n_samples = struct.unpack("<i", message[1:5])[0]
                    audio     = np.frombuffer(
                        message[5: 5 + n_samples * 2], dtype=np.int16
                    )
                    features = state.process(audio)
                    if features is not None:
                        out   = ww_session.run(None, {ww_input_name: features})
                        score = float(out[0].item())

                await websocket.send(json.dumps({"score": score}))

        except Exception as e:
            logger.warning(f"Client {client_addr} error: {e}")
        finally:
            if connected and security_manager is not None:
                security_manager.on_disconnect()
            logger.info(f"Client disconnected: {client_addr}")

    # Run server

    import asyncio

    async def _main():
        async with websockets.serve(
            handle_client,
            host,
            port,
            ssl=security_manager.ssl_context if security_manager else None,
        ):
            logger.info(f"RemoteVerifier ready on ws://{host}:{port}")
            await asyncio.Future()

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        logger.info("RemoteVerifier stopped.")


# _RemoteSession
# Drop-in replacement for an onnxruntime.InferenceSession.
# Used internally by NanoInterpreter when remote_verifier is set.
# The interpreter's predict() loop calls session.run() - this class intercepts
# that call and sends data over WebSocket instead.

class _RemoteSession:
    """
    Mimics the onnxruntime.InferenceSession interface but sends data to a
    remote WebSocket server instead of running inference locally.

    Internal class - users never instantiate this directly.
    NanoInterpreter.load_model() creates it when remote_verifier is set.
    """

    def __init__(
        self,
        uri: str,
        model_name: str,
        pipeline: str = PIPELINE_VERIFIER_ONLY,
        n_frames: int = 16,
        timeout: float = 2.0,
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        ssl_certfile: Optional[str] = None,
        ssl_keyfile: Optional[str] = None,
        ssl_ca_certs: Optional[str] = None,
    ):
        """
        Args:
            uri:        WebSocket URI, e.g. "ws://192.168.1.100:8765"
            model_name: Name used for logging.
            pipeline:   Must match the server's pipeline mode.
            n_frames:   Expected feature frames (used for get_inputs() shape).
            timeout:    Seconds to wait for a response before returning 0.0.
            api_key:    Optional API key to send as ``X-API-Key`` during the
                        websocket handshake.
            token:      Optional token to send as ``X-Token`` during the
                        websocket handshake.
            ssl_certfile: Optional client certificate bundle for mutual TLS.
            ssl_keyfile: Optional private key file for client certificate auth.
            ssl_ca_certs: Optional CA bundle path for WSS/TLS server verification.
        """
        try:
            import websockets  # noqa: F401
        except ImportError:
            raise ImportError(
                "websockets is required for remote_verifier. "
                "Install it with: pip install websockets"
            )

        if pipeline not in _VALID_PIPELINES:
            raise ValueError(f"Invalid pipeline '{pipeline}'.")

        import asyncio
        import threading

        self.uri         = uri
        self.model_name  = model_name
        self.pipeline    = pipeline
        self.n_frames    = n_frames
        self.timeout     = timeout
        self.api_key     = api_key
        self.token       = token
        self.ssl_certfile = ssl_certfile
        self.ssl_keyfile  = ssl_keyfile
        self.ssl_ca_certs = ssl_ca_certs
        self._loop       = asyncio.new_event_loop()
        self._ws         = None
        self._lock       = threading.Lock()

        self._connect()
        logger.info(f"[Nanowakeword] Connected to {uri}  pipeline='{pipeline}'")

    def _connect(self):
        import ssl
        import websockets

        async def _do():
            headers = None
            if self.token:
                headers = {"X-Token": self.token}
            elif self.api_key:
                headers = {"X-API-Key": self.api_key}

            ssl_ctx = None
            if self.uri.lower().startswith("wss://") or self.ssl_certfile or self.ssl_keyfile or self.ssl_ca_certs:
                ssl_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                if self.ssl_ca_certs:
                    ssl_ctx.load_verify_locations(cafile=self.ssl_ca_certs)
                if self.ssl_certfile:
                    ssl_ctx.load_cert_chain(certfile=self.ssl_certfile, keyfile=self.ssl_keyfile)

            return await websockets.connect(
                self.uri,
                ssl=ssl_ctx,
                additional_headers=headers,
            )

        self._ws = self._loop.run_until_complete(_do())

    def _reconnect(self):
        try:
            self._connect()
            logger.info(f"[Nanowakeword] Reconnected to {self.uri}")
        except Exception as e:
            logger.warning(f"[Nanowakeword] Reconnect failed: {e}")
            self._ws = None

    # onnxruntime.InferenceSession interface

    def get_inputs(self):
        """Fake input descriptor for NanoInterpreter bookkeeping."""
        class _FakeInput:
            def __init__(self, name, shape):
                self.name  = name
                self.shape = shape
        return [_FakeInput("input", ["batch_size", self.n_frames, 96])]

    def run(self, output_names, input_feed, run_options=None):
        """
        Sends data to the remote server and returns the score.
        What is sent depends on pipeline mode:
          verifier_only → pre-computed feature tensor (from input_feed["input"])
          full          → raw audio (from input_feed["audio"])
        """
        import asyncio

        # Build the wire message based on pipeline
        if self.pipeline == PIPELINE_FULL:
            audio   = input_feed.get("audio")
            if audio is None:
                return [np.array([[[0.0]]], dtype=np.float32)]
            message = encode_audio(audio)
        else:
            # verifier_only (default) - send pre-computed features
            features = input_feed["input"]
            message  = encode_features(features)

        async def _send_recv():
            try:
                await self._ws.send(message)
                response = await asyncio.wait_for(
                    self._ws.recv(), timeout=self.timeout
                )
                return float(json.loads(response).get("score", 0.0))
            except Exception as e:
                logger.warning(f"[Nanowakeword] Communication error: {e}")
                return None

        with self._lock:
            if self._ws is None:
                self._reconnect()
            if self._ws is None:
                score = 0.0
            else:
                score = self._loop.run_until_complete(_send_recv())
                if score is None:
                    self._reconnect()
                    score = 0.0

        return [np.array([[[score]]], dtype=np.float32)]

    def close(self):
        if self._ws is not None:
            try:
                self._loop.run_until_complete(self._ws.close())
            except Exception:
                pass
        try:
            self._loop.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# CLI entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NanoWakeWord RemoteVerifier - WebSocket inference server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to the wake word .onnx model file"
    )
    parser.add_argument(
        "--pipeline", default=PIPELINE_VERIFIER_ONLY,
        choices=sorted(_VALID_PIPELINES),
        help=(
            "verifier_only: expects pre-computed features from edge (lowest server load)\n"
            "embedding:     expects mel frames, runs embedding+verifier on server\n"
            "full:          expects raw audio, runs complete pipeline on server"
        ),
    )
    parser.add_argument("--host",  default="0.0.0.0", help="Bind address")
    parser.add_argument("--port",  default=8765, type=int, help="Port number")
    parser.add_argument("--log",   default="INFO", help="Log level")
    parser.add_argument(
        "--api-key",
        dest="api_keys",
        action="append",
        default=[],
        help="Add an API key for client authentication. Repeat to add multiple keys.",
    )
    parser.add_argument(
        "--enable-tokens",
        action="store_true",
        help="Allow clients to exchange an API key for a short-lived access token.",
    )
    parser.add_argument(
        "--token-ttl",
        type=int,
        default=3600,
        help="Token lifetime in seconds. Default: 3600.",
    )
    parser.add_argument(
        "--token-secret",
        default=None,
        help="Secret used to sign tokens. Auto-generated if omitted.",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Maximum messages per rate-window per IP. 0 disables rate limiting.",
    )
    parser.add_argument(
        "--rate-window",
        type=int,
        default=60,
        help="Rate-limit sliding window in seconds. Default: 60.",
    )
    parser.add_argument(
        "--ip-allowlist",
        action="append",
        default=[],
        help="Allow only connections from this IP or CIDR. Repeat for multiple entries.",
    )
    parser.add_argument(
        "--ssl-certfile",
        default=None,
        help="Path to PEM certificate file for WSS/TLS.",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=None,
        help="Path to PEM private key file for WSS/TLS.",
    )
    parser.add_argument(
        "--ssl-ca-certs",
        default=None,
        help="Optional CA bundle path for mutual TLS.",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=0,
        help="Maximum number of simultaneous client connections. 0 = unlimited.",
    )
    parser.add_argument(
        "--ban-duration",
        type=int,
        default=300,
        help="Ban duration in seconds after rate-limit breach. 0 = no ban.",
    )
    args = parser.parse_args()

    from nanowakeword.interpreter import build_security

    security = build_security(
        api_keys=args.api_keys,
        enable_tokens=args.enable_tokens,
        token_ttl=args.token_ttl,
        token_secret=args.token_secret,
        rate_limit=args.rate_limit,
        rate_window=args.rate_window,
        ip_allowlist=args.ip_allowlist,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
        ssl_ca_certs=args.ssl_ca_certs,
        max_connections=args.max_connections,
        ban_duration=args.ban_duration,
    )

    serve(
        model_path=args.model,
        pipeline=args.pipeline,
        host=args.host,
        port=args.port,
        log_level=args.log,
        security=security,
    )
