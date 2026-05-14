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


from __future__ import annotations
import os
import time
import wave
import logging
import threading
import numpy as np
import nanowakeword  # Required for VAD
from functools import partial
from collections import deque, defaultdict
from typing import Callable, List, Union, Dict, Optional, TYPE_CHECKING

# Conditionally import noisereduce to avoid a hard dependency.
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False

if TYPE_CHECKING:
    import onnxruntime

# Helper type for type hinting
HiddenState = Union[None, tuple[np.ndarray, np.ndarray]]


class DetectionResult:
    """
    Returned by :meth:`NanoInterpreter.predict` and available via properties
    after every inference call.

    Gives users a clean, attribute-based interface to all scores without
    having to index into raw dicts.

    Attributes:
        scores (dict):        Raw ``{model_name: score}`` dict - full access to
                              every loaded model's score.
        model_name (str):     The primary/verifier model name.
        score (float):        Score of the primary model (verifier in cascade,
                              only model otherwise).
        gate_name (str|None): Gate model name, or None if not in cascade mode.
        gate_score (float):   Gate model score (0.0 if not in cascade mode).
        detected (bool):      True when ``score >= threshold`` (only set when
                              a threshold was passed to ``predict()``).
        threshold (float):    The threshold used for ``detected``.
    """

    __slots__ = ("scores", "model_name", "gate_name", "threshold", "_detected")

    def __init__(
        self,
        scores: dict,
        model_name: str,
        gate_name: Optional[str],
        threshold: float = 0.0,
    ):
        self.scores     = scores
        self.model_name = model_name
        self.gate_name  = gate_name
        self.threshold  = threshold
        self._detected  = None  # lazy - only computed if accessed

    @property
    def score(self) -> float:
        """Score of the primary (verifier) model."""
        return self.scores.get(self.model_name, 0.0)

    @property
    def gate_score(self) -> float:
        """Score of the gate model, or 0.0 if not in cascade mode."""
        if self.gate_name:
            return self.scores.get(self.gate_name, 0.0)
        return 0.0

    @property
    def detected(self) -> bool:
        """True when the primary score meets or exceeds the threshold."""
        return self.score >= self.threshold if self.threshold > 0 else False

    def get(self, model_name: str, default: float = 0.0) -> float:
        """Dict-compatible access - ``result.get("my_model")``."""
        return self.scores.get(model_name, default)

    def __getitem__(self, key: str) -> float:
        """Dict-compatible subscript - ``result["my_model"]``."""
        return self.scores[key]

    def __contains__(self, key: str) -> bool:
        return key in self.scores

    def __repr__(self) -> str:
        parts = [f"score={self.score:.4f}"]
        if self.gate_name:
            parts.append(f"gate={self.gate_score:.4f}")
        if self.threshold > 0:
            parts.append(f"detected={self.detected}")
        return f"DetectionResult({', '.join(parts)})"

class NanoInterpreter:
    """
    Main inference engine for NanoWakeWord.

    Loads a custom-trained model, manages the audio preprocessing pipeline,
    and performs real-time, stateful wake word detection with optional
    noise reduction and voice activity detection.

    This class should not be instantiated directly. Use the class method:
    `NanoInterpreter.load_model()` to create an instance.
    """

    def __init__(self, wakeword_models: List[str], **kwargs):
        """
        Private constructor. Use `.load_model()` to create an instance.
        """
        self._ort = self._import_onnx_runtime()

        # Setup core attributes 
        self.models: Dict[str, "onnxruntime.InferenceSession"] = {}
        self.model_input_names: Dict[str, List[str]] = {}
        self.model_feature_length: Dict[str, int] = {}
        self.class_mapping: Dict[str, Dict[str, str]] = {}

        # State Management (for RNN/LSTM/GRU) -
        self.is_stateful: Dict[str, bool] = {}
        self.hidden_states: Dict[str, HiddenState] = {}

        #  Transparent Scoring Attributes 
        self.raw_scores: Dict[str, float] = {}
        self.post_processed_scores: Dict[str, float] = {}
        
        # Model Loading Loop 
        for mdl_path in wakeword_models:
            mdl_name = os.path.splitext(os.path.basename(mdl_path))[0]
            if mdl_name in self.models:
                logging.warning(f"Model with name '{mdl_name}' is already loaded. Skipping.")
                continue

            session = self._create_onnx_session(mdl_path)
            self.models[mdl_name] = session
            
            inputs = session.get_inputs()
            self.model_input_names[mdl_name] = [inp.name for inp in inputs]
            self.model_feature_length[mdl_name] = inputs[0].shape[1]
            
            self._initialize_state_management(mdl_name)
            self.class_mapping[mdl_name] = {"0": mdl_name}

            # Initialize scores for each model
            self.raw_scores[mdl_name] = 0.0
            self.post_processed_scores[mdl_name] = 0.0

        # Buffer, Preprocessor, and Optional Components Setup
        self._setup_components(**kwargs)

        # Cascade (2-stage) configuration
        # Populated by load_model when cascade mode is active.
        self.cascade_config: dict = {}

        # Listen thread management (for non-blocking listen())
        self._listen_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
    # Properties 

    @property
    def is_cascade(self) -> bool:
        """True when a 2-stage gate/verifier cascade is active."""
        return bool(self.cascade_config)

    @property
    def model_name(self) -> str:
        """
        The primary wake word model name.
        In cascade mode this is the verifier (Stage 2).
        In single-model mode this is the only loaded model.
        """
        if self.is_cascade:
            return self.cascade_config["verifier"]
        return next(iter(self.models))

    @property
    def gate_name(self) -> Optional[str]:
        """Name of the gate (Stage 1) model, or None if not in cascade mode."""
        return self.cascade_config.get("gate")

    @property
    def gate_score(self) -> float:
        """Latest post-processed score from the gate model (0.0 if not in cascade mode)."""
        if self.gate_name:
            return self.post_processed_scores.get(self.gate_name, 0.0)
        return 0.0

    @property
    def verifier_score(self) -> float:
        """Latest post-processed score from the main/verifier model."""
        return self.post_processed_scores.get(self.model_name, 0.0)

    @property
    def score(self) -> float:
        """
        Convenience alias for the most relevant score:
        - In cascade mode: the verifier score (Stage 2).
        - In single-model mode: the only model's score.
        """
        return self.verifier_score

    @property
    def info(self) -> dict:
        """
        Structured summary of the interpreter's current state.
        Useful for logging, debugging, or building status UIs.
        """
        from nanowakeword.interpreter.remote_verifier import _RemoteSession
        verifier_name = self.cascade_config.get("verifier", self.model_name)
        is_remote = isinstance(self.models.get(verifier_name), _RemoteSession)
        d = {
            "model_name":     self.model_name,
            "is_cascade":     self.is_cascade,
            "is_remote":      is_remote,
            "gate_name":      self.gate_name,
            "gate_threshold": self.cascade_config.get("gate_threshold", None),
            "loaded_models":  list(self.models.keys()),
            "score":          self.score,
            "gate_score":     self.gate_score,
            "raw_scores":     dict(self.raw_scores),
        }
        if is_remote:
            d["remote_uri"] = self.models[verifier_name].uri
        return d

    def __repr__(self) -> str:
        if self.is_cascade:
            return (
                f"NanoInterpreter(model='{self.model_name}', "
                f"gate='{self.gate_name}', "
                f"gate_threshold={self.cascade_config.get('gate_threshold', 0.3)})"
            )
        models = list(self.models.keys())
        if len(models) == 1:
            return f"NanoInterpreter(model='{models[0]}')"
        return f"NanoInterpreter(models={models})"

    def detected(self, threshold: float, model: Optional[str] = None) -> bool:
        """
        Returns True if the current score meets or exceeds ``threshold``.

        A clean alternative to ``interpreter.score > 0.95`` when you want
        to be explicit, or when checking a specific model in a multi-model setup.

        Args:
            threshold:  Detection threshold (0–1).
            model:      Model name to check. Defaults to the primary model.

        Examples::

            if interpreter.detected(0.95):
                ...

            # Check a specific model by name
            if interpreter.detected(0.90, model="my_other_model"):
                ...
        """
        name = model or self.model_name
        return self.post_processed_scores.get(name, 0.0) >= threshold

    def stop(self) -> None:
        """
        Stops a running non-blocking ``listen()`` loop started with
        ``blocking=False``. Safe to call even if no loop is running.
        """
        if self._stop_event is not None:
            self._stop_event.set()
        if self._listen_thread is not None and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=2.0)
        self._listen_thread = None
        self._stop_event    = None

    @classmethod
    def load_model(
        cls,
        model: Union[str, List[str], None] = None,
        cascade: bool = False,
        gate_model: Optional[str] = None,
        gate_threshold: float = 0.3,
        remote_verifier: Optional[str] = None,
        remote_pipeline: str = "verifier_only",
        remote_timeout: float = 2.0,
        remote_api_key: Optional[str] = None,
        remote_token: Optional[str] = None,
        remote_ssl_certfile: Optional[str] = None,
        remote_ssl_keyfile: Optional[str] = None,
        remote_ssl_ca_certs: Optional[str] = None,
        **kwargs,
    ):
        """
        Loads wake word model(s) from local file paths.

        Args:
            model:             Path (or list of paths) to the main .onnx model file(s).
                               Can be None when remote_pipeline="full" and no local
                               gate model is needed — the server handles everything.
            cascade:           Enable 2-stage cascade mode. When True, the interpreter
                               automatically discovers ``<model_name>_lite.onnx`` in the
                               same directory and uses it as a lightweight gatekeeper
                               (Stage 1). The main model only runs when the gate score
                               exceeds ``gate_threshold``.
            gate_model:        Optional explicit path to a custom gate model.
                               Implies ``cascade=True`` automatically.
            gate_threshold:    Score above which Stage 2 (main model) is activated.
                               Default 0.3.
            remote_verifier:   WebSocket URI of a RemoteVerifier server, e.g.
                               ``"ws://192.168.1.100:8765"``. When set, the verifier
                               runs on the remote machine.
            remote_pipeline:   What the remote server handles. Must match the server's
                               ``--pipeline`` argument:
                                 "verifier_only" (default) — edge sends pre-computed
                                   features; server runs only the wake word model.
                                   mel + embedding run locally on the edge device.
                                 "full" — edge sends raw audio; server runs the
                                   complete pipeline (mel + embedding + verifier).
                                   Use this for very low-power edge devices.
            remote_timeout:    Seconds to wait for a server response. Default 2.0.
            remote_api_key:    API key sent as ``X-API-Key`` during websocket
                               handshake for server auth.
            remote_token:      Token sent as ``X-Token`` during websocket
                               handshake when the remote server requires token auth.
            remote_ssl_certfile: Optional client certificate bundle for mutual TLS.
            remote_ssl_keyfile: Optional private key file for client certificate auth.
            remote_ssl_ca_certs: Optional CA bundle path for WSS/TLS server verification.
            **kwargs:          Passed through to NanoInterpreter (vad_threshold,
                               enable_noise_reduction, etc.)

        Examples::

            # Fully local — single model
            NanoInterpreter.load_model("my_model.onnx")

            # Local cascade — auto-discover lite model
            NanoInterpreter.load_model("my_model.onnx", cascade=True)

            # Gate local, verifier remote (default — edge runs mel+embedding+gate)
            NanoInterpreter.load_model(
                model="my_model_lite.onnx",
                remote_verifier="ws://192.168.1.100:8765",
                gate_threshold=0.25,
            )

            # Gate local, full pipeline remote (edge runs only the gate)
            NanoInterpreter.load_model(
                model="my_model_lite.onnx",
                remote_verifier="ws://192.168.1.100:8765",
                remote_pipeline="full",
                gate_threshold=0.25,
            )

            # No local model — server handles everything
            NanoInterpreter.load_model(
                remote_verifier="ws://192.168.1.100:8765",
                remote_pipeline="full",
            )
        """
        from nanowakeword.interpreter.remote_verifier import (
            PIPELINE_VERIFIER_ONLY, PIPELINE_FULL, _VALID_PIPELINES
        )

        if remote_pipeline not in _VALID_PIPELINES:
            raise ValueError(
                f"Invalid remote_pipeline '{remote_pipeline}'. "
                f"Choose from: {sorted(_VALID_PIPELINES)}"
            )

        #  Resolve local model paths 
        paths: List[str] = []
        if model is not None:
            if isinstance(model, str):
                paths = [model]
            elif isinstance(model, list):
                paths = model
            else:
                raise TypeError("`model` must be a string, list of strings, or None.")
            for path in paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Model file not found: {path}")

        # Remote verifier mode 
        remote_cfg: Optional[dict] = None
        if remote_verifier is not None:
            if len(paths) > 1:
                raise ValueError(
                    "remote_verifier supports at most one local model path (the gate). "
                    "The verifier runs on the remote server."
                )

            if paths:
                gate_path  = paths[0]
                gate_stem  = os.path.splitext(os.path.basename(gate_path))[0]
                verifier_stem = (
                    gate_stem[:-5] if gate_stem.endswith("_lite")
                    else gate_stem + "_remote"
                )
            else:
                # No local model — server handles everything including gate
                gate_stem     = None
                verifier_stem = "remote_model"

            remote_cfg = {
                "gate":           gate_stem,
                "verifier":       verifier_stem,
                "gate_threshold": gate_threshold,
                "uri":            remote_verifier,
                "pipeline":       remote_pipeline,
                "timeout":        remote_timeout,
                "api_key":        remote_api_key,
                "token":          remote_token,
                "ssl_certfile":   remote_ssl_certfile,
                "ssl_keyfile":    remote_ssl_keyfile,
                "ssl_ca_certs":   remote_ssl_ca_certs,
            }
            logging.info(
                f"[NanoInterpreter] Remote mode: "
                f"gate='{gate_stem or 'none'}' (local) "
                f"-> verifier='{verifier_stem}' "
                f"(remote @ {remote_verifier}, pipeline='{remote_pipeline}')"
            )

        # Local cascade setup 
        cascade_cfg: dict = {}
        effective_cascade = cascade or (gate_model is not None)

        if remote_cfg is None and effective_cascade and len(paths) == 1:
            main_path = paths[0]
            stem      = os.path.splitext(os.path.basename(main_path))[0]

            if gate_model is not None:
                if not os.path.exists(gate_model):
                    raise FileNotFoundError(
                        f"The specified gate model does not exist: {gate_model}"
                    )
                resolved_gate_path = gate_model
                gate_name = os.path.splitext(os.path.basename(gate_model))[0]
                logging.info(
                    f"[NanoInterpreter] Cascade (custom gate): "
                    f"gate='{gate_name}' -> verifier='{stem}'"
                )
            else:
                model_dir          = os.path.dirname(os.path.abspath(main_path))
                gate_name          = stem + "_lite"
                resolved_gate_path = os.path.join(model_dir, gate_name + ".onnx")

                if not os.path.exists(resolved_gate_path):
                    logging.warning(
                        f"[NanoInterpreter] cascade=True but no lite model found at "
                        f"'{resolved_gate_path}'. Falling back to single-model mode. "
                        f"Generate one with: nanowakeword-train -c config.yaml --distill"
                    )
                    resolved_gate_path = None

                if resolved_gate_path:
                    logging.info(
                        f"[NanoInterpreter] Cascade (auto-discovered): "
                        f"gate='{gate_name}' -> verifier='{stem}'"
                    )

            if resolved_gate_path:
                paths = [resolved_gate_path, main_path]
                cascade_cfg = {
                    "gate":           gate_name,
                    "verifier":       stem,
                    "gate_threshold": gate_threshold,
                }

        # Build instance
        # When remote_pipeline="full" and no local model, we need an interpreter
        # with no local ONNX models — skip AudioFeatures download too.
        no_local_models = (remote_cfg is not None and not paths)

        if no_local_models:
            # Create a minimal instance without AudioFeatures
            # (server handles mel+embedding, edge just sends raw audio)
            instance = cls.__new__(cls)
            instance._ort                  = instance._import_onnx_runtime()
            instance.models                = {}
            instance.model_input_names     = {}
            instance.model_feature_length  = {}
            instance.class_mapping         = {}
            instance.is_stateful           = {}
            instance.hidden_states         = {}
            instance.raw_scores            = {}
            instance.post_processed_scores = {}
            instance.cascade_config        = {}
            instance._listen_thread        = None
            instance._stop_event           = None
            # Still set up VAD and noise reduction from kwargs, but skip AudioFeatures
            instance._setup_components_no_preprocessor(**kwargs)
        else:
            instance = cls(wakeword_models=paths, **kwargs)

        if remote_cfg is not None:
            instance._inject_remote_session(remote_cfg, no_local_models)
            instance.cascade_config = {
                "gate":           remote_cfg["gate"],
                "verifier":       remote_cfg["verifier"],
                "gate_threshold": remote_cfg["gate_threshold"],
            }
            # If no gate, clear cascade so predict() doesn't try to gate-check
            if remote_cfg["gate"] is None:
                instance.cascade_config = {}
        else:
            instance.cascade_config = cascade_cfg

        return instance

    def _inject_remote_session(self, remote_cfg: dict, no_local_models: bool = False) -> None:
        """
        Registers a _RemoteSession as the verifier model slot.
        Called internally by load_model when remote_verifier is set.
        """
        from nanowakeword.interpreter.remote_verifier import _RemoteSession, PIPELINE_FULL

        verifier_name = remote_cfg["verifier"]
        pipeline      = remote_cfg["pipeline"]

        remote_session = _RemoteSession(
            uri=remote_cfg["uri"],
            model_name=verifier_name,
            pipeline=pipeline,
            timeout=remote_cfg["timeout"],
            api_key=remote_cfg.get("api_key"),
            token=remote_cfg.get("token"),
            ssl_certfile=remote_cfg.get("ssl_certfile"),
            ssl_keyfile=remote_cfg.get("ssl_keyfile"),
            ssl_ca_certs=remote_cfg.get("ssl_ca_certs"),
        )

        self.models[verifier_name]               = remote_session
        self.model_input_names[verifier_name]    = ["input"]
        self.model_feature_length[verifier_name] = remote_session.get_inputs()[0].shape[1]
        self.is_stateful[verifier_name]          = False
        self.hidden_states[verifier_name]        = None
        self.raw_scores[verifier_name]           = 0.0
        self.post_processed_scores[verifier_name] = 0.0
        self.class_mapping[verifier_name]        = {"0": verifier_name}

        logging.info(
            f"[NanoInterpreter] Remote verifier '{verifier_name}' registered "
            f"(pipeline='{pipeline}')."
        )

    def _setup_components_no_preprocessor(self, **kwargs):
        """
        Minimal component setup for the no-local-model case (remote_pipeline='full'
        with no gate). Sets up VAD and noise reduction but skips AudioFeatures
        so mel/embedding models are NOT downloaded on the edge device.
        """
        from functools import partial
        from collections import deque, defaultdict

        self.prediction_buffer = defaultdict(partial(deque, maxlen=30))

        enable_nr = kwargs.pop("enable_noise_reduction", False)
        self.noise_reducer_enabled = enable_nr
        if enable_nr and not NOISEREDUCE_AVAILABLE:
            logging.warning(
                "`enable_noise_reduction` is True but `noisereduce` is not installed."
            )
            self.noise_reducer_enabled = False

        self.vad_threshold = kwargs.pop("vad_threshold", 0)
        if self.vad_threshold > 0:
            self.vad = nanowakeword.VAD()

        # No preprocessor — raw audio is sent directly to the remote server
        self.preprocessor = None

    def predict(self, x: np.ndarray, patience: dict = {}, threshold: dict = {}, debounce_time: float = 0.0) -> DetectionResult:
        """
        Performs inference on a chunk of audio data.

        Returns a :class:`DetectionResult` - a rich object that supports both
        attribute access and dict-style access for full backward compatibility.

        Args:
            x:             16-bit PCM audio chunk as a numpy array.
            patience:      ``{model_name: n_frames}`` - require N consecutive
                           frames above threshold before firing.
            threshold:     ``{model_name: value}`` - per-model thresholds used
                           by patience and debounce filters.
            debounce_time: Suppress repeated detections within this many seconds.

        Returns:
            :class:`DetectionResult` with ``.score``, ``.gate_score``,
            ``.detected``, ``.scores`` (full dict), and dict-compatible
            ``.get()`` / ``[]`` access.

        Raw (pre-filter) scores are always available on ``self.raw_scores``.
        """
        if not isinstance(x, np.ndarray):
            raise ValueError("Input audio `x` must be a Numpy array.")

        # Noise Reduction
        if self.noise_reducer_enabled:
            x = self._reduce_noise(x)

        #  Full-remote mode: no local preprocessor 
        # When remote_pipeline="full" and no local model, send raw audio directly
        # to the remote session and return its score.
        if self.preprocessor is None:
            current_raw_preds = {}
            for mdl_name, session in self.models.items():
                output_raw = session.run(None, {"audio": x})
                score      = output_raw[0].item()
                self.raw_scores[mdl_name] = score
                if len(self.prediction_buffer.get(mdl_name, [])) < 5:
                    score = 0.0
                current_raw_preds[mdl_name] = score

            for mdl_name, score in current_raw_preds.items():
                self.prediction_buffer[mdl_name].append(score)
                self.post_processed_scores[mdl_name] = score

            return DetectionResult(
                scores=dict(current_raw_preds),
                model_name=self.model_name,
                gate_name=self.gate_name,
            )

        # Pre-process Audio & Get Features
        n_prepared_samples = self.preprocessor(x)

        # If not enough new audio, return the last known state.
        if n_prepared_samples < 1280:
            return DetectionResult(
                scores=dict(self.post_processed_scores),
                model_name=self.model_name,
                gate_name=self.gate_name,
            )

        current_raw_preds = {}
        for mdl_name, session in self.models.items():
            required_frames = self.model_feature_length[mdl_name]

            # Skip inference if the feature buffer hasn't warmed up enough yet.
            if self.preprocessor.feature_buffer.shape[0] < required_frames:
                current_raw_preds[mdl_name] = 0.0
                continue

            # Cascade gate check - skip verifier if gate hasn't fired
            if self.cascade_config:
                gate_name_     = self.cascade_config["gate"]
                verifier_name  = self.cascade_config["verifier"]
                gate_threshold = self.cascade_config["gate_threshold"]

                if mdl_name == verifier_name:
                    gate_score_ = current_raw_preds.get(gate_name_, 0.0)
                    if gate_score_ < gate_threshold:
                        current_raw_preds[mdl_name] = 0.0
                        continue

            features   = self.preprocessor.get_features(required_frames)
            input_feed = {"input": features}

            if self.is_stateful[mdl_name]:
                h_in, c_in = self.hidden_states.get(mdl_name) or self._get_initial_state(session)
                input_feed["hidden_in"], input_feed["cell_in"] = h_in, c_in
                output_raw = session.run(None, input_feed)
                prediction_scores = output_raw[0]
                self.hidden_states[mdl_name] = (output_raw[1], output_raw[2])
            else:
                output_raw        = session.run(None, input_feed)
                prediction_scores = output_raw[0]

            score = prediction_scores.item()

            # Store the RAW score before any filtering
            self.raw_scores[mdl_name] = score

            # Zero out initial predictions to prevent instability
            if len(self.prediction_buffer.get(mdl_name, [])) < 5:
                score = 0.0

            current_raw_preds[mdl_name] = score

        # Apply Filters (VAD, Patience, Debounce)
        final_predictions = current_raw_preds.copy()

        if self.vad_threshold > 0:
            self.vad(x)
            vad_frames    = list(self.vad.prediction_buffer)[-7:-4]
            vad_max_score = np.max(vad_frames) if len(vad_frames) > 0 else 0
            if vad_max_score < self.vad_threshold:
                for mdl_name in final_predictions:
                    final_predictions[mdl_name] = 0.0

        self._apply_post_processing(final_predictions, patience, threshold, debounce_time, n_prepared_samples)

        # Update buffers and state
        for mdl_name, score in final_predictions.items():
            self.prediction_buffer[mdl_name].append(score)
            self.post_processed_scores[mdl_name] = score

        return DetectionResult(
            scores=dict(final_predictions),
            model_name=self.model_name,
            gate_name=self.gate_name,
        )

    def reset(self):
        """Resets the interpreter's internal state for a new session."""
        self.prediction_buffer.clear()
        self.preprocessor.reset()
        for mdl_name in self.hidden_states:
            self.hidden_states[mdl_name] = None
        for mdl_name in self.raw_scores:
            self.raw_scores[mdl_name] = 0.0
            self.post_processed_scores[mdl_name] = 0.0

    def predict_clip(self, clip: Union[str, np.ndarray], chunk_size: int = 1280, **kwargs) -> list:
        """Predicts on a full audio clip by simulating a stream."""
        if isinstance(clip, str):
            with wave.open(clip, mode='rb') as f:
                if f.getframerate() != 16000 or f.getsampwidth() != 2 or f.getnchannels() != 1:
                    raise ValueError("Audio clip must be a 16kHz, 16-bit, single-channel WAV file.")
                data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        elif isinstance(clip, np.ndarray):
            data = clip
        else:
            raise TypeError("`clip` must be a file path (string) or a numpy array.")

        predictions = [self.predict(data[i:i+chunk_size], **kwargs) for i in range(0, len(data), chunk_size)]
        return predictions

    def listen(
        self,
        on_detection: Optional[Callable[[str, float], None]] = None,
        threshold: float = 0.5,
        cooldown: float = 1.0,
        chunk_size: int = 1280,
        on_score: Optional[Callable[[float, float], None]] = None,
        on_audio: Optional[Callable[[np.ndarray], None]] = None,
        blocking: bool = True,
    ) -> None:
        """
        Microphone loop. Handles audio capture, inference, cooldown, and reset.

        Args:
            on_detection:  Called when a detection fires.
                           Signature: ``on_detection(model_name: str, score: float)``
                           Defaults to printing a message if not provided.
            threshold:     Detection threshold (0–1). Default 0.5.
            cooldown:      Minimum seconds between detections. Default 1.0.
            chunk_size:    Audio frames per inference chunk. Default 1280 (80 ms).
            on_score:      Optional callback called every chunk with the latest scores.
                           Signature: ``on_score(verifier_score: float, gate_score: float)``
                           Gate score is 0.0 when not in cascade mode.
            on_audio:      Optional callback called with every raw audio chunk before
                           inference. Signature: ``on_audio(audio: np.ndarray)``
                           Useful for recording, visualization, or custom preprocessing.
            blocking:      If True (default), blocks until Ctrl+C. If False, starts
                           a background thread and returns immediately. Call
                           ``.stop()`` to terminate the background loop.

        Example - minimal::

            interpreter.listen(threshold=0.95)

        Example - with callbacks::

            def detected(name, score):
                print(f"Wake word '{name}' detected! ({score:.3f})")

            def show(verifier, gate):
                print(f"gate={gate:.2f}  verifier={verifier:.4f}", end="\\r")

            interpreter.listen(on_detection=detected, threshold=0.95, on_score=show)

        Example - non-blocking::

            interpreter.listen(threshold=0.95, blocking=False)
            # ... do other work ...
            interpreter.stop()  # when done
        """
        try:
            import pyaudio
        except ImportError:
            raise ImportError(
                "PyAudio is required for listen(). Install it with: pip install pyaudio"
            )

        if on_detection is None:
            def on_detection(name: str, score: float) -> None:
                print(f"\nDetected '{name}'!  (score: {score:.5f})")

        def _loop():
            pa     = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=chunk_size,
            )

            last_detection = 0.0
            stop_event     = self._stop_event

            try:
                while not (stop_event and stop_event.is_set()):
                    audio = np.frombuffer(
                        stream.read(chunk_size, exception_on_overflow=False),
                        dtype=np.int16,
                    )

                    if on_audio is not None:
                        on_audio(audio)

                    self.predict(audio)

                    v_score = self.verifier_score
                    g_score = self.gate_score

                    if on_score is not None:
                        on_score(v_score, g_score)

                    now = time.monotonic()
                    if v_score > threshold and (now - last_detection) > cooldown:
                        on_detection(self.model_name, v_score)
                        last_detection = now
                        self.reset()

            except KeyboardInterrupt:
                pass
            finally:
                stream.stop_stream()
                stream.close()
                pa.terminate()

        if blocking:
            _loop()
        else:
            self._stop_event = threading.Event()
            self._listen_thread = threading.Thread(target=_loop, daemon=True)
            self._listen_thread.start()


    def _import_onnx_runtime(self):
        try:
            import onnxruntime as ort
            return ort
        except ImportError:
            raise ImportError("ONNX Runtime is not installed. Please run `pip install onnxruntime`.")

    def _create_onnx_session(self, path: str) -> "onnxruntime.InferenceSession":
        session_options = self._ort.SessionOptions()
        session_options.inter_op_num_threads = 1
        session_options.intra_op_num_threads = 1
        return self._ort.InferenceSession(path, sess_options=session_options, providers=["CPUExecutionProvider"])

    def _initialize_state_management(self, mdl_name: str):
        if 'hidden_in' in self.model_input_names[mdl_name]:
            self.is_stateful[mdl_name] = True
            self.hidden_states[mdl_name] = None
        else:
            self.is_stateful[mdl_name] = False

    def _get_initial_state(self, session: "onnxruntime.InferenceSession") -> HiddenState:
        h_input = next(inp for inp in session.get_inputs() if inp.name == 'hidden_in')
        c_input = next(inp for inp in session.get_inputs() if inp.name == 'cell_in')
        h0 = np.zeros(h_input.shape, dtype=np.float32)
        c0 = np.zeros(c_input.shape, dtype=np.float32)
        return (h0, c0)

    def _setup_components(self, **kwargs):
        from nanowakeword.data.AudioFeatures import AudioFeatures
        self.prediction_buffer = defaultdict(partial(deque, maxlen=30))

        # Pop and handle known interpreter-specific arguments
        enable_nr = kwargs.pop("enable_noise_reduction", False)
        self.noise_reducer_enabled = enable_nr
        if enable_nr and not NOISEREDUCE_AVAILABLE:
            logging.warning(
                "`enable_noise_reduction` is True, but `noisereduce` is not installed. "
                "Disabling feature. Please run `pip install noisereduce`."
            )
            self.noise_reducer_enabled = False

        self.vad_threshold = kwargs.pop("vad_threshold", 0)
        if self.vad_threshold > 0:
            self.vad = nanowakeword.VAD()

        # Initialize the preprocessor with any remaining kwargs
        self.preprocessor = AudioFeatures(**kwargs)

    def _reduce_noise(self, x: np.ndarray) -> np.ndarray:
        """Applies stationary noise reduction to an audio chunk."""
        try:
            audio_float = x.astype(np.float32) / 32767.0
            reduced_noise_audio = nr.reduce_noise(y=audio_float, sr=16000, stationary=True)
            return (reduced_noise_audio * 32767.0).astype(np.int16)
        except Exception as e:
            logging.warning(f"Noise reduction failed: {e}. Returning original audio.")
            return x

    def _apply_post_processing(self, predictions, patience, threshold, debounce_time, n_prepared_samples):
        if not patience and debounce_time <= 0:
            return

        if (patience or debounce_time > 0) and not threshold:
            raise ValueError("`threshold` must be provided when using `patience` or `debounce_time`.")
        if patience and debounce_time > 0:
            raise ValueError("`patience` and `debounce_time` cannot be used together.")
            
        for mdl_name in predictions.keys():
            if predictions[mdl_name] == 0.0:
                continue

            if mdl_name in patience:
                required_frames = patience[mdl_name]
                # Ensure we have enough frames in buffer before checking
                if len(self.prediction_buffer[mdl_name]) < required_frames:
                    predictions[mdl_name] = 0.0
                    continue
                
                recent_frames = np.array(list(self.prediction_buffer[mdl_name])[-(required_frames-1):] + [predictions[mdl_name]])
                if (recent_frames >= threshold[mdl_name]).sum() < required_frames:
                    predictions[mdl_name] = 0.0
            
            elif debounce_time > 0 and mdl_name in threshold:
                audio_frame_duration = n_prepared_samples / 16000.0
                if audio_frame_duration <= 0: continue # Avoid division by zero
                n_frames_to_check = int(np.ceil(debounce_time / audio_frame_duration))
                recent_predictions = np.array(self.prediction_buffer[mdl_name])[-n_frames_to_check:]
                if predictions[mdl_name] >= threshold[mdl_name] and (recent_predictions >= threshold[mdl_name]).any():
                    predictions[mdl_name] = 0.0
