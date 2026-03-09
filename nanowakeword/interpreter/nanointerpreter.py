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
import wave
import logging
import numpy as np
import nanowakeword # Required for VAD
from functools import partial
from collections import deque, defaultdict
from typing import List, Union, Dict, TYPE_CHECKING

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
        

    @classmethod
    def load_model(cls, model_path: Union[str, List[str]], **kwargs):
        """
        Loads wake word model(s) from local file paths. This is the primary
        factory method for creating an interpreter instance.
        """
        if isinstance(model_path, str):
            paths = [model_path]
        elif isinstance(model_path, list):
            paths = model_path
        else:
            raise TypeError("`model_path` must be a string or a list of strings.")

        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"The specified model file does not exist: {path}")

        return cls(wakeword_models=paths, **kwargs)

    def predict(self, x: np.ndarray, patience: dict = {}, threshold: dict = {}, debounce_time: float = 0.0) -> dict:
        """
        Performs inference on a chunk of audio data and returns the final,
        post-processed scores. Raw scores are stored in the `self.raw_scores` attribute.
        """
        if not isinstance(x, np.ndarray):
            raise ValueError("Input audio `x` must be a Numpy array.")

        # Noise Reduction 
        if self.noise_reducer_enabled:
            x = self._reduce_noise(x)

        # Pre-process Audio & Get Features 
        n_prepared_samples = self.preprocessor(x)
        
        # Run Inference 
        # If not enough new audio, don't run the model, just return the last known state.
        if n_prepared_samples < 1280:
             return self.post_processed_scores

        current_raw_preds = {}
        for mdl_name, session in self.models.items():
            features = self.preprocessor.get_features(self.model_feature_length[mdl_name])
            
            input_feed = {'input': features}

            if self.is_stateful[mdl_name]:
                h_in, c_in = self.hidden_states.get(mdl_name) or self._get_initial_state(session)
                input_feed['hidden_in'], input_feed['cell_in'] = h_in, c_in
                
                output_raw = session.run(None, input_feed)
                prediction_scores = output_raw[0]
                self.hidden_states[mdl_name] = (output_raw[1], output_raw[2])
            else:
                output_raw = session.run(None, input_feed)
                prediction_scores = output_raw[0]

            score = prediction_scores.item()
            
            # Store the RAW score before any filtering
            self.raw_scores[mdl_name] = score 
            
            # Zero out initial predictions to prevent instability
            if len(self.prediction_buffer.get(mdl_name, [])) < 5:
                score = 0.0
            
            current_raw_preds[mdl_name] = score

        # Apply Filters (VAD, Patience, etc.) 
        final_predictions = current_raw_preds.copy()

        # VAD Filter
        if self.vad_threshold > 0:
            self.vad(x)
            vad_frames = list(self.vad.prediction_buffer)[-7:-4]
            vad_max_score = np.max(vad_frames) if len(vad_frames) > 0 else 0
            if vad_max_score < self.vad_threshold:
                for mdl_name in final_predictions:
                    final_predictions[mdl_name] = 0.0

        # Patience & Debounce Filter
        self._apply_post_processing(final_predictions, patience, threshold, debounce_time, n_prepared_samples)

        # Update Buffers and Final State 
        for mdl_name, score in final_predictions.items():
            self.prediction_buffer[mdl_name].append(score)
            self.post_processed_scores[mdl_name] = score
            
        return self.post_processed_scores

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
        # (This method can remain as it was in your previous version)
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
