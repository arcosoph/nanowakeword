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



from typing import Deque

class VAD():
    """
    A model class for a voice activity detection (VAD) based on Silero's model.
    Dependencies are lazy-loaded to optimize performance when VAD is not in use.
    """

    def __init__(self,
                 model_path: str = None,
                 n_threads: int = 1
                 ):
        """Initialize the VAD model object."""
        
        # --- Truly Lazy Imports ---
        # All imports are moved inside to completely break any import cycle.
        import onnxruntime as ort
        import numpy as np
        from collections import deque
        import os

        # --- The Core Fix: A fully self-contained model loading logic ---
        if model_path is None:
            # To break the cycle, we avoid top-level imports and get the path dynamically.
            # This is a robust way to find the models directory.
            try:
                from .resources.models import models
                model_path = models.silero_vad_onnx
            except ImportError as e:
                # This is a fallback if the above still fails due to complex cycles.
                print(f"[FATAL] A critical import cycle detected. Cannot auto-load VAD model: {e}")
                # Provide a manual path as a last resort for the error message.
                models_dir = os.path.join(os.path.dirname(__file__), 'resources', 'models')
                expected_path = os.path.join(models_dir, 'silero_vad.onnx')
                raise ImportError(
                    "Could not lazy-load the VAD model due to a circular import. "
                    f"Please check your project's __init__.py files. "
                    f"Expected model path: {expected_path}"
                ) from e
        
        # Initialize the ONNX model
        sessionOptions = ort.SessionOptions()
        sessionOptions.inter_op_num_threads = n_threads
        sessionOptions.intra_op_num_threads = n_threads
        
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"VAD model not found at path: {model_path}")

        self.model = ort.InferenceSession(model_path, sess_options=sessionOptions,
                                          providers=["CPUExecutionProvider"])
        self.prediction_buffer: Deque[float] = deque(maxlen=125)
        self.sample_rate = np.array(16000).astype(np.int64)
        self.reset_states()

    def reset_states(self, batch_size=1):
        # Lazy-load numpy here as well, in case this method is called independently.
        import numpy as np
        self._h = np.zeros((2, batch_size, 64), dtype=np.float32)
        self._c = np.zeros((2, batch_size, 64), dtype=np.float32)
        self._last_sr = 0
        self._last_batch_size = 0


    def predict(self, x, frame_size=480):
        """
        Get the VAD predictions for the input audio frame.
        ...
        """
        # Lazy-load numpy for array operations.
        import numpy as np

        chunks = [(x[i:i + frame_size] / 32767.0).astype(np.float32)
                  for i in range(0, x.shape[0], frame_size)]

        frame_predictions = []
        for chunk in chunks:
            ort_inputs = {'input': chunk[None,],
                          'h': self._h, 'c': self._c, 'sr': self.sample_rate}
            ort_outs = self.model.run(None, ort_inputs)
            out, self._h, self._c = ort_outs
            frame_predictions.append(out[0][0])

        return np.mean(frame_predictions)


    def __call__(self, x, frame_size=160 * 4):
        self.prediction_buffer.append(self.predict(x, frame_size))

