# ______________________
# Silero VAD License
# ______________________

# MIT License

# Copyright (c) 2020-present Silero Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

########################################

# This file contains the implementation of a class for voice activity detection (VAD),
# based on the pre-trained model from Silero (https://github.com/snakers4/silero-vad).
# It can be used as with the NanoWakeWord library, or independently.

# ================================
# Modified and maintained by: Abid
# ================================



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
        """Initialize the VAD model object.

            Args:
                model_path (str): The path to the Silero VAD ONNX model. If not provided,
                                  the default model will be loaded from resources.
                n_threads (int): The number of threads to use for the VAD model.
        """
        # --- Lazy-load all required libraries here ---
        import onnxruntime as ort
        import numpy as np
        import os
        from collections import deque
        # ---------------------------------------------

        # মডেলের পাথ নির্ধারণ করুন যদি না দেওয়া থাকে
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "resources",
                "models",
                "silero_vad.onnx"
            )

        # Initialize the ONNX model
        sessionOptions = ort.SessionOptions()
        sessionOptions.inter_op_num_threads = n_threads
        sessionOptions.intra_op_num_threads = n_threads
        self.model = ort.InferenceSession(model_path, sess_options=sessionOptions,
                                          providers=["CPUExecutionProvider"])

        # Create buffer
        self.prediction_buffer: Deque[float] = deque(maxlen=125)

        # Set model parameters
        self.sample_rate = np.array(16000).astype(np.int64)

        # Reset model to start
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