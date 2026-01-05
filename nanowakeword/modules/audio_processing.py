# Copyright 2022 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

############################################################
# Modified by Muhammad Abid
#  For more information, visit the official repository:
#       https://github.com/arcosoph/nanowakeword
############################################################

import os
import numpy as np
from collections import deque
from multiprocessing.pool import ThreadPool
from typing import Union, List, Callable, Deque
import matplotlib.pyplot as plt
from nanowakeword.resources.models import models
import scipy.io.wavfile


# Base class for computing audio features using Google's speech_embedding
# model (https://tfhub.dev/google/speech_embedding/1)
class AudioFeatures():
    """
    A class for creating audio features from audio data, including melspectograms and Google's
    `speech_embedding` features.
    """
    def __init__(self,
                 melspec_model_path: str = "",
                 embedding_model_path: str = "",
                 sr: int = 16000,
                 ncpu: int = 1,
                 inference_framework: str = "onnx", # currently only onnx  suport
                 device: str = 'cpu'
                 ):
        """
        Initialize the AudioFeatures object.

        Args:
            melspec_model_path (str): The path to the model for computing melspectograms from audio data
            embedding_model_path (str): The path to the model for Google's `speech_embedding` model
            sr (int): The sample rate of the audio (default: 16000 khz)
            ncpu (int): The number of CPUs to use when computing melspectrograms and audio features (default: 1)
            inference_framework (str): The inference framework to use when for model prediction. Options are"onnx".
                                       scenarios ONNX models may be preferable.
            device (str): The device to use when running the models, either "cpu" or "gpu" (default is "cpu".)
                          Note that depending on the inference framework selected and system configuration,
                          this setting may not have an effect. For example, to use a GPU with the ONNX
                          framework the appropriate onnxruntime package must be installed.
        """
        # Initialize the models with the appropriate framework
        self.inference_framework = inference_framework
        self.device = device
        self.debug_mode = False
        self.debug_count = 0
        self.debug_limit = 30
        self.debug_dir = "debug_pipeline_visuals"

        if self.debug_mode:
            os.makedirs(self.debug_dir, exist_ok=True)
            for f in os.listdir(self.debug_dir):
                os.remove(os.path.join(self.debug_dir, f))
        self.sr = sr 
        
        if inference_framework == "onnx":
            try:
                import onnxruntime as ort
            except ImportError:
                raise ValueError("Tried to import onnxruntime, but it was not found. Please install it using `pip install onnxruntime`")

            melspec_model_path = models.melspectrogram_onnx
            embedding_model_path = models.embedding_model_onnx


            if ".tflite" in melspec_model_path or ".tflite" in embedding_model_path:
                raise ValueError("The onnx inference framework is selected, but tflite models were provided!")

            # Initialize ONNX options
            sessionOptions = ort.SessionOptions()
            sessionOptions.inter_op_num_threads = ncpu
            sessionOptions.intra_op_num_threads = ncpu

            # Melspectrogram model
            self.melspec_model = ort.InferenceSession(melspec_model_path, sess_options=sessionOptions,
                                                      providers=["CUDAExecutionProvider"] if device == "gpu" else ["CPUExecutionProvider"])
            self.onnx_execution_provider = self.melspec_model.get_providers()[0]
            self.melspec_model_predict = lambda x: self.melspec_model.run(None, {'input': x})

            # Audio embedding model
            self.embedding_model = ort.InferenceSession(embedding_model_path, sess_options=sessionOptions,
                                                        providers=["CUDAExecutionProvider"] if device == "gpu"
                                                        else ["CPUExecutionProvider"])
            self.embedding_model_predict = lambda x: self.embedding_model.run(None, {'input_1': x})[0].squeeze()

        # Create databuffers with empty/random data
        self.raw_data_buffer: Deque = deque(maxlen=sr*10)
        self.melspectrogram_buffer = np.ones((76, 32))  # n_frames x num_features
        self.melspectrogram_max_len = 10*97  # 97 is the number of frames in 1 second of 16hz audio
        self.accumulated_samples = 0  # the samples added to the buffer since the audio preprocessor was last called
        self.raw_data_remainder = np.empty(0)
        self.feature_buffer = self._get_embeddings(np.random.randint(-1000, 1000, 16000*4).astype(np.int16))
        self.feature_buffer_max_len = 120  # ~10 seconds of feature buffer history



    def reset(self):
        """Reset the internal buffers"""
        self.raw_data_buffer.clear()
        self.melspectrogram_buffer = np.ones((76, 32))
        self.accumulated_samples = 0
        self.raw_data_remainder = np.empty(0)
        self.feature_buffer = self._get_embeddings(np.random.randint(-1000, 1000, 16000*4).astype(np.int16))

    def _get_melspectrogram(self, x: Union[np.ndarray, List], melspec_transform: Callable = lambda x: x/10 + 2):
        """
        Function to compute the mel-spectrogram of the provided audio samples.

        Args:
            x (Union[np.ndarray, List]): The input audio data to compute the melspectrogram from
            melspec_transform (Callable): A function to transform the computed melspectrogram. Defaults to a transform
                                          that makes the ONNX melspectrogram model closer to the native Tensorflow
                                          implementation from Google (https://tfhub.dev/google/speech_embedding/1).

        Return:
            np.ndarray: The computed melspectrogram of the input audio data
        """
        # Get input data and adjust type/shape as needed
        x = np.array(x).astype(np.int16) if isinstance(x, list) else x
        if x.dtype != np.int16:
            raise ValueError("Input data must be 16-bit integers (i.e., 16-bit PCM audio)."
                             f"You provided {x.dtype} data.")
        x = x[None, ] if len(x.shape) < 2 else x
        x = x.astype(np.float32) if x.dtype != np.float32 else x

        # Get melspectrogram
        outputs = self.melspec_model_predict(x)
        spec = np.squeeze(outputs[0])

        # Arbitrary transform of melspectrogram
        spec = melspec_transform(spec)


        return spec


    def _get_embeddings_from_melspec(self, melspec):
        """
        Computes the Google `speech_embedding` features from a melspectrogram input

        Args:
            melspec (np.ndarray): The input melspectrogram

        Returns:
            np.ndarray: The computed audio features/embeddings
        """
        if melspec.shape[0] != 1:
            melspec = melspec[None, ]
        embedding = self.embedding_model_predict(melspec)
        return embedding


    def _get_embeddings(self, x: np.ndarray, window_size: int = 76, step_size: int = 8, **kwargs):
        """Function to compute the embeddings of the provide audio samples."""
        spec = self._get_melspectrogram(x, **kwargs)
        windows = []
        for i in range(0, spec.shape[0], 8):
            window = spec[i:i+window_size]
            if window.shape[0] == window_size:  # truncate short windows
                windows.append(window)

        batch = np.expand_dims(np.array(windows), axis=-1).astype(np.float32)
        embedding = self.embedding_model_predict(batch)
        return embedding


    def get_embedding_shape(self, audio_length: float, sr: int = 16000):
        """Function that determines the size of the output embedding array for a given audio clip length (in seconds)"""
        x = (np.random.uniform(-1, 1, int(audio_length*sr))*32767).astype(np.int16)
        return self._get_embeddings(x).shape


    def _get_melspectrogram_batch(self, x, batch_size=128, ncpu=1):
        """
        Compute the melspectrogram of the input audio samples in batches.
        """
        # Prepare ThreadPool object, if needed for multithreading
        pool = None
        if hasattr(self, 'onnx_execution_provider') and "CPU" in self.onnx_execution_provider:
            pool = ThreadPool(processes=ncpu)

        all_melspecs = []
        
        for i in range(0, x.shape[0], batch_size):
            batch = x[i:i+batch_size]

            if hasattr(self, 'onnx_execution_provider') and "CUDA" in self.onnx_execution_provider:
                results = self._get_melspectrogram(batch).squeeze(1) # Squeeze the channel dimension if present
            
            elif pool:
                chunksize = max(1, batch.shape[0] // ncpu)
                results = pool.map(self._get_melspectrogram, batch, chunksize=chunksize)
            else:
                results = [self._get_melspectrogram(sample) for sample in batch]

            all_melspecs.extend([np.squeeze(res) for res in results])

        # Cleanup ThreadPool
        if pool:
            pool.close()
            pool.join() 

        max_frames = max(spec.shape[0] for spec in all_melspecs)
        mel_bins = all_melspecs[0].shape[1] 
        
        final_melspecs_array = np.full((len(all_melspecs), max_frames, mel_bins), -80.0, dtype=np.float32) # -80dB with Padding

        for i, spec in enumerate(all_melspecs):
            current_frames = spec.shape[0]
            final_melspecs_array[i, :current_frames, :] = spec
            
        return final_melspecs_array



    def _get_embeddings_batch(self, x, batch_size=128, ncpu=1):
        """
        Compute the embeddings of the input melspectrograms in batches.

        Note that the optimal performance will depend in the interaction between the device,
        batch size, and ncpu (if a CPU device is used). The user is encouraged
        to experiment with different values of these parameters to identify
        which combination is best for their data, as often differences of 1-4x are seen.

        Args:
            x (ndarray): A numpy array of melspectrograms of shape (N, frames, melbins).
                        Assumes that all of the melspectrograms have the same shape.
            batch_size (int): The batch size to use when computing the embeddings
            ncpu (int): The number of CPUs to use when computing the embeddings. This argument has
                        no effect if the underlying model is executing on a GPU.

        Returns:
            ndarray: A numpy array of shape (N, frames, embedding_dim) containing the embeddings of
                    all N input melspectrograms
        """
        # Ensure input is the correct shape
        if x.shape[1] < 76:
            raise ValueError("Embedding model requires the input melspectrograms to have at least 76 frames")

        # Prepare ThreadPool object, if needed for multithreading
        pool = None
        if "CPU" in self.onnx_execution_provider:
            pool = ThreadPool(processes=ncpu)

        # Calculate array sizes and make batches
        n_frames = (x.shape[1] - 76)//8 + 1
        embedding_dim = 96  # fixed by embedding model
        embeddings = np.empty((x.shape[0], n_frames, embedding_dim), dtype=np.float32)

        batch = []
        ndcs = []
        for ndx, melspec in enumerate(x):
            window_size = 76
            for i in range(0, melspec.shape[0], 8):
                window = melspec[i:i+window_size]
                if window.shape[0] == window_size:  # ignore windows that are too short (truncates end of clip)
                    batch.append(window)
            ndcs.append(ndx)

            if len(batch) >= batch_size or ndx+1 == x.shape[0]:
                batch = np.array(batch).astype(np.float32)
                if "CUDA" in self.onnx_execution_provider:
                    result = self.embedding_model_predict(batch)

                elif pool:
                    chunksize = batch.shape[0]//ncpu if batch.shape[0] >= ncpu else 1
                    result = np.array(pool.map(self._get_embeddings_from_melspec,
                                      batch, chunksize=chunksize))

                for j, ndx2 in zip(range(0, result.shape[0], n_frames), ndcs):
                    embeddings[ndx2, :, :] = result[j:j+n_frames]

                batch = []
                ndcs = []

        # Cleanup ThreadPool
        if pool:
            pool.close()

        return embeddings

    def embed_clips(self, x, batch_size=128, ncpu=1):
        """
        Compute the embeddings of the input audio clips in batches.

        Note that the optimal performance will depend in the interaction between the device,
        batch size, and ncpu (if a CPU device is used). The user is encouraged
        to experiment with different values of these parameters to identify
        which combination is best for their data, as often differences of 1-4x are seen.

        Args:
            x (ndarray): A numpy array of 16 khz input audio data in shape (N, samples).
                        Assumes that all of the audio data is the same length (same number of samples).
            batch_size (int): The batch size to use when computing the embeddings
            ncpu (int): The number of CPUs to use when computing the melspectrogram. This argument has
                        no effect if the underlying model is executing on a GPU.

        Returns:
            ndarray: A numpy array of shape (N, frames, embedding_dim) containing the embeddings of
                    all N input audio clips
        """

        # Compute melspectrograms
        melspecs = self._get_melspectrogram_batch(x, batch_size=batch_size, ncpu=ncpu)

        # Compute embeddings from melspectrograms
        embeddings = self._get_embeddings_batch(melspecs[:, :, :, None], batch_size=batch_size, ncpu=ncpu)

        if self.debug_mode and self.debug_count < self.debug_limit:
            
            samples_to_save = min(batch_size, len(x))
            
            for i in range(samples_to_save):
                if self.debug_count >= self.debug_limit: break
                
                # Data Extraction
                audio_sample = x[i] if isinstance(x, list) else x[i]
                mel_sample = melspecs[i]
                emb_sample = embeddings[i]

                # SAVE AUDIO (.wav) 
                wav_filename = f"debug_sample_{self.debug_count}_augmented.wav"
                wav_path = os.path.join(self.debug_dir, wav_filename)
                
                try:
                    scipy.io.wavfile.write(wav_path, self.sr, audio_sample)
                    print(f"[DEBUG] Saved Audio: {wav_path}")
                except Exception as e:
                    print(f"Could not save wav: {e}")

                # SAVE VISUALS (.png) 
                fig, axes = plt.subplots(3, 1, figsize=(10, 12))
                
                # A. Raw Waveform
                axes[0].plot(audio_sample)
                axes[0].set_title(f"Sample {self.debug_count}: Augmented Audio Input", fontweight='bold')
                axes[0].set_ylabel("Amplitude")
                
                # B. Mel Spectrogram
                im1 = axes[1].imshow(mel_sample.T, aspect='auto', origin='lower', cmap='viridis')
                axes[1].set_title("Step 1: Mel-Spectrogram", fontweight='bold')
                axes[1].set_ylabel("Freq Bins")
                fig.colorbar(im1, ax=axes[1])

                # C. Embeddings
                im2 = axes[2].imshow(emb_sample.T, aspect='auto', origin='lower', cmap='magma')
                axes[2].set_title("Step 2: Final Embeddings", fontweight='bold')
                axes[2].set_ylabel("Feature Dim")
                axes[2].set_xlabel("Time Frames")
                fig.colorbar(im2, ax=axes[2])

                plt.tight_layout()
                
                img_filename = f"debug_sample_{self.debug_count}_visuals.png"
                img_path = os.path.join(self.debug_dir, img_filename)
                plt.savefig(img_path)
                plt.close()
                
                print(f"[DEBUG] Saved Visuals: {img_path}")
                
                self.debug_count += 1

        return embeddings



    def _streaming_melspectrogram(self, n_samples):
        """Note! There seem to be some slight numerical issues depending on the underlying audio data
        such that the streaming method is not exactly the same as when the melspectrogram of the entire
        clip is calculated. It's unclear if this difference is significant and will impact model performance.
        In particular padding with 0 or very small values seems to demonstrate the differences well.
        """
        if len(self.raw_data_buffer) < 400:
            raise ValueError("The number of input frames must be at least 400 samples @ 16khz (25 ms)!")

        self.melspectrogram_buffer = np.vstack(
            (self.melspectrogram_buffer, self._get_melspectrogram(list(self.raw_data_buffer)[-n_samples-160*3:]))
        )

        if self.melspectrogram_buffer.shape[0] > self.melspectrogram_max_len:
            self.melspectrogram_buffer = self.melspectrogram_buffer[-self.melspectrogram_max_len:, :]

    def _buffer_raw_data(self, x):
        """
        Adds raw audio data to the input buffer
        """
        self.raw_data_buffer.extend(x.tolist() if isinstance(x, np.ndarray) else x)

    def _streaming_features(self, x):
        # Add raw audio data to buffer, temporarily storing extra frames if not an even number of 80 ms chunks
        processed_samples = 0

        if self.raw_data_remainder.shape[0] != 0:
            x = np.concatenate((self.raw_data_remainder, x))
            self.raw_data_remainder = np.empty(0)

        if self.accumulated_samples + x.shape[0] >= 1280:
            remainder = (self.accumulated_samples + x.shape[0]) % 1280
            if remainder != 0:
                x_even_chunks = x[0:-remainder]
                self._buffer_raw_data(x_even_chunks)
                self.accumulated_samples += len(x_even_chunks)
                self.raw_data_remainder = x[-remainder:]
            elif remainder == 0:
                self._buffer_raw_data(x)
                self.accumulated_samples += x.shape[0]
                self.raw_data_remainder = np.empty(0)
        else:
            self.accumulated_samples += x.shape[0]
            self._buffer_raw_data(x)

        # Only calculate melspectrogram once minimum samples are accumulated
        if self.accumulated_samples >= 1280 and self.accumulated_samples % 1280 == 0:
            self._streaming_melspectrogram(self.accumulated_samples)

            # Calculate new audio embeddings/features based on update melspectrograms
            for i in np.arange(self.accumulated_samples//1280-1, -1, -1):
                ndx = -8*i
                ndx = ndx if ndx != 0 else len(self.melspectrogram_buffer)
                x = self.melspectrogram_buffer[-76 + ndx:ndx].astype(np.float32)[None, :, :, None]
                if x.shape[1] == 76:
                    self.feature_buffer = np.vstack((self.feature_buffer,
                                                    self.embedding_model_predict(x)))

            # Reset raw data buffer counter
            processed_samples = self.accumulated_samples
            self.accumulated_samples = 0

        if self.feature_buffer.shape[0] > self.feature_buffer_max_len:
            self.feature_buffer = self.feature_buffer[-self.feature_buffer_max_len:, :]

        return processed_samples if processed_samples != 0 else self.accumulated_samples

    def get_features(self, n_feature_frames: int = 16, start_ndx: int = -1):
        if start_ndx != -1:
            end_ndx = start_ndx + int(n_feature_frames) \
                if start_ndx + n_feature_frames != 0 else len(self.feature_buffer)
            return self.feature_buffer[start_ndx:end_ndx, :][None, ].astype(np.float32)
        else:
            return self.feature_buffer[int(-1*n_feature_frames):, :][None, ].astype(np.float32)

    def __call__(self, x):
        return self._streaming_features(x)
