# Copyright 2025 Arcosoph. All rights reserved.
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



import os
import torch
import torchaudio
from tqdm import tqdm
import numpy as np
import logging
import warnings


# All warning hide 
warnings.filterwarnings("ignore")




logging.basicConfig(level=logging.INFO)
logging.getLogger("torchaudio").setLevel(logging.ERROR)

class DatasetAnalyzer:
    """
    Analyzes audio datasets to extract key statistical features for the
    Intelligent Configuration Engine.
    """
    def __init__(self, positive_path, negative_path, noise_path, rir_path):
        """
        Initializes the analyzer with paths to the clean, processed datasets.

        Args:
            positive_path (str): Path to the directory of positive (wakeword) clips.
            negative_path (str): Path to the directory of negative clips.
            noise_path (str): Path to the directory of background noise clips.
            rir_path (str): Path to the directory of Room Impulse Response (RIR) clips.
        """
        self.paths = {
            'positive': positive_path,
            'negative': negative_path,
            'noise': noise_path,
            'rir': rir_path
        }
        self.stats = {}

    def _get_directory_files(self, dir_path):
        """Helper function to get all file paths in a directory, handling errors."""
        if not os.path.isdir(dir_path):
            print(f"WARNING: Directory not found: {dir_path}. Skipping analysis for this path.")
            return []
        try:
            return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        except Exception as e:
            print(f"ERROR: Could not read directory {dir_path}: {e}")
            return []



    def _analyze_duration_and_power(self, file_paths):
        """
        Calculates total duration and average RMS power for a list of audio files.
        RMS power is used as a proxy for loudness.
        """
        total_duration_secs = 0
        total_rms = 0
        valid_files = 0

        if not file_paths:
            return 0, 0
        
   
        dir_name = os.path.basename(os.path.dirname(file_paths[0]))

        for f in tqdm(file_paths, desc=f"Analyzing {dir_name} files"):
        # ======================================================================
            try:
                waveform, sr = torchaudio.load(f)
                duration = waveform.shape[1] / sr
                total_duration_secs += duration

                rms_val = torch.sqrt(torch.mean(waveform**2))
                total_rms += rms_val.item()
                valid_files += 1


            except Exception:
                continue
        
        avg_rms = (total_rms / valid_files) if valid_files > 0 else 0
        return total_duration_secs, avg_rms


    def analyze(self):
        """
        Runs the full analysis on all provided dataset paths.

        Returns:
            dict: A dictionary containing the extracted statistical features.
        """
        print("Analyzing dataset characteristics...")

        # --- Positive Clips Analysis ---
        pos_files = self._get_directory_files(self.paths['positive'])
        self.stats['H_pos'] = 0
        if pos_files:
            duration_secs, _ = self._analyze_duration_and_power(pos_files)
            self.stats['H_pos'] = duration_secs / 3600  

        # --- Negative Clips Analysis ---
        neg_files = self._get_directory_files(self.paths['negative'])
        self.stats['H_neg'] = 0
        if neg_files:
            duration_secs, _ = self._analyze_duration_and_power(neg_files)
            self.stats['H_neg'] = duration_secs / 3600  

        # --- Noise Clips Analysis ---
        noise_files = self._get_directory_files(self.paths['noise'])
        self.stats['H_noise'] = 0
        self.stats['A_noise'] = 0
        if noise_files:
            duration_secs, avg_rms = self._analyze_duration_and_power(noise_files)
            self.stats['H_noise'] = duration_secs / 3600  
            self.stats['A_noise'] = avg_rms  

        # --- RIR Clips Analysis ---
        rir_files = self._get_directory_files(self.paths['rir'])
        self.stats['N_rir'] = len(rir_files)

        print("Analysis complete!\n")
        return self.stats

if __name__ == '__main__':
    # This block allows you to test the analyzer independently.
    # Create dummy directories and files to test.
    # Example usage:
    print("Running standalone test for DatasetAnalyzer...")
    
    # আপনার কম্পিউটারে টেস্ট করার জন্য এই পাথগুলো পরিবর্তন করুন
    test_positive_path = "./training_data/positive"
    test_negative_path = "./training_data/negative"
    test_noise_path = "./training_data/noise"
    test_rir_path = "./training_data/rir"

    
    os.makedirs(test_positive_path, exist_ok=True)
    os.makedirs(test_negative_path, exist_ok=True)
    os.makedirs(test_noise_path, exist_ok=True)
    os.makedirs(test_rir_path, exist_ok=True)

    
    if not os.listdir(test_positive_path):
        print("Creating a dummy audio file for testing...")
        sample_rate = 16000
        dummy_waveform = torch.randn(1, sample_rate * 2) # 2 seconds of noise
        torchaudio.save(os.path.join(test_positive_path, "dummy.wav"), dummy_waveform, sample_rate)

    analyzer = DatasetAnalyzer(
        positive_path=test_positive_path,
        negative_path=test_negative_path,
        noise_path=test_noise_path,
        rir_path=test_rir_path
    )
    
    stats = analyzer.analyze()
    
    print("\n--- Analysis Results ---")
    print(stats)
    print("------------------------")


