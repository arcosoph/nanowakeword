# ==============================================================================
#  NanoWakeWord — Lightweight and Intelligent Wake Word Detection System
#  © 2025 Arcosoph. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is provided on an "AS IS" BASIS,
#  without warranties or conditions of any kind, either express or implied.
#
#  For more information, visit the official repository:
#      https://github.com/arcosoph/nanowakeword
# ==============================================================================



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
    # def __init__(self, positive_path, negative_path, noise_path: list, rir_path):
    def __init__(self, positive_path, negative_path, noise_path: list, rir_path, 
             future_positive_samples=0, future_negative_samples=0):
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
        
        self.future_positive_samples = future_positive_samples
        self.future_negative_samples = future_negative_samples
        self.avg_clip_duration_sec = 2.0

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


    def _analyze_duration_and_power(self, file_paths, desc="files"):
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

       
        all_noise_duration = 0
        all_noise_rms = 0
        num_noise_files = 0
        self.stats['H_noise_paths'] = {} # for ConfigGenerator

        background_paths = self.paths.get('noise', []) # `__init__` 
        
        for path in background_paths:
            if not path: continue
            
            noise_files = self._get_directory_files(path)
            if not noise_files: continue
            
            # 
            dir_name = os.path.basename(path) if path else "unknown_noise"
            duration, rms = self._analyze_duration_and_power(noise_files, desc=dir_name)
            
            self.stats['H_noise_paths'][path] = duration / 3600
            
            all_noise_duration += duration
            all_noise_rms += rms * len(noise_files)
            num_noise_files += len(noise_files)
        
        self.stats['H_noise'] = all_noise_duration / 3600
        self.stats['A_noise'] = (all_noise_rms / num_noise_files) if num_noise_files > 0 else 0

        # --- RIR Clips Analysis ---
        rir_files = self._get_directory_files(self.paths['rir'])
        self.stats['N_rir'] = len(rir_files)

        print("Analysis complete!\n")
        return self.stats

