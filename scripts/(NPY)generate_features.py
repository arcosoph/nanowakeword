# ---->>>
"""
Feature Generation Utility for NanoWakeWord

Purpose:
    This script converts raw audio files (.wav format) into a model-readable
    feature file (.npy). Its primary use case is to create custom validation
    or testing datasets, especially for measuring the false-positive rate of
    a trained wake word model.

How It Works:
    1.  Scans a specified input path for all `.wav` audio files.
    2.  Reads each audio file and breaks it down into uniform chunks (e.g., 2 seconds).
    3.  This process is memory-efficient, using a generator to handle very large
        files without loading them entirely into memory.
    4.  It then computes numerical features (e.g., MFCCs) from these audio chunks.
    5.  Finally, it saves all the computed features into a single, compact .npy file,
        which can be used for model evaluation.

Prerequisites:
    All input audio files MUST be in the following format:
    - Format: .wav
    - Sample Rate: 16000 Hz
    - Bit Depth: 16-bit
    - Channels: 1 (Mono)

Usage:
    Run this script from the root directory of the project.

    python -m scripts.generate_features --input <path_to_audio> --output <path_for_npy_file>

Command-Line Arguments:
    --input (required):
        Path to the source audio. Can be a single .wav file or a directory
        containing multiple .wav files.

    --output (required):
        The full path where the final .npy feature file will be saved.
        Example: 'data/validation/my_custom_features.npy'

    --chunk_duration (optional):
        The duration of each audio chunk in seconds. This should match the
        duration your model was trained on. Defaults to 2.0 seconds.

Example 1: Processing a directory of short negative clips
    python -m scripts.generate_features \\
      --input ./training_data/negative/ \\
      --output ./validation_data/negative_set_features.npy

Example 2: Processing a single, long 3-hour audio file
    python -m scripts.generate_features \\
      --input ./long_recordings/3_hour_podcast.wav \\
      --output ./validation_data/podcast_features.npy
"""

import argparse
import os
import numpy as np
import torch
import wave
from pathlib import Path
from tqdm import tqdm
from typing import List, Generator

# Import necessary functions from the NanoWakeWord project.
# Ensure that the project is installed or paths are correctly set.
from nanowakeword.utils.audio_processing import compute_features_from_generator


# --- Helper Functions ---

def get_audio_files(input_path: str) -> List[str]:
    """
    Checks if the input path is a file or a directory and returns a list of .wav files.
    """
    path = Path(input_path)
    if path.is_file() and path.suffix.lower() == '.wav':
        return [str(path)]
    elif path.is_dir():
        # Recursively find all .wav files in the directory.
        return [str(p) for p in path.glob('**/*.wav')]
    else:
        return []

def stream_long_audio(audio_path: str, chunk_duration_sec: float = 2.0, sample_rate: int = 16000) -> Generator[np.ndarray, None, None]:
    """
    Yields audio chunks of a specified duration from a WAV file.

    This function is a generator, making it memory-efficient for processing
    very large audio files that cannot be loaded into memory all at once.
    """
    chunk_size_samples = int(chunk_duration_sec * sample_rate)
    bytes_per_sample = 2  # Corresponds to 16-bit audio

    try:
        with wave.open(str(audio_path), 'rb') as wf:
            # Verify that the audio format is correct (16kHz, 16-bit, mono).
            if wf.getframerate() != sample_rate or wf.getsampwidth() != bytes_per_sample or wf.getnchannels() != 1:
                print(f"Warning: File '{audio_path}' is not 16kHz, 16-bit, mono. Skipping.")
                return

            num_frames = wf.getnframes()
            for pos in range(0, num_frames, chunk_size_samples):
                frames = wf.readframes(chunk_size_samples)
                # Ensure the last chunk is a full one before yielding.
                if len(frames) < (chunk_size_samples * bytes_per_sample):
                    break
                yield np.frombuffer(frames, dtype=np.int16)
    except Exception as e:
        print(f"Error processing '{audio_path}': {e}")


def main():
    """Main function to parse arguments and run the feature generation process."""
    parser = argparse.ArgumentParser(
        description="Generate a feature .npy file from a directory of audio clips or a single long audio file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a directory of .wav files OR a single long .wav file."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the output .npy file (e.g., ./my_validation_set.npy)."
    )
    parser.add_argument(
        "--chunk_duration",
        type=float,
        default=2.0,
        help="Duration of each audio chunk in seconds. Default: 2.0 seconds."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Ensure the output directory exists before saving the file.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting feature generation from '{input_path}'...")

    # --- Main Logic ---
    all_chunks = []

    audio_files = get_audio_files(input_path)
    if not audio_files:
        print(f"Error: No .wav files found at '{input_path}'. Please check the path.")
        return

    print(f"Found {len(audio_files)} audio file(s) to process.")

    # Iterate through all found audio files and collect chunks.
    # A progress bar is shown using tqdm for better user experience.
    for file_path in tqdm(audio_files, desc="Processing audio files"):
        for chunk in stream_long_audio(file_path, chunk_duration_sec=args.chunk_duration):
            all_chunks.append(chunk)

    if not all_chunks:
        print("Error: No valid audio chunks were generated. Please ensure audio files are 16kHz, 16-bit, mono WAV format.")
        return

    print(f"Generated a total of {len(all_chunks)} audio chunks.")

    # A generator function to feed the list of chunks in batches.
    def chunk_generator(chunks: List[np.ndarray], batch_size: int = 32) -> Generator[List[np.ndarray], None, None]:
        """Yields batches of audio chunks from a list."""
        for i in range(0, len(chunks), batch_size):
            yield chunks[i:i + batch_size]

    # Calculate the clip duration in samples.
    clip_duration_samples = int(args.chunk_duration * 16000)

    # Determine the device (GPU or CPU) for feature computation.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use multiple CPU cores if running on CPU for faster processing.
    ncpu = os.cpu_count() // 2 if device == "cpu" else 1

    print(f"Computing features using device: {device}...")
    compute_features_from_generator(
        generator=chunk_generator(all_chunks),
        n_total=len(all_chunks),
        clip_duration=clip_duration_samples,
        output_file=str(output_path),
        device=device,
        ncpu=ncpu
    )

    print(f"Feature generation complete! File saved to: {output_path}")


if __name__ == "__main__":
    main()
