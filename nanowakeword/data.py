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


import os
import re
import torch
import random
import logging
import requests
import itertools
import torchaudio
import numpy as np
from tqdm import tqdm
from typing import List
from torch.utils.data import Dataset, Sampler
from numpy.lib.format import open_memmap
from .utils.logger import print_info
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
    import pronouncing


def augment_clips(
        clip_paths: List[str],
        total_length: int,
        sr: int = 16000,
        batch_size: int = 128,
        augmentation_settings: dict = None,
        background_clip_paths: List[str] = [],
        RIR_paths: List[str] = []
        ):
    """
    Applies batch-wise audio augmentation to a list of audio clips.

    This function loads audio files, resamples them to a fixed sample rate,
    converts them to mono, and enforces a fixed length through random cropping
    or zero-padding. It then applies a configurable sequence of audio
    augmentations including gain, background noise, room impulse response (RIR),
    pitch shifting, and colored noise. The function is implemented as a
    generator for memory-efficient large-scale data processing.

    Args:
        clip_paths (List[str]):
            Paths to input audio files.
        total_length (int):
            Target number of samples per audio clip.
        sr (int, optional):
            Target sample rate in Hz. Defaults to 16000.
        batch_size (int, optional):
            Number of audio clips processed per batch. Defaults to 128.
        augmentation_settings (dict, optional):
            Dictionary to override default augmentation parameters. Supported keys:
                - min_snr_in_db (float): Minimum SNR for background noise.
                - max_snr_in_db (float): Maximum SNR for background noise.
                - rir_prob (float): Probability of applying RIR convolution.
                - pitch_prob (float): Probability of applying pitch shifting.
                - min_pitch_semitones (float): Minimum pitch shift in semitones.
                - max_pitch_semitones (float): Maximum pitch shift in semitones.
                - gain_prob (float): Probability of applying gain adjustment.
                - min_gain_in_db (float): Minimum gain in decibels.
                - max_gain_in_db (float): Maximum gain in decibels.
        background_clip_paths (List[str], optional):
            Paths to background noise audio files.
        RIR_paths (List[str], optional):
            Paths to room impulse response (RIR) files.

    Returns:
        Generator[np.ndarray]:
            Yields batches of augmented audio with shape
            (batch_size, total_length) and dtype int16.
    """

    from torch_audiomentations import Compose, AddBackgroundNoise, ApplyImpulseResponse, PitchShift, Gain, AddColoredNoise

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cfg = {
        "min_snr_in_db": 3.0, "max_snr_in_db": 30.0,
        "rir_prob": 0.2,
        "pitch_prob": 0.3, "min_pitch_semitones": -2.0, "max_pitch_semitones": 2.0,
        "gain_prob": 1.0, "min_gain_in_db": -6.0, "max_gain_in_db": 6.0
    }
    if augmentation_settings:
        cfg.update(augmentation_settings)

    transforms = [
        Gain(min_gain_in_db=cfg["min_gain_in_db"], max_gain_in_db=cfg["max_gain_in_db"], p=cfg["gain_prob"], sample_rate=sr),
    ]

    if background_clip_paths: 
        transforms.append(
            AddBackgroundNoise(background_paths=background_clip_paths, min_snr_in_db=cfg["min_snr_in_db"], max_snr_in_db=cfg["max_snr_in_db"], p=0.8, sample_rate=sr)
        )

    if RIR_paths: 
        transforms.append(
            ApplyImpulseResponse(ir_paths=RIR_paths, p=cfg["rir_prob"], sample_rate=sr)
        )

    transforms.append(
        PitchShift(min_transpose_semitones=cfg["min_pitch_semitones"], max_transpose_semitones=cfg["max_pitch_semitones"], p=cfg["pitch_prob"], sample_rate=sr)
    )    
    
    transforms.append(
        AddColoredNoise(min_snr_in_db=20.0, max_snr_in_db=40.0, p=1.0, sample_rate=sr)
    )

    augmenter = Compose(transforms=transforms, output_type="dict")
    augmenter.to(device)

    # The list is being shuffled so that a new batch is created each time.
    random.shuffle(clip_paths)

    for i in range(0, len(clip_paths), batch_size):
        batch_paths = clip_paths[i:i+batch_size]
        
        batch_audio = []
        for clip_path in batch_paths:
            try:
                # Load the audio and trim it to the correct length
                waveform, clip_sr = torchaudio.load(clip_path)
                if clip_sr != sr:
                    waveform = torchaudio.transforms.Resample(orig_freq=clip_sr, new_freq=sr)(waveform)
                
                # Mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                current_len = waveform.shape[1]
                if current_len > total_length:
                    start = random.randint(0, current_len - total_length)
                    waveform = waveform[:, start:start+total_length]
                elif current_len < total_length:
                    padding_needed = total_length - current_len
                    start_pad = random.randint(0, padding_needed)
                    end_pad = padding_needed - start_pad
                    waveform = torch.nn.functional.pad(waveform, (start_pad, end_pad))
                
                batch_audio.append(waveform)

            except Exception as e:
                print(f"Warning: Skipping corrupted file {clip_path}: {e}")
                continue
        
        if not batch_audio:
            continue
        batch_tensor = torch.stack(batch_audio).to(device)
        
        with torch.no_grad():
            augmented_tensor = augmenter(samples=batch_tensor, sample_rate=sr)['samples']

        final_batch_np = augmented_tensor.cpu().numpy()
        
        # Normalization (peak normalization is safe here because it is done at the end of all augmentation)
        max_val = np.abs(final_batch_np).max(axis=-1, keepdims=True)
        max_val[max_val < 1e-8] = 1.0 
        
        # The output will be in the form (batch_size, samples)
        output_batch = (final_batch_np / max_val).squeeze(1) 
        
        yield (output_batch * 32767).astype(np.int16)



def mmap_batch_generator(source_registry, blueprints, batch_size, input_shape):
    """
    Standard Classification Batch Generator.
    - Creates balanced batches of (features, labels).
    - Uses blueprints to create complex acoustic scenes for both positive and negative samples.
    """
    import numpy as np
    import torch

    memmaps = {}
    indices = {}
    target_keys, negative_keys, background_keys = [], [], []

    for alias, meta in source_registry.items():
        try:
            path = meta['path']
            memmaps[alias] = np.load(path, mmap_mode='r')
            indices[alias] = len(memmaps[alias])
            t = meta['type']
            if t == 'target': target_keys.append(alias)
            elif t == 'negative': negative_keys.append(alias)
            elif t == 'background': background_keys.append(alias)
        except Exception as e:
            import logging
            logging.warning(f"[Data Warning] Could not load source '{alias}': {e}")

    if not target_keys:
        raise ValueError("[CRITICAL] No Target sources found! Cannot train.")
    if not (negative_keys or background_keys):
        raise ValueError("[CRITICAL] No Negative or Background sources found! Cannot train.")

    bp_list = [b['composition'] for b in blueprints]
    bp_weights = np.array([b.get('weight', 1.0) for b in blueprints], dtype=np.float32)
    bp_probs = bp_weights / np.sum(bp_weights)
    
    required_samples, feature_dim = input_shape

    while True:
        batch_x = np.zeros((batch_size, required_samples, feature_dim), dtype=np.float32)
        batch_y = np.zeros(batch_size, dtype=np.float32)

        for i in range(batch_size):
            template_idx = np.random.choice(len(bp_list), p=bp_probs)
            template = bp_list[template_idx]
            
            stitched_clips = []
            is_target_present = False
            target_clip_len = 0 
            
            for item in template:
                key = None
                source_pool = []
                
                if item == 'targets': source_pool = target_keys
                elif item == 'negatives': source_pool = negative_keys
                elif item == 'backgrounds': source_pool = background_keys
                elif item in memmaps: key = item

                if source_pool:
                    key = np.random.choice(source_pool)

                if key:
                    idx = np.random.randint(0, indices[key])
                    clip = memmaps[key][idx]
                    stitched_clips.append(clip)
                    if key in target_keys:
                        is_target_present = True
                        target_clip_len = clip.shape[0] 
            
            if not stitched_clips:
                continue

            full_audio = np.vstack(stitched_clips)
            curr_len = full_audio.shape[0]

            final_clip = np.zeros((required_samples, feature_dim), dtype=np.float32)

            if is_target_present:
                target_start_in_full = curr_len - target_clip_len
                
                if required_samples >= target_clip_len:
                    max_start_pos_in_final = required_samples - target_clip_len
                    start_pos_in_final = np.random.randint(0, max_start_pos_in_final + 1)                    
                    start_copy_from_full = max(0, target_start_in_full - start_pos_in_final)                    
                    start_paste_in_final = max(0, start_pos_in_final - target_start_in_full)
                    len_to_copy = min(required_samples - start_paste_in_final, curr_len - start_copy_from_full)
                    final_clip[start_paste_in_final : start_paste_in_final + len_to_copy] = \
                        full_audio[start_copy_from_full : start_copy_from_full + len_to_copy]

                else: 
                    start = np.random.randint(0, target_clip_len - required_samples + 1)
                    final_clip = full_audio[target_start_in_full + start : target_start_in_full + start + required_samples]
            else: 
                if curr_len > required_samples:
                    start = np.random.randint(0, curr_len - required_samples + 1)
                    final_clip = full_audio[start : start + required_samples]
                else: 
                    start = np.random.randint(0, required_samples - curr_len + 1)
                    final_clip[start : start + curr_len, :] = full_audio

            batch_x[i] = final_clip
            if is_target_present:
                batch_y[i] = 1.0
        
        yield (
            torch.from_numpy(batch_x),
            torch.from_numpy(batch_y)
        )


class WakeWordDataset(Dataset):
    """
    An enhanced Dataset class for wake word data that supports dynamic filtering.

    This class handles multiple .npy files for each class (positives, negatives)
    using efficient memory-mapping. It has been updated to track the "hardness"
    of each sample. The __len__ and __getitem__ methods are designed to work
    only on the subset of samples currently marked as "hard", effectively
    filtering out the "easy" ones that the model has already learned.
    """
    def __init__(self, positive_feature_paths: list, negative_feature_paths: list):
        super().__init__()

        # Adding its label (1.0 or 0.0) and path to each source
        self.sources = []
        if positive_feature_paths:
            self.sources.extend([(path, 1.0) for path in positive_feature_paths])
        if negative_feature_paths:
            self.sources.extend([(path, 0.0) for path in negative_feature_paths])

        if not self.sources:
            raise ValueError("No feature files provided to the dataset!")

        self.memmaps = []
        self.source_info = [] # Will store each file's information (label, length, cumulative length)
        
        cumulative_len = 0
        self.total_positives = 0
        self.total_negatives = 0

        for path, label in self.sources:
            try:
                memmap = np.load(path, mmap_mode='r')
                length = len(memmap)
                
                self.memmaps.append(memmap)
                self.source_info.append({
                    'label': label,
                    'length': length,
                    'start_index': cumulative_len,
                    'end_index': cumulative_len + length - 1
                })
                
                cumulative_len += length

                if label == 1.0:
                    self.total_positives += length
                else:
                    self.total_negatives += length

            except FileNotFoundError:
                print(f"WARNING: File not found, skipping: {path}")
            except Exception as e:
                print(f"WARNING: Could not load file {path}. Error: {e}")

        self.total_samples = cumulative_len
        
        # A boolean tensor where `True` indicates a sample is still "hard"
        # and should be used for training. All samples start as hard.
        self._is_hard_mask = torch.ones(self.total_samples, dtype=torch.bool)
        
        # This list holds the original indices (0 to total_samples-1) of all
        # samples that are currently considered hard. This is the active
        # set of data the model will train on.
        self._hard_indices_map = self._get_current_hard_indices()
        
        print_info(f"Dataset initialized | Positives: {self.total_positives} | Negatives: {self.total_negatives} | Total: {self.total_samples}")

    def _get_current_hard_indices(self):
        """Helper function to get a tensor of original indices for all hard samples."""
        return torch.where(self._is_hard_mask)[0]
        
    def __len__(self):
        """
        The effective length of the dataset is the number of samples
        currently marked as hard.
        """
        return len(self._hard_indices_map)

    def __getitem__(self, index_in_hard_list):
        """
        Fetches a single data sample using a filtered index.
        The incoming `index_in_hard_list` is an index into the list of
        hard samples, which is then mapped to the sample's original index.
        """
        if index_in_hard_list < 0 or index_in_hard_list >= len(self):
            raise IndexError("Index out of bounds for the current set of hard samples.")
        
        # Map the filtered index back to the original, absolute index.
        original_index = self._hard_indices_map[index_in_hard_list]

        # Use the original index to find the correct file and data.
        # This logic is the same as your original implementation.
        file_idx = 0
        for i, info in enumerate(self.source_info):
            if info['start_index'] <= original_index <= info['end_index']:
                file_idx = i
                break
        
        info = self.source_info[file_idx]
        local_index = original_index - info['start_index']
        feature = self.memmaps[file_idx][local_index]
        label = torch.tensor(info['label'], dtype=torch.float32)

        # Return the original index along with the data.
        # This is crucial for the training loop to know which sample's
        # status needs to be updated.
        return original_index.item(), torch.from_numpy(feature.astype(np.float32)), label

    def get_sample_weights(self):
        """
        This function is no longer needed if we use a simple random sampler,
        but it can be adapted to work on the hard set if class balancing is still desired.
        For now, we will replace it with a simpler sampler.
        """
        pass # This can be removed or adapted later.

    def mark_as_easy(self, original_indices: list):
        """
        Marks a list of samples as "easy" by updating the hardness mask.
        This effectively removes them from the training set for subsequent epochs.
        """
        if not original_indices:
            return
        
        # Set the mask to False for the given indices
        self._is_hard_mask[original_indices] = False
        
        # Refresh the map of hard indices
        self._hard_indices_map = self._get_current_hard_indices()
        
        # num_remaining = len(self)
        # percent_remaining = (num_remaining / self.total_samples) * 100 if self.total_samples > 0 else 0
        # print_info(
            # f"Filtered {len(original_indices)} easy samples. "
            # f"Remaining hard samples: {num_remaining}/{self.total_samples} ({percent_remaining:.2f}%)"
        # )


class HardSampleFilterSampler(Sampler):
    """
    A Sampler that works with the dynamically changing WakeWordDataset.

    Its primary job is to generate a randomly shuffled sequence of indices
    from 0 to the current number of "hard" samples in the dataset.
    The DataLoader uses this sequence to fetch batches.
    Since the dataset's length itself changes, this sampler remains very simple.
    """
    def __init__(self, data_source: WakeWordDataset):
        """
        Initializes the sampler.
        Args:
            data_source (WakeWordDataset): An instance of our modified dataset
                                           which knows its own effective length.
        """
        self.data_source = data_source

    def __iter__(self):
        """
        Provides an iterator over the indices of the hard samples.
        This is called by the DataLoader at the beginning of each epoch.
        """
        # self.data_source.__len__() dynamically returns the number of hard samples.
        num_hard_samples = len(self.data_source)
        
        # We simply generate a random permutation of indices from 0 to num_hard_samples - 1.
        # The WakeWordDataset.__getitem__ will handle mapping these temporary indices
        # to the correct, original sample indices.
        shuffled_indices = torch.randperm(num_hard_samples).tolist()
        
        return iter(shuffled_indices)

    def __len__(self):
        """
        Returns the number of samples in the iterator, which is the current
        number of hard samples.
        """
        return len(self.data_source)
    

def trim_mmap(target_path):
    """
    Refactored version: Removes trailing zero-filled rows from a .npy mmap file.
    Functionality remains identical to the original logic but implementation details vary.
    """
    # 1. Load the source file in read-only mode to inspect data
    source_data = np.load(target_path, mmap_mode='r')
    total_rows, dim_h, dim_w = source_data.shape
    
    # 2. Determine the cut-off point by scanning backwards
    # We start from the end and move up until we find non-zero data
    active_rows = total_rows
    while active_rows > 0:
        # Check if the row at (current index - 1) contains any non-zero value
        if np.any(source_data[active_rows - 1]):
            break
        active_rows -= 1
        
    # 'active_rows' is now the count of rows we want to keep
    
    # 3. Prepare the temporary file path and container
    # Using a slightly different naming convention for the temp file to avoid conflicts
    temp_filepath = target_path.replace(".npy", "_tmp.npy")
    
    destination_map = open_memmap(
        temp_filepath, 
        mode='w+', 
        dtype=np.float32,
        shape=(active_rows, dim_h, dim_w)
    )

    # 4. Copy data in chunks (Block processing)
    # Using a while loop with explicit bounds instead of range iteration
    block_size = 1024
    cursor = 0
    
    # Calculate total iterations for progress bar
    total_blocks = (active_rows // block_size) + (1 if active_rows % block_size != 0 else 0)

    with tqdm(total=total_blocks, desc="Optimizing mmap storage") as pbar:
        while cursor < active_rows:
            # Define the upper limit for the current slice
            limit = min(cursor + block_size, active_rows)
            
            # Transfer the slice
            destination_map[cursor:limit] = source_data[cursor:limit]
            
            # Ensure data is written to disk
            destination_map.flush()
            
            # Move cursor forward
            cursor = limit
            pbar.update(1)

    # 5. Clean up and Swap
    # Explicitly release file handles
    del source_data
    del destination_map
    
    # Replace the original file with the trimmed version
    if os.path.exists(target_path):
        os.remove(target_path)
        
    os.rename(temp_filepath, target_path)


_PHONEMIZE_AVAILABLE = True
try:
    from phonemize.preprocessing.text import (
        Preprocessor,
        LanguageTokenizer,
        SequenceTokenizer,
    )
    torch.serialization.add_safe_globals(
        [Preprocessor, LanguageTokenizer, SequenceTokenizer]
    )
except ImportError:
    _PHONEMIZE_AVAILABLE = False

def _require_phonemize():
    if not _PHONEMIZE_AVAILABLE:
        raise RuntimeError(
            "Phonemize support is not available. Please `pip install phonemize` package to use this feature."
        )
    
# Generate words that sound similar ("adversarial") to the input phrase using phoneme overlap
def _require_phonemize():
    """Helper to ensure phonemize is ready"""
    pass

def clean_input_text(text: str) -> str:
    """
    Cleans the input text based on requirements:
    1. Removes apostrophes merging the word (police's -> polices).
    2. Removes other special characters/punctuation.
    """
    # Remove apostrophes specifically to merge (e.g. "torsiello's" -> "torsiellos")
    text = text.replace("'", "")
    # Remove any other non-alphanumeric characters (keep spaces)
    text = re.sub(r"[^\w\s]", "", text)
    return text

def phoneme_replacement(input_chars, max_replace, replace_char='"(.){1,3}"'):
    results = []
    chars = list(input_chars)
    for r in range(1, max_replace+1):
        comb = itertools.combinations(range(len(chars)), r)
        for indices in comb:
            chars_copy = chars.copy()
            for i in indices:
                chars_copy[i] = replace_char
            results.append(' '.join(chars_copy))
    return results

_DICTIONARY_WORDS = None

def _get_dictionary_words():
    """Loads and filters words from the Pronunciation library dictionary."""
    global _DICTIONARY_WORDS
    if _DICTIONARY_WORDS is not None:
        return _DICTIONARY_WORDS
    
    print_info("Preparing dictionary, this may take a moment...")

    all_words = pronouncing.cmudict.words()
    # Only common words of 3 to 8 letters are being selected
    _DICTIONARY_WORDS = [w for w in all_words if 3 <= len(w) <= 8 and w.isalpha()]
    return _DICTIONARY_WORDS

def generate_adversarial_texts(input_text: str, N: int, include_partial_phrase: float = 0.0, include_input_words: float = 0.0, multi_word_prob: float = 0.4, max_multi_word_len: int = 3):

    """Generates phonetically similar adversarial words and phrases.

        This function creates words or phrases that sound similar to the input
        text but are spelled differently. These "adversarial examples" are useful
        for testing speech recognition models and other phonetic-based systems.

        The process works in two potential stages:
        1.  A "base" adversarial phrase is generated using phonetic matching. The
            parameters `include_partial_phrase` and `include_input_words` control
            the structure of this base phrase.
        2.  Optionally, this base phrase is then embedded within a longer, more
            complex phrase using random words from a dictionary. This expansion is
            controlled by `multi_word_prob` and `max_multi_word_len`.

        Note:
            This function currently supports English text only. Exact homophones
            (words with the same sound and spelling) are excluded from results.

        Parameters
        ----------
        input_text : str
                     The target text for which to generate adversarial examples.
        N : int
            The total number of unique adversarial texts to return.
        include_partial_phrase : float, optional
                                 Probability (0.0–1.0) of creating a base adversarial phrase with
                                 fewer words than the original `input_text`. Default is 0.0.
        include_input_words : float, optional
                              Probability (0.0–1.0) of keeping an original word from `input_text`
                              in the base adversarial phrase instead of replacing it with a
                              phonetically similar word. Default is 0.0.
                                   
                Example for "ok google":
                                        A value > 0.0 allows outputs like "ok module", preserving "ok".
        multi_word_prob : float, optional
                          Probability (0.0–1.0) of expanding the base adversarial phrase by
                          embedding it within a longer phrase of random dictionary words. This
                          applies to any input length. Default is 0.4.
        max_multi_word_len : int, optional
                             The maximum total words for an expanded phrase generated via
                             `multi_word_prob`. This sets the upper limit for the final phrase
                             length. Default is 3.

        Returns
        -------
        list[str]
            A list of phonetically similar words or phrases.
        """

    _require_phonemize()
    
    # CLEAN THE TEXT (police's -> polices)
    cleaned_input = clean_input_text(input_text)
    input_words = cleaned_input.split()
    
    if not input_words:
        return []

    # Get phonemes for english vowels (CMUDICT labels)
    vowel_phones = ["AA", "AE", "AH", "AO", "AW", "AX", "AXR", "AY", "EH", "ER", "EY", "IH", "IX", "IY", "OW", "OY", "UH", "UW", "UX"]

    word_phones = []
    input_text_phones = [pronouncing.phones_for_word(i) for i in input_words]

    # Phonemize Model Setup 
    if [] in input_text_phones:
        phonemizer_mdl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "resources", "phonemize_model", "phonemize_m1.pt")
        os.makedirs(os.path.dirname(phonemizer_mdl_path), exist_ok=True)

        if not os.path.exists(phonemizer_mdl_path):
            file_url = "https://github.com/arcosoph/phonemize/releases/download/v0.2.0/phonemize_m1.pt"
            logging.warning(f"Downloading phonemize model from {file_url}...")
            r = requests.get(file_url, stream=True)
            with open(phonemizer_mdl_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=2048):
                    if chunk:
                        f.write(chunk)

        from phonemize.phonemizer import Phonemizer
        phonemizer = Phonemizer.from_checkpoint(phonemizer_mdl_path)

    for phones, word in zip(input_text_phones, input_words):
        if phones:
            word_phones.extend(phones)
        else:
            logging.warning(f"Word '{word}' not found in dictionary. Using Phonemize_m1 model.")
            phones_pred = phonemizer(word, lang='en_us')
            logging.warning(f"Phones for '{word}': {phones_pred}")
            word_phones.append(re.sub(r"[\]|\[]", "", re.sub(r"\]\[", " ", phones_pred)))
        
        # Handle multiple pronunciations (take the first one for simplicity in this logic)
        if isinstance(word_phones[-1], list):
             word_phones[-1] = word_phones[-1][0]

    # Add lexical stresses regex
    word_phones = [re.sub('|'.join(vowel_phones), lambda x: str(x.group(0)) + '[0|1|2]', re.sub(r'\d+', '', i)) for i in word_phones]

    # Generate Candidates per Word 
    adversarial_phrases_map = [] # List of lists
    
    for phones_str, word in zip(word_phones, input_words):
        query_exps = []
        phones_list = phones_str.split()
        candidates = []
        
        if len(phones_list) <= 2:
            query_exps.append(" ".join(phones_list))
        else:
            query_exps.extend(phoneme_replacement(phones_list, max_replace=max(0, len(phones_list)-2), replace_char="(.){1,3}"))

        found_matches = set()
        for query in query_exps:
            matches = pronouncing.search(query)
            for m in matches:
                m_phones = pronouncing.phones_for_word(m)[0]
                # Must not have exact same phonemes and must not be exact same word
                if m_phones != phones_str and m.lower() != word.lower():

                    # Output word theke apostrophe (') soriye felun
                    # m = m.replace("'", "") 
                    m = re.sub(r"[^\w\s]", "", m)

                    found_matches.add(m)
        
        adversarial_phrases_map.append(list(found_matches))

    # Generation Loop with Automatic Relaxation 
    results = set() # Use a set to prevent duplicates
    
    # Tracking variables for automatic relaxation
    curr_inc_input_prob = include_input_words
    curr_inc_partial_prob = include_partial_phrase
    
    max_attempts = N * 50 # Fail-safe to prevent infinite loops
    attempts = 0
    consecutive_failures = 0
    
    dictionary_words = _get_dictionary_words()
    
    while len(results) < N and attempts < max_attempts:
        attempts += 1
        
        current_selection = []
        
        # Determining the number of words according to the include_partial_phrase parameter
        if len(input_words) > 1 and np.random.random() <= include_partial_phrase:
            target_len = np.random.randint(1, len(input_words) + 1)
            indices = sorted(np.random.choice(range(len(input_words)), target_len, replace=False))
        else:
            indices = range(len(input_words))

        # Selecting the main word or its synonyms according to the include_input_words parameter
        changeable_indices = [
            idx for idx in indices 
            if adversarial_phrases_map[idx]
        ]

        guaranteed_change_index = -1
        if changeable_indices:
            guaranteed_change_index = np.random.choice(changeable_indices)

        if guaranteed_change_index == -1:
            continue

        for idx in indices:
            original_word = input_words[idx]
            candidates = adversarial_phrases_map[idx]
            
            if idx == guaranteed_change_index:
                current_selection.append(np.random.choice(candidates))
            else:
                use_original = False
                if not candidates or np.random.random() <= include_input_words:
                    use_original = True
                
                if use_original:
                    current_selection.append(original_word)
                else:
                    current_selection.append(np.random.choice(candidates))
        
        base_adversarial_phrase = " ".join(current_selection)

        final_candidate_text = base_adversarial_phrase 
        dictionary_words = _get_dictionary_words()

        if dictionary_words and np.random.random() < multi_word_prob:
            
            # Set the length of the new sentence (longer than the base phrase, up to max_multi_word_len)
            base_len = len(base_adversarial_phrase.split())
            if base_len < max_multi_word_len:
                phrase_len = np.random.randint(base_len + 1, max_multi_word_len + 1)
                
                num_random_words = phrase_len - base_len
                
                random_words = list(np.random.choice(dictionary_words, num_random_words, replace=False))
                
                new_phrase_list = random_words + [base_adversarial_phrase]
                np.random.shuffle(new_phrase_list)
                
                final_candidate_text = " ".join(new_phrase_list)

        if final_candidate_text and final_candidate_text != cleaned_input:
            if final_candidate_text not in results:
                results.add(final_candidate_text)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
        else:
            consecutive_failures += 1
        
        # Automatic Parameter Adjustment Logic 
        if consecutive_failures > 50:
            changed = False
            if curr_inc_input_prob < 1.0:
                curr_inc_input_prob = min(1.0, curr_inc_input_prob + 0.1)
                changed = True
            
            if curr_inc_partial_prob < 1.0:
                curr_inc_partial_prob = min(1.0, curr_inc_partial_prob + 0.1)
                changed = True
            
            if changed:
                consecutive_failures = 0


    return list(results)