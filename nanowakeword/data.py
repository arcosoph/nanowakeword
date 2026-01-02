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
from torch.utils.data import Dataset
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
    Multi-file capable Dataset class for wake word data.
    Handles multiple .npy files for each class (positives, negatives).
    Uses efficient memory-mapping for large datasets.
    Maps a global index to a specific file and a local index within that file.
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
                
                # Save information for this file
                self.source_info.append({
                    'label': label,
                    'length': length,
                    'start_index': cumulative_len,
                    'end_index': cumulative_len + length -1
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
        
        print_info(f"Dataset initialized | Positives: {self.total_positives} | Negatives: {self.total_negatives} | Total: {self.total_samples}")


    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        """
        Fetches a single data sample and its label using a global index.
        This is highly efficient as it uses binary search logic internally.
        """
        if index < 0 or index >= self.total_samples:
            raise IndexError("Index out of bounds")

        # A loop is being used here, but it is fast enough if the number of datasets is small.
        file_idx = 0
        for i, info in enumerate(self.source_info):
            if info['start_index'] <= index <= info['end_index']:
                file_idx = i
                break
        
        info = self.source_info[file_idx]
        local_index = index - info['start_index']
        feature = self.memmaps[file_idx][local_index]
        label = torch.tensor(info['label'], dtype=torch.float32)

        return torch.from_numpy(feature.astype(np.float32)), label

    def get_sample_weights(self):
        """
        Generates a weight for each sample in the dataset, used for balancing classes.
        This is crucial for the WeightedRandomSampler.
        """
        # print("Calculating sample weights for class balancing...") 
        weight_pos = 1.0 / self.total_positives if self.total_positives > 0 else 1.0
        weight_neg = 1.0 / self.total_negatives if self.total_negatives > 0 else 1.0

        sample_weights = np.zeros(self.total_samples, dtype=np.float32)

        for info in self.source_info:
            start, end = info['start_index'], info['end_index']
            if info['label'] == 1.0:
                sample_weights[start : end + 1] = weight_pos
            else:
                sample_weights[start : end + 1] = weight_neg
        
        return sample_weights


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

def generate_adversarial_texts(input_text: str, N: int, include_partial_phrase: float = 0.0, include_input_words: float = 0.0):

    """
    Generate adversarial words and phrases based on phonetic (phoneme-level) similarity.

    This function creates phonetically similar but non-identical words or phrases
    to a given input text. The generated outputs are intended to act as adversarial
    examples, meaning they sound similar to the original input but differ in spelling
    and meaning.

    - Currently supports **English text only**.
    - Exact homophones are intentionally excluded, as they do not constitute true
    adversarial examples for the original input.

    Parameters
    ----------
    input_text : str
        The target text for which adversarial words or phrases will be generated.

    N : int
        The total number of adversarial texts to return. The generation process
        relies on sampling, so:
        - Not all possible combinations are guaranteed to appear.
        - Duplicate outputs may occur.

    include_partial_phrase : float
                            Probability (0.0–1.0) of generating adversarial phrases that contain fewer
                            words than the input text. The number of words will always be between
                            1 and the total number of words in `input_text`.

    include_input_words : float
                        Probability (0.0–1.0) of retaining individual words from the original
                        input text when generating adversarial phrases (only applicable when
                        the input consists of multiple words).
                            
        Example:
                input_text = "ok google"
                A value > 0.0 allows outputs like "ok noodle",
                whereas a value of 0.0 prevents original words like "ok"
                from appearing in the adversarial results.

    Returns
    -------
    list[str]
        A list of words or phrases that are phonetically similar—but not identical—
        to the input text.
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
    
    while len(results) < N and attempts < max_attempts:
        attempts += 1
        
        # Build one candidate phrase
        current_selection = []
        
        # Decide word count (Partial logic), If partial prob is high, we might pick fewer words
        if len(input_words) > 1 and np.random.random() <= curr_inc_partial_prob:
            # Pick a random length between 1 and total words
            target_len = np.random.randint(1, len(input_words) + 1)
            # Pick random indices to keep
            indices = sorted(np.random.choice(range(len(input_words)), target_len, replace=False))
        else:
            indices = range(len(input_words))

        # Build the phrase based on indices
        for idx in indices:
            original_word = input_words[idx]
            candidates = adversarial_phrases_map[idx]
            
            # Decide: Use original word or adversarial word?
            use_original = False
            
            if not candidates:
                # If no candidates exist, we MUST use original (or skip if logic allowed, but here we keep structure)
                use_original = True
            elif np.random.random() <= curr_inc_input_prob:
                use_original = True
            
            if use_original:
                current_selection.append(original_word)
            else:
                current_selection.append(np.random.choice(candidates))
        
        candidate_text = " ".join(current_selection)
        
        # Check validity: Not empty, not exactly the input (unless input was 1 word and no candidates exist)
        if candidate_text and candidate_text != cleaned_input:
            if candidate_text not in results:
                results.add(candidate_text)
                consecutive_failures = 0 # Reset failure counter
            else:
                consecutive_failures += 1
        else:
            consecutive_failures += 1
            
        # Automatic Parameter Adjustment Logic
        # If we fail to find a NEW word for 50 tries in a row, relax constraints
        if consecutive_failures > 50:
            changed = False
            if curr_inc_input_prob < 1.0:
                curr_inc_input_prob = min(1.0, curr_inc_input_prob + 0.1)
                changed = True
            
            if curr_inc_partial_prob < 1.0:
                curr_inc_partial_prob = min(1.0, curr_inc_partial_prob + 0.1)
                changed = True
            
            if changed:
                # logging.info(f"Struggling to find unique phrases. Relaxing constraints: InputProb={curr_inc_input_prob:.2f}, PartialProb={curr_inc_partial_prob:.2f}")
                consecutive_failures = 0 # Reset to give new params a chance
            else:
                # If we are already at 1.0 for both and still failing, we might just break early
                # But let's keep trying until max_attempts just in case random sampling helps
                pass

    return list(results)