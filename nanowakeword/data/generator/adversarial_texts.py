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
import numpy as np
from typing import List
from nanowakeword.utils.logger import print_info
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
    import pronouncing



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

def adversarial_texts(input_text: str, N: int, include_partial_phrase: float = 0.0, include_input_words: float = 0.0, multi_word_prob: float = 0.4, max_multi_word_len: int = 3):

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
        phonemizer_mdl_path = os.path.join("NwwResourcesModel", "phonemize_model", "phonemize_m1.pt")
        
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





                    # # # We will update the following codes in the future. # # #
import os
import sys
import random
from typing import List
import torch
import requests

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
    print("ERROR: 'phonemize' library not installed. Run: pip install phonemize")
    sys.exit(1)

from phonemize.phonemizer import Phonemizer


class PhonemeAdversarialGenerator:
    """
    Generate phonetically DISTINCT adversarial samples using real phoneme models.
    For wake-word training with guaranteed phonetic distance.
    """
    
    def __init__(self, phonemizer: Phonemizer, min_distance: float = 0.35):
        self.phonemizer = phonemizer
        self.min_distance = min_distance
        
        self.phoneme_substitutions = {
            'AA': ['AO', 'AH', 'AE', 'ER', 'OW', 'UH', 'IH', 'EH', 'UW', 'AW', 'AX', 'IX'],
            'AE': ['EH', 'AH', 'AA', 'IH', 'AE', 'AX', 'EY', 'IY', 'IX', 'UH', 'ER'],
            'AH': ['ER', 'AA', 'UH', 'IH', 'AX', 'EH', 'AO', 'IX', 'UX', 'AE', 'OW', 'UW'],
            'AO': ['AA', 'OW', 'UH', 'AH', 'ER', 'AW', 'UW', 'AX', 'UX', 'AE', 'IH'],
            'AW': ['OW', 'AO', 'AH', 'UW', 'AA', 'UH', 'AX', 'ER', 'AE', 'OY'],
            'AY': ['EY', 'OY', 'IY', 'AE', 'AH', 'IH', 'AW', 'OW', 'EH', 'AX', 'UH'],
            'EH': ['AE', 'IH', 'AH', 'EY', 'IY', 'ER', 'AX', 'IX', 'UH', 'AA', 'EH'],
            'ER': ['AH', 'UH', 'AA', 'AO', 'ER', 'UX', 'AX', 'OW', 'IH', 'UW', 'IX'],
            'EY': ['IY', 'AY', 'EH', 'IH', 'AE', 'AH', 'AX', 'OY', 'UH', 'IX', 'ER'],
            'IH': ['IY', 'EH', 'AH', 'AE', 'IX', 'AX', 'ER', 'EY', 'UH', 'IH', 'AA'],
            'IY': ['IH', 'EY', 'EH', 'AY', 'AE', 'AH', 'IX', 'AX', 'UW', 'IY', 'ER', 'OY'],
            'OW': ['AO', 'UW', 'AW', 'AH', 'OY', 'AA', 'UH', 'ER', 'AX', 'OW', 'IH', 'UX'],
            'OY': ['AY', 'OW', 'AO', 'EY', 'IY', 'AW', 'UW', 'AH', 'UH', 'AA', 'OY'],
            'UH': ['UW', 'AH', 'ER', 'AO', 'OW', 'UX', 'AX', 'IH', 'AA', 'UH', 'IX', 'AW'],
            'UW': ['UH', 'OW', 'IY', 'AO', 'AW', 'UX', 'AH', 'ER', 'AA', 'UW', 'OY', 'AX'],
            'AX': ['AH', 'IH', 'UH', 'ER', 'AX', 'IX', 'UX', 'AA', 'AE', 'EH', 'AO', 'OW'],
            'IX': ['IH', 'AH', 'IY', 'AX', 'EH', 'IX', 'UX', 'ER', 'AE', 'UH', 'EY'],
            'UX': ['UH', 'AH', 'UW', 'AX', 'ER', 'UX', 'IX', 'AO', 'OW', 'IH', 'AA'],
            'B': ['P', 'V', 'D', 'M', 'W', 'G', 'B', 'F', 'DH', 'N', 'Z', 'NG'],
            'D': ['T', 'DH', 'N', 'B', 'G', 'L', 'D', 'Z', 'TH', 'R', 'V', 'P', 'DX'],
            'G': ['K', 'NG', 'D', 'B', 'G', 'GW', 'V', 'Z', 'N', 'DH', 'M', 'W'],
            'P': ['B', 'F', 'T', 'K', 'P', 'V', 'TH', 'M', 'W', 'S', 'CH'],
            'T': ['D', 'TH', 'K', 'P', 'T', 'S', 'DH', 'N', 'CH', 'TS', 'F', 'DX'],
            'K': ['G', 'T', 'P', 'NG', 'K', 'KW', 'HH', 'F', 'S', 'CH', 'TH'],
            'CH': ['SH', 'JH', 'T', 'TH', 'CH', 'S', 'TS', 'K', 'ZH', 'F', 'P'],
            'JH': ['CH', 'ZH', 'D', 'DH', 'JH', 'Z', 'DZ', 'SH', 'G', 'V', 'B'],
            'F': ['V', 'TH', 'P', 'S', 'F', 'DH', 'B', 'T', 'SH', 'HH', 'CH'],
            'V': ['F', 'DH', 'B', 'W', 'V', 'TH', 'Z', 'P', 'ZH', 'M', 'D'],
            'TH': ['F', 'T', 'S', 'DH', 'TH', 'P', 'SH', 'K', 'CH', 'D'],
            'DH': ['V', 'D', 'TH', 'Z', 'DH', 'B', 'ZH', 'F', 'T', 'N', 'L'],
            'S': ['Z', 'TH', 'SH', 'T', 'S', 'F', 'DH', 'TS', 'CH', 'K', 'P'],
            'Z': ['S', 'DH', 'ZH', 'D', 'Z', 'TH', 'DZ', 'V', 'B', 'SH', 'JH'],
            'SH': ['ZH', 'S', 'CH', 'TH', 'SH', 'F', 'Z', 'JH', 'T', 'K', 'TS'],
            'ZH': ['SH', 'Z', 'JH', 'DH', 'ZH', 'S', 'CH', 'V', 'DZ', 'D', 'TH'],
            'HH': ['F', 'K', 'P', 'HH', 'TH', 'T', 'S', 'KW', 'CH', 'SH'],
            'M': ['N', 'B', 'NG', 'W', 'M', 'P', 'V', 'EM', 'D', 'L', 'F'],
            'N': ['M', 'D', 'NG', 'L', 'N', 'T', 'EN', 'B', 'R', 'DH', 'Z', 'NX'],
            'NG': ['N', 'M', 'G', 'K', 'NG', 'ENG', 'D', 'B', 'W', 'NX'],
            'L': ['R', 'N', 'W', 'Y', 'L', 'D', 'EL', 'M', 'T', 'V', 'DH', 'B'],
            'R': ['L', 'W', 'ER', 'Y', 'R', 'N', 'D', 'V', 'UW', 'M', 'B', 'G'],
            'W': ['V', 'R', 'UW', 'L', 'W', 'OW', 'B', 'M', 'F', 'GW', 'P'],
            'Y': ['IY', 'L', 'R', 'J', 'Y', 'EY', 'W', 'N', 'IH', 'AY', 'D'],
            'TS': ['T', 'S', 'CH', 'TS', 'K', 'TH', 'SH', 'P', 'F'],
            'DZ': ['D', 'Z', 'JH', 'DZ', 'G', 'DH', 'ZH', 'B', 'V'],
            'KW': ['K', 'W', 'KW', 'G', 'P', 'T', 'GW', 'HH'],
            'GW': ['G', 'W', 'GW', 'K', 'B', 'D', 'KW', 'V'],
            'DX': ['D', 'T', 'N', 'DX', 'L', 'R', 'DH', 'TH'],
            'NX': ['N', 'NG', 'M', 'NX', 'D', 'EN', 'L', 'T'],
            'EM': ['M', 'AH', 'N', 'EM', 'B', 'NG', 'P', 'UH'],
            'EN': ['N', 'AH', 'M', 'EN', 'D', 'NG', 'T', 'IH'],
            'EL': ['L', 'AH', 'R', 'EL', 'N', 'W', 'D', 'UH'],
            'ENG': ['NG', 'AH', 'N', 'ENG', 'M', 'G', 'K', 'UH'],
            'J': ['Y', 'JH', 'G', 'J', 'D', 'Z', 'IY', 'DH'],
            'Q': ['K', 'G', 'Q', 'T', 'P', 'HH', 'KW'],
            'H': ['HH', 'F', 'K', 'H', 'P', 'TH'],
            'AXR': ['ER', 'AH', 'R', 'AXR', 'AA', 'AX', 'UH'],
            'EW': ['UW', 'IY', 'EW', 'AY', 'OY', 'IH'],
            'IW': ['IY', 'IW', 'EW', 'UW', 'Y'],
        }



    def get_phonemes(self, text: str) -> List[str]:
        try:
            phoneme_str = self.phonemizer(text, lang='en_us')
            phonemes = []
            i = 0
            while i < len(phoneme_str):
                if phoneme_str[i] == '[':
                    end = phoneme_str.find(']', i)
                    if end != -1:
                        phoneme = phoneme_str[i+1:end]
                        phonemes.append(phoneme)
                        i = end + 1
                    else:
                        i += 1
                elif phoneme_str[i] == ' ':
                    phonemes.append(' ')
                    i += 1
                else:
                    i += 1
            return phonemes
        except Exception as e:
            return []
    
    def phonemes_to_string(self, phonemes: List[str]) -> str:
        result = []
        for p in phonemes:
            if p == ' ':
                result.append(' ')
            else:
                result.append(f"[{p}]")
        return ''.join(result)
    
    def phonemes_to_plain_text(self, phonemes: List[str]) -> str:
        words = []
        current = []

        for p in phonemes:
            if p == ' ':
                if current:
                    words.append("".join(current))
                    current = []
            else:
                current.append(p)

        if current:
            words.append("".join(current))

        return " ".join(words)



    def calculate_distance(self, phonemes1: List[str], phonemes2: List[str]) -> float:
        max_len = max(len(phonemes1), len(phonemes2))
        if max_len == 0:
            return 0.0
        
        p1 = phonemes1 + [''] * (max_len - len(phonemes1))
        p2 = phonemes2 + [''] * (max_len - len(phonemes2))
        
        differences = 0
        for ph1, ph2 in zip(p1, p2):
            if ph1 == ' ' or ph2 == ' ':
                if ph1 != ph2:
                    differences += 0.3
                continue
            
            if ph1 != ph2:
                if ph1 in self.phoneme_substitutions and ph2 in self.phoneme_substitutions[ph1]:
                    differences += 1.0
                else:
                    differences += 0.8
        
        return differences / max_len
    
    def substitute_phonemes(self, phonemes: List[str], num_changes: int) -> List[str]:
        result = phonemes.copy()
        changeable = [i for i, p in enumerate(result) if p != ' ' and p in self.phoneme_substitutions]
        
        if not changeable:
            return result
        
        num_changes = min(num_changes, len(changeable))
        positions = random.sample(changeable, num_changes)
        
        for pos in positions:
            original = result[pos]
            candidates = self.phoneme_substitutions[original]
            result[pos] = random.choice(candidates)
        
        return result
    
    def generate(self, input_text: str, n: int = 10) -> List[str]:
        """
        Generate N adversarial phoneme sequences.
        
        Args:
            input_text: Original wake-word text
            n: Number of variants
            
        Returns:
            List of phoneme strings in format: "[AA][R][K][AH][S][AO][F]"
        """
        original_phonemes = self.get_phonemes(input_text)
        if not original_phonemes:
            return []
        
        original_phoneme_str = self.phonemes_to_string(original_phonemes)
        
        variants = []
        seen = set()
        seen.add(original_phoneme_str)
        
        attempts = 0
        max_attempts = n * 50
        
        num_phonemes = len([p for p in original_phonemes if p != ' '])
        
        while len(variants) < n and attempts < max_attempts:
            attempts += 1
            
            min_changes = max(1, int(num_phonemes * 0.35))
            max_changes = max(2, int(num_phonemes * 0.60))
            num_changes = random.randint(min_changes, max_changes)
            
            new_phonemes = self.substitute_phonemes(original_phonemes, num_changes)
            # new_phoneme_str = self.phonemes_to_string(new_phonemes)
            new_phoneme_str = self.phonemes_to_plain_text(new_phonemes)

            
            distance = self.calculate_distance(original_phonemes, new_phonemes)
            
            if new_phoneme_str not in seen and distance >= self.min_distance:
                variants.append(new_phoneme_str)
                seen.add(new_phoneme_str)
        
        return variants[:n]


def get_phonemizer_model(model_path: str) -> Phonemizer:
    """Download and load the phonemizer model."""
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        file_url = "https://github.com/arcosoph/phonemize/releases/download/v0.2.0/phonemize_m1.pt"
        try:
            r = requests.get(file_url, stream=True, timeout=30)
            r.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.exceptions.RequestException as e:
            print(f"ERROR downloading model: {e}")
            sys.exit(1)
    
    try:
        phonemizer = Phonemizer.from_checkpoint(model_path)
        return phonemizer
    except Exception as e:
        print(f"FATAL: Error loading phonemizer: {e}")
        sys.exit(1)


def collapse_repeated_letters(text: str) -> str:
    if not text:
        return text

    result = [text[0]]
    for ch in text[1:]:
        if ch != result[-1]:
            result.append(ch)
    return "".join(result)



# if __name__ == "__main__":
#     model_path = os.path.join(os.path.dirname(__file__), "models", "phonemize_m1.pt")
#     phonemizer = get_phonemizer_model(model_path)
#     generator = PhonemeAdversarialGenerator(phonemizer, min_distance=0.35)
    
#     input_text = input("Input: ").strip()
#     n = int(input("N: ").strip())
    
#     variants = generator.generate(input_text, n)
    
#     # collapse repeated letters
#     variants = [collapse_repeated_letters(v) for v in variants]

#     output_file = "adversarial_variants.txt"

#     with open(output_file, "w") as f:
#         f.write(f"[{', '.join(f'\"{v}\"' for v in variants)}]")

#     print(f"Saved to {output_file}")





