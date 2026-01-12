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


import requests
from tqdm import tqdm
import os

def download_file(url, target_directory, file_size=None):
    """A simple function to download a file from a URL with a progress bar using only the requests library"""
    local_filename = url.split('/')[-1]

    with requests.get(url, stream=True) as r:
        if file_size is not None:
            progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True, desc=f"{local_filename}")
        else:
            total_size = int(r.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"{local_filename}")

        with open(os.path.join(target_directory, local_filename), 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                progress_bar.update(len(chunk))

    progress_bar.close()
