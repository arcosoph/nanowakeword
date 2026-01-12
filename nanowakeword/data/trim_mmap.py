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
import numpy as np
from tqdm import tqdm
from numpy.lib.format import open_memmap


def trim_mmap(target_path):
    """
    Removes trailing zero-filled rows from a .npy mmap file.
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
