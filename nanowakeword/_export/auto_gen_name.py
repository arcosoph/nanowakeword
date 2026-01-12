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


def auto_gen_name(model_type: str, base_dir: str = ".", prefix: str = "nww"):
    """
    Automatically generate model name including type and maintain version professionally
    The version option will be kept the same, meaning if there is an older model, it will be overwritten.
    
    Example:
        nww_dnn_model_v1  → nww_dnn_model_v2  (if v1 already exists)
    Args:
        model_type (str): Type of the model (e.g. 'dnn', 'cnn', 'lstm').
        base_dir (str): Directory where model files are stored.
        prefix (str): Prefix used before model name (default 'nww').
    
    Returns:
        str: A unique, versioned model name.
    """
    import os
    import re
    # Normalize
    model_type = model_type.lower().strip()
    pattern = re.compile(rf"^{prefix}_{model_type}_model_v(\d+)$")

    # Find existing models in the directory
    existing = []
    for name in os.listdir(base_dir):
        match = pattern.match(name)
        if match:
            existing.append(int(match.group(1)))

    # Determine next version
    next_version = max(existing, default=0) + 1
    return f"{prefix}_{model_type}_model_v{next_version}"
