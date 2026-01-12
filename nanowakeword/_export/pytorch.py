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

# (✿◕‿◕✿)
import os
import torch
from nanowakeword.utils.logger import print_info


def export_pytorch_model(model, model_name, output_dir):
    """
    Saves the final trained PyTorch model's state_dict to a .pt file.
    This is the recommended way to save PyTorch models for robustness and portability.

    Args:
        model (torch.nn.Module): The final, trained model object (e.g., best_model).
        model_name (str): The base name for the output model file.
        output_dir (str): The directory where the model file will be saved.
    """
    model.eval()
    pytorch_path = os.path.join(output_dir, model_name + '.pt')
    
    print_info(f"Saving final PyTorch model (state_dict) to '{pytorch_path}'")
    
    try:
        torch.save(model.state_dict(), pytorch_path)
        print_info("PyTorch model saved successfully.")
        
    except Exception as e:
        print_info(f"ERROR: PyTorch model save failed. Details: {e}")
