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


def export_onnx_model(model, input_shape, config, model_name, output_dir):
    """
    Exports the final trained model to a standard, inference-ready ONNX format.

    This function ensures hardware independence by moving both the model and a
    dummy input to the CPU before export. It also guarantees a standardized
    output shape of [batch_size, 1, 1] for maximum compatibility.
    """
    # A robust wrapper to apply sigmoid and ensure the final output shape
    class InferenceWrapper(torch.nn.Module):
        def __init__(self, trained_model):
            super().__init__()
            self.trained_model = trained_model

        def forward(self, x):
            logits = self.trained_model(x)
            probabilities = torch.sigmoid(logits)
            # Forcefully reshape the output to a standard 3D tensor
            return probabilities.view(-1, 1, 1)

    exportable_model = InferenceWrapper(model)
    exportable_model.eval()

    # Define a dummy input for tracing the model graph
    # dummy_input = torch.rand(1, *self.input_shape)
    # New line (Ensures float32 and correct device)
    dummy_input = torch.randn(1, *input_shape, device='cpu', dtype=torch.float32)
    
    onnx_path = os.path.join(output_dir, model_name + '.onnx')
    
    print_info(f"Saving inference-ready ONNX model to '{onnx_path}'")
    
    opset_version = config.get("onnx_opset_version", 17)
    print_info(f"Using ONNX opset version: {opset_version}")

    try:
        # For maximum compatibility and to prevent device errors, always move both
        # the model and the dummy input to the CPU before exporting to ONNX.
        
        model_cpu = exportable_model.cpu()
        dummy_input_cpu = dummy_input.cpu()
        
        torch.onnx.export(
            model_cpu,
            dummy_input_cpu,
            onnx_path,
            opset_version=opset_version,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print_info("ONNX model saved successfully.")

    except Exception as e:
        # Provide a more detailed error message
        print_info("ERROR: ONNX export failed. Fix the issue and run again with --resume if a checkpoint exists.")
        print_info(f"   Details: {e}")

