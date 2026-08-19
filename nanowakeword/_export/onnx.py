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
import math
import torch
import torchaudio
from nanowakeword.utils.logger import print_info, print_error

class ONNXSafeMelSpectrogram(torch.nn.Module):
    def __init__(self, original_mel):
        super().__init__()
        spectrogram = original_mel.spectrogram
        self.n_fft = spectrogram.n_fft
        self.hop_length = spectrogram.hop_length
        self.win_length = spectrogram.win_length
        self.center = spectrogram.center
        self.pad_mode = spectrogram.pad_mode
        self.power = spectrogram.power
        
        window = spectrogram.window
        if window is None:
            window = torch.ones(self.win_length)
            
        real_fourier_basis = torch.zeros(self.n_fft, self.n_fft)
        imag_fourier_basis = torch.zeros(self.n_fft, self.n_fft)
        
        for k in range(self.n_fft):
            for n in range(self.n_fft):
                angle = -2 * math.pi * k * n / self.n_fft
                real_fourier_basis[k, n] = math.cos(angle)
                imag_fourier_basis[k, n] = math.sin(angle)
                
        window_padded = torch.zeros(self.n_fft)
        pad_left = (self.n_fft - self.win_length) // 2
        window_padded[pad_left : pad_left + self.win_length] = window
        
        real_fourier_basis = real_fourier_basis * window_padded.unsqueeze(0)
        imag_fourier_basis = imag_fourier_basis * window_padded.unsqueeze(0)
        
        n_bins = self.n_fft // 2 + 1
        real_basis = real_fourier_basis[:n_bins, :].unsqueeze(1).float()
        imag_basis = imag_fourier_basis[:n_bins, :].unsqueeze(1).float()
        
        self.register_buffer('real_basis', real_basis)
        self.register_buffer('imag_basis', imag_basis)
        self.register_buffer('mel_fb', original_mel.mel_scale.fb.float())
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        if self.center:
            pad_amount = self.n_fft // 2
            x = torch.nn.functional.pad(x, (pad_amount, pad_amount), mode=self.pad_mode)
        
        real_part = torch.nn.functional.conv1d(x, self.real_basis, stride=self.hop_length)
        imag_part = torch.nn.functional.conv1d(x, self.imag_basis, stride=self.hop_length)
        
        power_spec = real_part**2 + imag_part**2
        if self.power is not None and self.power == 1.0:
            power_spec = torch.sqrt(power_spec + 1e-9)
            
        power_spec = power_spec.transpose(1, 2)
        mel_spec = torch.matmul(power_spec, self.mel_fb).transpose(1, 2)
        return mel_spec


def replace_mel_spectrogram(module):
    """Recursively replaces torchaudio MelSpectrogram with the ONNX-Safe version."""
    for name, child in module.named_children():
        if isinstance(child, torchaudio.transforms.MelSpectrogram):
            setattr(module, name, ONNXSafeMelSpectrogram(child))
            print_info(f"  -> Successfully patched '{name}' with ONNX-safe Convolutional STFT.")
        else:
            replace_mel_spectrogram(child)


def make_onnx_safe_adaptive_pool(model, dummy_input):
    """
    Auto Shape Detective:
    Runs a dummy input to find the exact shape hitting the AdaptiveAvgPool2d,
    calculates the exact mathematical equivalent for ONNX, and swaps it out!
    """
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.AdaptiveAvgPool2d):
            targets.append((name, module))
            
    if not targets:
        return

    print_info("Scanning model to patch unsupported AdaptiveAvgPool2d layers...")
    
    shapes = {}
    def hook_fn(name):
        def hook(module, inp, out):
            shapes[name] = inp[0].shape
        return hook
        
    hooks = []
    for name, mod in targets:
        hooks.append(mod.register_forward_hook(hook_fn(name)))
        
    with torch.no_grad():
        model(dummy_input)
        
    for h in hooks:
        h.remove()
        
    def set_module(model, name, new_module):
        parts = name.split('.')
        target = model
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], new_module)

    for name, mod in targets:
        if name in shapes:
            in_shape = shapes[name]
            in_h, in_w = in_shape[2], in_shape[3]
            
            out_size = mod.output_size
            if isinstance(out_size, int):
                out_h, out_w = out_size, out_size
            else:
                out_h, out_w = out_size
                
            stride_h = in_h // out_h
            kernel_h = in_h - (out_h - 1) * stride_h
            
            stride_w = in_w // out_w
            kernel_w = in_w - (out_w - 1) * stride_w
            
            safe_pool = torch.nn.AvgPool2d(kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))
            set_module(model, name, safe_pool)
            print_info(f"  -> Successfully patched '{name}' with ONNX-safe AvgPool2d(kernel={(kernel_h, kernel_w)}, stride={(stride_h, stride_w)})")


def export_onnx_model(model, input_shape, config, model_name, output_dir):
    is_e2e = getattr(model, 'is_e2e', False)
    
    if is_e2e:
        print_info("Initializing E2E ONNX Exporter optimizations...")
        replace_mel_spectrogram(model)

    class InferenceWrapper(torch.nn.Module):
        def __init__(self, trained_model):
            super().__init__()
            self.trained_model = trained_model

        def forward(self, x):
            logits = self.trained_model(x)
            probabilities = torch.sigmoid(logits)
            return probabilities.view(-1, 1, 1)

    try:
        exportable_model = InferenceWrapper(model)
        exportable_model.eval()

        if is_e2e:
            if len(input_shape) == 1:
                dummy_input = torch.randn(1, 1, input_shape[0], device='cpu', dtype=torch.float32)
            else:
                dummy_input = torch.randn(1, *input_shape, device='cpu', dtype=torch.float32)
        else:
            dummy_input = torch.randn(1, *input_shape, device='cpu', dtype=torch.float32)

        # Apply our brilliant Adaptive Pool Fix dynamically!
        make_onnx_safe_adaptive_pool(exportable_model, dummy_input)

        onnx_path = os.path.join(output_dir, model_name + '.onnx')

        print_info(f"Saving inference-ready ONNX model to '{onnx_path}'")
        if is_e2e:
            print_info(f"  E2E mode: input shape {input_shape}")

        x_export = config.get("custom_export") or config.get("export_model") or {}
        opset_version = x_export.get("onnx_opset_version", config.get("onnx_opset_version", 17))
        print_info(f"Using ONNX opset version: {opset_version}")

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

        if is_e2e:
            try:
                import onnx
                proto = onnx.load(onnx_path)
                from onnx import helper
                mode_prop = helper.make_string_string_entry("mode", "e2e")
                proto.metadata_props.append(mode_prop)
                onnx.save(proto, onnx_path)
            except Exception:
                pass

        print_info("ONNX model exported successfully!")

    except Exception as e:
        print_error("ONNX export failed. Fix the issue and run again with --resume if a checkpoint exists.")
        print_info(f"   Details: {e}")