"""User-configurable export helper.

Provides `export_custom_model` which trainer.py calls after built-in exports.

The helper supports two modes configured via the training config:
- `script`: path to a Python file. A callable (default `export_model`) will be imported and invoked.
- `command`: a shell command string. It supports `.format()` placeholders: `{model_path}`, `{model_name}`, `{output_dir}`.

"""
import os
import importlib.util
import subprocess
from typing import Any, Tuple

from nanowakeword.utils.logger import print_info, print_warning


def export_custom_model(model: Any, input_shape: Tuple[int, ...], config: dict, model_name: str, output_dir: str) -> bool:
    export_cfg = config.get("custom_export") or config.get("export_model") or {}
    if not export_cfg:
        return False

    # Python script execution (recommended)
    script_path = export_cfg.get("script")
    func_name = export_cfg.get("function", "export_model")
    if script_path:
        try:
            spec = importlib.util.spec_from_file_location("user_export_module", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            func = getattr(module, func_name, None)
            if not func:
                print_warning(f"User export script '{script_path}' has no function '{func_name}'. Skipping.")
                return False

            try:
                func(model=model, input_shape=input_shape, config=config, model_name=model_name, output_dir=output_dir)
            except TypeError:
                func(model, input_shape, config, model_name, output_dir)

            print_info(f"User export script '{script_path}' executed successfully.")
            return True
        except Exception as e:
            print_warning(f"User export script failed: {e}")

    # Shell command fallback
    cmd = export_cfg.get("command")
    if cmd:
        try:
            model_path = os.path.join(output_dir, model_name + ".onnx")
            formatted = cmd.format(model_path=model_path, model_name=model_name, output_dir=output_dir)
            subprocess.run(formatted, shell=True, check=True)
            print_info("User export command executed successfully.")
            return True
        except Exception as e:
            print_warning(f"User export command failed: {e}")

    return False
