# nanowakeword/resources/models/_registry.py

import os
from pathlib import Path

class ModelRegistry:
    def __init__(self):
        self._models_dir = Path(__file__).parent.resolve()
        self._url_map = {
            "melspectrogram.onnx": "https://github.com/arcosoph/nanowakeword/releases/download/models3/melspectrogram.onnx",
            "embedding_model.onnx": "https://github.com/arcosoph/nanowakeword/releases/download/models3/embedding_model.onnx",
            "silero_vad.onnx": "https://github.com/arcosoph/nanowakeword/releases/download/models3/silero_vad.onnx"

        }

    def _download_if_needed(self, filename: str) -> Path:
        model_path = self._models_dir / filename
        if not model_path.exists():
            if filename not in self._url_map:
                raise FileNotFoundError(f"Model '{filename}' is not a known downloadable model.")
            
            url = self._url_map[filename]
            print(f"[nanowakeword] Required model '{filename}' not found. Downloading...")
            try:
                
                from ...utils.download_file import download_file
                download_file(url, str(self._models_dir))
                print(f"[nanowakeword] Download of '{filename}' complete.")
            except Exception as e:
                raise IOError(f"Could not download required model: {filename}") from e
        
        return model_path



    def __getattr__(self, name: str) -> str:
        """
        Magic method to dynamically get model paths.
        Allows access like: models.melspectrogram_onnx
        which corresponds to the file "melspectrogram.onnx"
        """
 
        if '_' in name:
            parts = name.rsplit('_', 1)
            filename = '.'.join(parts)
        else:
            filename = name 

        try:
            model_path = self._download_if_needed(filename)
            return str(model_path)
        except FileNotFoundError:
            raise AttributeError(f"Could not find or download '{filename}'. '{self.__class__.__name__}' has no attribute '{name}'")

models = ModelRegistry()