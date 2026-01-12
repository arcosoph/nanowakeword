# from pathlib import Path
# from nanowakeword.utils.download_files import download_file

# class ModelRegistry:
#     def __init__(self):
#         self._models_dir = Path(__file__).parent.resolve()
#         self._url_map = {
#             "melspectrogram.onnx": "https://github.com/arcosoph/nanowakeword/releases/download/models3/melspectrogram.onnx",
#             "embedding_model.onnx": "https://github.com/arcosoph/nanowakeword/releases/download/models3/embedding_model.onnx",
#             "silero_vad.onnx": "https://github.com/arcosoph/nanowakeword/releases/download/models3/silero_vad.onnx"

#         }

#     def _download_if_needed(self, filename: str) -> Path:
#         model_path = self._models_dir / filename
#         if not model_path.exists():
#             if filename not in self._url_map:
#                 raise FileNotFoundError(f"Model '{filename}' is not a known downloadable model.")
            
#             url = self._url_map[filename]
#             print(f"[nanowakeword] Required model '{filename}' not found. Downloading...")
#             try:
#                 download_file(url, str(self._models_dir))
#                 print(f"[nanowakeword] Download of '{filename}' complete.")
#             except Exception as e:
#                 raise IOError(f"Could not download required model: {filename}") from e
        
#         return model_path



#     def __getattr__(self, name: str) -> str:
#         """
#         Magic method to dynamically get model paths.
#         Allows access like: models.melspectrogram_onnx
#         which corresponds to the file "melspectrogram.onnx"
#         """
 
#         if '_' in name:
#             parts = name.rsplit('_', 1)
#             filename = '.'.join(parts)
#         else:
#             filename = name 

#         try:
#             model_path = self._download_if_needed(filename)
#             return str(model_path)
#         except FileNotFoundError:
#             raise AttributeError(f"Could not find or download '{filename}'. '{self.__class__.__name__}' has no attribute '{name}'")

# models = ModelRegistry()




from pathlib import Path
from nanowakeword.utils.download_files import download_file


class ModelRegistry:
    """
    Central registry for NanoWakeWord model files.

    This class provides lazy access to required model files. Model paths
    are resolved dynamically using attribute access. If a requested model
    is not present locally, it is automatically downloaded and cached
    in a structured directory layout.

    Example:
        models = ModelRegistry()
        path = models.embedding_model_onnx
    """

    def __init__(self):
        """
        Initialize the model registry.

        Models are stored under:
            nanowakeword/interpreter/models/

        Each model type is placed inside its own subdirectory.
        """
        self._base_dir = (
            # Path(__file__).parent / "interpreter" / "models"
            Path(__file__).parent.resolve()
        )
        self._base_dir.mkdir(parents=True, exist_ok=True)

        self._registry = {
            "melspectrogram.onnx": {
                "subdir": "mel_spectrogram",
                "url": "https://github.com/arcosoph/nanowakeword/releases/download/models3/melspectrogram.onnx",
            },
            "embedding_model.onnx": {
                "subdir": "embedding",
                "url": "https://github.com/arcosoph/nanowakeword/releases/download/models3/embedding_model.onnx",
            },
            "silero_vad.onnx": {
                "subdir": "vad",
                "url": "https://github.com/arcosoph/nanowakeword/releases/download/models3/silero_vad.onnx",
            },
        }

    def _download_if_needed(self, filename: str) -> Path:
        """
        Ensure that the requested model file exists locally.

        If the file is not found, it is downloaded from the registered URL
        and stored in its designated subdirectory.

        Args:
            filename: Name of the model file (e.g., "embedding_model.onnx").

        Returns:
            Path to the local model file.

        Raises:
            FileNotFoundError: If the filename is not registered.
            IOError: If the file cannot be downloaded.
        """
        if filename not in self._registry:
            raise FileNotFoundError(filename)

        entry = self._registry[filename]
        model_dir = self._base_dir / entry["subdir"]
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / filename

        if not model_path.exists():
            print(f"[nanowakeword] Downloading model '{filename}'...")
            try:
                download_file(entry["url"], str(model_dir))
            except Exception as e:
                raise IOError(f"Failed to download model: {filename}") from e

        return model_path

    def __getattr__(self, name: str) -> str:
        """
        Dynamically resolve model paths via attribute access.

        Attribute names must follow the pattern:
            <model_name>_<extension>

        Example:
            models.embedding_model_onnx  -> embedding_model.onnx

        Args:
            name: Attribute name being accessed.

        Returns:
            String path to the resolved model file.

        Raises:
            AttributeError: If the model cannot be resolved or downloaded.
        """
        if "_" not in name:
            raise AttributeError(name)

        base, ext = name.rsplit("_", 1)
        filename = f"{base}.{ext}"

        try:
            return str(self._download_if_needed(filename))
        except FileNotFoundError:
            raise AttributeError(
                f"Model '{filename}' is not registered."
            )


models = ModelRegistry()
