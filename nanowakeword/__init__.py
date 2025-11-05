import os
from nanowakeword.nanointerpreter import NanoInterpreter
from nanowakeword.vad import VAD

__all__ = ['NanoInterpreter', 'VAD']


from pathlib import Path

_INIT_PY_PATH = Path(__file__).resolve()

PROJECT_ROOT = _INIT_PY_PATH.parent
