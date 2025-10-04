# from distutils.core import setup
# from pathlib import Path

# import numpy
# from Cython.Build import cythonize

# _DIR = Path(__file__).parent

# setup(
#     name="monotonic_align",
#     ext_modules=cythonize(str(_DIR / "core.pyx")),
#     include_dirs=[numpy.get_include()],
# )

from setuptools import setup
from setuptools.extension import Extension
from pathlib import Path
import numpy
from Cython.Build import cythonize

_DIR = Path(__file__).parent

ext_modules = [
    Extension(
        name="monotonic_align",          # Name must match the import used in Python
        sources=[str(_DIR / "core.pyx")],
        include_dirs=[numpy.get_include()],
        language="c++"                  # if your core.pyx uses C++ features
    )
]

setup(
    name="monotonic_align",
    ext_modules=cythonize(ext_modules),
)
