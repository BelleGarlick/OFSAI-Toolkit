import numpy
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("fsai/geometry.pyx"),
    include_dirs=[numpy.get_include()]
)

# python3 setup.py build_ext --inplace