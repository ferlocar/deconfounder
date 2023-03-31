from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize("deconfounder/mse_causal.pyx", language_level=3), include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize("deconfounder/mse_deconfound.pyx", language_level=3), include_dirs=[numpy.get_include()])

# To build the Cython files, run the commands below:
# python setup.py build_ext --inplace
