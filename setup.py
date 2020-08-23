from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize("deconfounder/mse_causal.pyx"), include_dirs=[numpy.get_include()],
      build_dir='deconfounder')
setup(ext_modules=cythonize("deconfounder/mse_deconfound.pyx"), include_dirs=[numpy.get_include()],
      build_dir='deconfounder')

# To build the Cython files, run the commands below:
# python setup.py build_ext --build-lib=deconfounder
