from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize("deconfounder/mse_causal.pyx"), include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize("deconfounder/mse_deconfound_legacy.pyx"), include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize("deconfounder/deconfound_criterion.pyx"), include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize("deconfounder/deconfound_criterion_swift.pyx"), include_dirs=[numpy.get_include()])

# To build the Cython files, run the commands below:
# python setup.py build_ext --inplace
