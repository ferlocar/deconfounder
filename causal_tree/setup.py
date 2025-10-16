from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy


extensions = [
    Extension("causal_residual_mse", ["causal_residual_mse.pyx"], include_dirs=[numpy.get_include()]),
]

setup(
    ext_modules=cythonize(extensions)
)

# To build the Cython files, run the commands below:
# python setup.py build_ext --inplace
