from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

setup(ext_modules=cythonize("deconfounder/mse_causal.pyx"), include_dirs=[numpy.get_include()])

setup(ext_modules=cythonize("deconfounder/deconfound_mse.pyx"), include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize("deconfounder/deconfound_classification.pyx"), include_dirs=[numpy.get_include()])
extensions = [
    Extension("deconfound_auuc", ["deconfounder/deconfound_auuc.pyx"], include_dirs=[numpy.get_include()], language="c++")
]
setup(ext_modules=cythonize(extensions))

# To build the Cython files, run the commands below:
# python setup.py build_ext --inplace
