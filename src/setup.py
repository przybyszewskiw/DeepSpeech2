from distutils.core import setup, Extension
import numpy as np

ctcbeam_module = Extension('ctcbeam', sources=['ctcbeam.cpp'])

setup(name='deep speech2',
      version='1.0',
      description='ctc beam in c++',
      install_requires = ["numpy==1.16.1"],
      include_dirs = [np.get_include()],
      ext_modules=[ctcbeam_module])
