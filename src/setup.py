from distutils.core import setup, Extension
import numpy as np

ctcbeam_module = Extension('ctcbeam', sources=['ctcwrapper.cpp'], extra_compile_args=['-std=c++11'])

setup(name='deep speech2',
      version='1.0',
      description='ctc beam in c++',
      include_dirs = [],
      ext_modules=[ctcbeam_module])
