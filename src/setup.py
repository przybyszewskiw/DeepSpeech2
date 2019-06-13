from distutils.core import setup, Extension
import numpy as np
import glob

#print(glob.glob(["kenlm/lm/*.cc", "kenlm/util/*.cc", "kenlm/util/double-conversion/*.cc",
#                                 "kenlm/lm/*.hh", "kenlm/util/*.hh", "kenlm/util/double-conversion/*.h"]))

source_list = glob.glob("kenlm/lm/*.cc", recursive=True) + glob.glob("kenlm/util/*.cc", recursive=True) + glob.glob("kenlm/util/double-conversion/*.cc", recursive=True)
excludes = glob.glob("kenlm/*/*test.cc", recursive=True) + glob.glob("kenlm/*/*main.cc", recursive=True)
sources = [elem for elem in source_list if elem not in excludes]
sources += ["ctcwrapper.cpp"]


ctcbeam_module = Extension('ctcbeam',
                           sources=sources,
                           extra_compile_args=['-std=c++11', '-DKENLM_MAX_ORDER=6', '-Ikenlm'])

setup(name='deep speech2',
      version='1.0',
      description='ctc beam in c++',
      include_dirs = [],
      ext_modules=[ctcbeam_module])
