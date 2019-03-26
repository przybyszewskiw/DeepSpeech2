# to run this example execute:
# $ python setup.py install
import numpy as np
import ctcbeam
a = np.random.rand(3, 4)
print("python here:")
print(a)
ctcbeam.ctcbeam(a)
