# to run this example execute:
# $ python setup.py install
import torch
import ctcbeam
import numpy as np

a = torch.load(f='t.pt', map_location='cpu')
#aa = a.detach().numpy();
aa = a.tolist();
#aa = [[1.0, 2.0], [3.0, 4.0]]
#print(type(aa))
print(ctcbeam.ctcbeam(aa, "ngrams.txt"))
