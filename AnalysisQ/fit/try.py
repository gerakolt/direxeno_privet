import multiprocessing
import numpy as np
import sys
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf, expi
from scipy.special import comb
from scipy.signal import convolve2d
import time
from admin import make_iter
from PMTgiom import whichPMT
import matplotlib.pyplot as plt

p=np.array([[12274.0737, 10960.7316],
[12668.6168, 10522.9509],
[12722.6638, 10825.6141],
[12571.3322, 10252.7159],
[12873.9954, 10901.2799],
[12355.1442, 10620.2355],
[12738.8779, 10566.1885],
[12403.7865, 11003.9692],
[12565.9275, 11003.9692],
[12614.5698, 10679.6872],
[12614.5698, 10517.5462],
])

print(np.sqrt(np.mean(p, axis=0)))
print(np.std(p, axis=0))
