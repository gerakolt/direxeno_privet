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
from mpl_toolkits.mplot3d import Axes3D

def func(x, a,b):
    return a*x+b

A=1.2
B=0.25/1000
E=np.linspace(4,100,100)*1000
a=A*(1-np.exp(-E*B))
N=0.15/13.7*(0.9+a)/(1+a)*E

p, pa=curve_fit(func, E, N)
print(p)

plt.figure()
plt.plot(E, N, 'ko')
plt.plot(np.linspace(4,100, 1000)*1000, func(np.linspace(4,100, 1000)*1000, *p), 'r--')
plt.show()
