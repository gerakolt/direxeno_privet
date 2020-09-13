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

def func(x, a,b):
    return a*np.exp(-(x-75)/b)

ts=[]
fuck=[]
plt.figure()

for a in np.linspace(0.01, 1, 10):
    fuck.append(a)
    h=0
    for i in range(5):
        t=np.random.exponential(32, 10000)
        u=np.random.uniform(size=10000)
        t+=1/a*(u/(1-u))
        h+=np.histogram(t, bins=np.arange(201))[0]/5e4
    x=np.arange(75, 200)
    p, cov=curve_fit(func, x, h[75:])
    ts.append(p[-1])
    plt.plot(x, func(x, *p), 'r--', label=p[-1])
    plt.plot(np.arange(200), h, '.', label=a)
plt.legend()

plt.figure()
plt.plot(fuck, ts, 'ko')
plt.show()
