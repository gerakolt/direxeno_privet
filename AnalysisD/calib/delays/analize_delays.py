import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import os
import sys


path='/home/gerak/Desktop/DireXeno/190803/pulser/DelayRecon/'
data=np.load(path+'delays_lists.npz')
dT=data['dT']
dT10=data['dT10']
names=data['names']


def func(x, a,m,s):
    dx=x[1]-x[0]
    return a*np.exp(-0.5*(x-m)**2/s**2)
for i, dt in enumerate(dT):
    h, bins=np.histogram(dt, bins=np.arange(20)-10)
    x=0.5*(bins[1:]+bins[:-1])
    p=[np.amax(h), x[np.argmax(h)], 1]
    rng=np.nonzero(np.logical_and(x>x[np.argmax(h)]-3, x<x[np.argmax(h)]+3))
    p, cov=curve_fit(func, x[rng], h[rng], p0=p)
    plt.figure()
    plt.plot(np.linspace(x[rng][0], x[rng][-1], 100), func(np.linspace(x[rng][0], x[rng][-1], 100), *p), 'r-.')
    plt.bar(x, h, label=names[i]+', m={}'.format(p[-2]))
    plt.hist(dT10[i], bins=np.arange(20)-10, histtype='step', label='init10', color='g', linewidth=5)
    plt.legend()
    plt.show()
    np.savez(path+'delay_hist{}'.format(names[i]), h=h, x=x, a=p[0], m=p[-2], s=p[-1])
