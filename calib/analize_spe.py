import matplotlib.pyplot as plt
import os, sys
import numpy as np
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
# from fun import do_smd, do_dif, find_peaks, analize_peak
def func(x, a, tau, sigma):
    return -(a*np.exp(-0.5*np.log(x/tau)**2/sigma**2))


pmt=4
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
SPE=np.load(path+'SPEs{}.npz'.format(pmt))['SPE']
spe=np.mean(SPE, axis=0)
v=np.zeros(1000)
for i in range(1000):
    v[i]=np.sum(spe[:i])

x=np.arange(1000)
plt.figure()
plt.plot(x, spe, 'k.')
plt.plot(x, v*np.amin(spe)/np.amin(v), 'r.-')

rng=np.nonzero(np.logical_or(np.logical_and(x>204, x<230), np.logical_and(x>500, x<800)))[0]
rng1=np.nonzero(np.logical_and(x>204, x<230))[0]
rng2=np.nonzero(np.logical_and(x>400, x<900))[0]
p1, cov=curve_fit(func, x[rng1], v[rng1]*np.amin(spe)/np.amin(v))
p2, cov=curve_fit(func, x[rng2], v[rng2]*np.amin(spe)/np.amin(v))
plt.plot(x, func(x, *p1), 'y--')
plt.plot(x, func(x, *p2), 'g--')


plt.show()
