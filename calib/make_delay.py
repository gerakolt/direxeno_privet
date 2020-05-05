import matplotlib.pyplot as plt
import os, sys
import numpy as np
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from fun import do_smd, do_dif, find_peaks, analize_peaks
########## This is a develop branch#########

def func(X, a,b,c):
    y=np.zeros(len(X))
    dx=X[1]-X[0]
    for i, x in enumerate(X):
        r=np.linspace(x-0.5*dx, x+0.5*dx, 100)
        y[i]=np.sum(a*np.exp(-0.5*(r-b)**2/c**2))*(r[1]-r[0])
    return y

path='/home/gerak/Desktop/DireXeno/190803/pulser/'
pmts=np.array([0,1,4,7,8,11])
for pmt1 in pmts:
    SPE1=np.load(path+'PMT{}/SPEs{}.npz'.format(pmt1,pmt1))['rec']
    for pmt2 in pmts[pmts>pmt1]:
        SPE2=np.load(path+'PMT{}/SPEs{}.npz'.format(pmt2,pmt2))['rec']
        ids1=SPE1['id']
        ids2=SPE2['id']
        t1=SPE1['init10']
        t2=SPE2['init10']

        ids=ids1[np.nonzero(np.isin(ids1, ids2))[0]]
        dt=np.zeros(len(ids))
        for i, id in enumerate(ids):
            print(pmt1, pmt2, i, 'out of', len(ids))
            dt[i]=(t1[ids1==id]-t2[ids2==id])/5
        plt.figure()
        plt.title('PMT{}-PMT{}'.format(pmt1, pmt2))
        h, bins, patch=plt.hist(dt, bins=np.arange(-10, 10))
        x=0.5*(bins[1:]+bins[:-1])
        p0=[np.amax(h), x[np.argmax(h)], 1]
        p, cov=curve_fit(func, x, h, p0=p0)
        plt.plot(x, func(x, *p), 'r.-', label='{}+-{}'.format(p[1], p[2]))
        plt.legend()
        np.savez(path+'Delays/delay_{}_{}'.format(pmt1,pmt2), h_delay=h, bins_delay=bins, M=p[1], S=p[2])
plt.show()
