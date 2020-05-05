import matplotlib.pyplot as plt
import os, sys
import numpy as np
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
import itertools

def func(x, a,b,c):
    return a*np.exp(-0.5*(x-b)**2/c**2)

path='/home/gerak/Desktop/DireXeno/190803/pulser/'
pmts=[0,1,2,3,4,5,6,7,8,9,10,11,17,14,15,16,18,19]
D=np.load(path+'Delay3DTable.npz')['D']
S=np.zeros((len(pmts), len(pmts)))
Chi2=np.zeros((len(pmts), len(pmts)))

for i in range(len(pmts)-1):
    for j in range(i+1, len(pmts)):
        d=D[i,j,:]
        plt.figure()
        plt.title('PMT{}-PMT{}'.format(pmts[i],pmts[j]))
        h, bins, patch=plt.hist(d[np.nonzero(np.abs(d)>0)[0]], bins=np.linspace(-30,30,30))
        x=0.5*(bins[1:]+bins[:-1])
        rng=np.nonzero(np.logical_and(h>0, np.logical_and(x>x[np.argmax(h)]-3*5, x<x[np.argmax(h)]+3*5)))[0]
        h=h[rng]
        x=x[rng]
        if len(h)>0:
            p0=[np.amax(h), x[np.argmax(h)], 0.5*np.abs(x[np.argmax(h)])]
            try:
                p, cov=curve_fit(func, x, h, p0=p0)
                r=np.linspace(np.amin(x), np.amax(x), 100)
                if p[2]<0.5:
                    p[1]=np.average(x, weights=h)
                    p[2]=np.sqrt(np.average((x-p[1])**2, weights=h))
                    p[0]=h[np.argmin(np.abs(x-p[1]))]
                    print(pmts[i], pmts[j])
                plt.plot(r, func(r, *p), 'r--', label='{}+-{}'.format(p[1]/5, p[2]/5))
                plt.legend()
                S[i,j]=p[2]/5
            except:
                print('no fit')
np.savez(path+'Sigma_table', S=S)
plt.show()
