import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(X, a, m,s):
    y=np.zeros(len(X))
    dx=X[1]-X[0]
    for i, x in enumerate(X):
        r=np.linspace(x-0.5*dx,x+0.5*dx,100)
        dr=r[1]-r[0]
        y[i]=np.sum(a*np.exp(-0.5*(r-m)**2/s**2))*dr
    return y

path='/home/gerak/Desktop/DireXeno/190803/pulser/delays/'
pmts=[0,7,8]
for i, pmt1 in enumerate(pmts):
    for j, pmt2 in enumerate(pmts):
        if pmt2>pmt1:
            ts=np.load(path+'ts_{}_{}.npz'.format(pmt1, pmt2))['ts']
            plt.figure()
            plt.title('{}-{}'.format(pmt1, pmt2))
            h,bins, pa=plt.hist(ts, bins=np.arange(-25,26)/5-0.5)
            t=0.5*(bins[1:]+bins[:-1])
            m=np.sum(h*t)/np.sum(h)
            rng=np.nonzero(np.logical_and(t>m-2, t<m+2))
            p0=[np.amax(h), m, 1]
            p,cov=curve_fit(func, t[rng], h[rng], p0=p0)
            plt.plot(t[rng], func(t[rng], *p), 'r.-', label='{},{}'.format(p[-1], p[-2]))
            plt.legend()
            print(int(np.round(p[-2]*5)))
            np.savez(path+'delays_{}_{}'.format(pmt1, pmt2), delays=t-p[-2], h_delays=h, delay=p[-2])
plt.show()
