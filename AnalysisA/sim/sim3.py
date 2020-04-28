from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf
from scipy.special import comb

def make_data(N_events, NQ, R, F, Tf, Ts, Spe, s_pad, St, T, Strig):
    d=np.zeros((N_events, 1000, len(NQ)))
    for i in range(N_events):
        t0=np.zeros(len(NQ))
        for j in range(len(NQ)):
            n=np.random.poisson(NQ)
            ch=np.random.choice(3, size=n, replace=True, p=[R[j], (1-R[j])*F[j], (1-R[j])*(1-F[j])])
            nd=len(np.nonzero(ch==0)[0])
            nf=len(np.nonzero(ch==1)[0])
            ns=len(np.nonzero(ch==2)[0])
            trig=np.random.normal(T[j]*5, Strig[j]*5, 1)
            tf=np.random.normal(trig+np.random.exponential(Tf[j]*5, nf), St[j]*5, nf)
            ts=np.random.normal(trig+np.random.exponential(Ts[j]*5, ns), St[j]*5, ns)
            td=np.random.normal(trig, St[j]*5, nd)
            t=np.append(td, np.append(ts, tf))
            h, bins=np.histogram(t, bins=np.arange(1001)-0.5)
            t0[j]=np.amin(np.nonzero(h>0)[0])
            d[i,:,j]=h
        for j in range(len(NQ)):
            d[i,:,j]=np.roll(d[i,:,j], -int(np.amin(t0)))
    return d


D=make_data(10000, [34.64089111, 34.58835435], [0,0], [0.22970467, 0.04599156], [14.2135601,  0.1520107], [44.51383159, 39.81847958], [0.3, 0.3], [0.3, 0.3], [0.41830008, 0.97103924], [38.76175882, 39.30330286], [5, 5])

x=np.arange(1000)/5
fig, (ax1, ax2)=plt.subplots(2,1, sharex=True)

ax1.plot(x, np.mean(D[:,:,0].T*np.arange(np.shape(D)[0]), axis=1), 'k.', label='sim')

ax2.plot(x, np.mean(D[:,:,1].T*np.arange(np.shape(D)[0]), axis=1), 'k.', label='sim')


plt.subplots_adjust(hspace=0)
plt.show()
