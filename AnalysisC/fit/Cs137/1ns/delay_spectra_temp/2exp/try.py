from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf
from scipy.special import comb
from PMTgiom import make_pmts


def whichPMT(costheta, phi, pmt_mid, pmt_r, pmt_up):
    x=np.sin(np.arccos(costheta))*np.cos(phi)
    y=np.sin(np.arccos(costheta))*np.sin(phi)
    z=costheta
    R=np.vstack((x, np.vstack((y,z))))
    I=np.argmin(-np.matmul(pmt_mid, R), axis=0)
    if len(I)>0:
        Mid=pmt_mid[I]
        Right=pmt_r[I]
        Up=pmt_up[I]
        k=np.arange(len(I))
        v=Mid-(np.matmul(Mid, Mid.T)[k,k]/np.matmul(Mid, R)[k,k]*R).T
        return I[np.nonzero(np.logical_and(np.abs(np.matmul(v, Right.T)[k,k])<np.matmul(Right, Right.T)[k,k], np.abs(np.matmul(v, Up.T)[k,k])<np.matmul(Up, Up.T)[k,k]))[0]]
    else:
        return []

pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn, r=make_pmts()
PMTs=np.zeros(len(pmt_mid))
N_events=100000
for i in range(N_events):
    N=np.random.poisson(5)
    costheta=np.random.uniform(-1,1,N)
    phi=np.random.uniform(0,2*np.pi,N)
    pmts=whichPMT(costheta, phi, pmt_mid, pmt_r, pmt_up)
    I, n=np.unique(pmts, return_counts=True)
    if len(I)>0:
        PMTs[I]+=n



plt.step(np.arange(len(PMTs))-0.5, PMTs)
plt.show()
