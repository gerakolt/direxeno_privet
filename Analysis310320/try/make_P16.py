import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from scipy.stats import poisson, binom
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.special import erf as erf
import sys
import random
from datetime import datetime
random.seed(datetime.now())


def make_P(Spe, r0, Q):
    n=100
    P=np.zeros((n,n))
    F=np.zeros((n,n))

    P[0,1:]=0.5*(1+erf((r0-np.arange(1,n))/(np.sqrt(2*np.arange(1,n))*Spe)))
    P[1,1:]=0.5*(erf((1.5-np.arange(1,n))/(np.sqrt(2*np.arange(1,n))*Spe))-erf((r0-np.arange(1,n))/(np.sqrt(2*np.arange(1,n))*Spe)))
    for i in range(2,n):
        P[i,1:]=0.5*(erf((i+0.5-np.arange(1,n))/(np.sqrt(2*np.arange(1,n))*Spe))-erf((i-0.5-np.arange(1,n))/(np.sqrt(2*np.arange(1,n))*Spe)))

    F[1:,0]=binom.pmf(np.arange(1,n), np.arange(1,n), Q)
    F[1:,0]=F[1:,0]/np.sum(F[1:,0])

    for j in range(1,n):
        for i in range(1, n):
            F[i,j]=np.sum(binom.pmf(np.arange(i+1), i, Q)*np.flip(P[:i+1,j]))
        F[:,j]=F[:,j]/np.sum(F[:,j])
    return F

P=make_P(0.5, 0.1, 0.1)

plt.plot(np.sum(P, axis=1), 'k.')
plt.plot(np.sum(P, axis=0), 'r.')
plt.show()

plt.plot(P[:,1], 'k.')
plt.plot(np.cumsum(P[:,1]), 'r.')
plt.axhline(1, xmin=0, xmax=1)
plt.show()
