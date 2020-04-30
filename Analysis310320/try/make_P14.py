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


def make_P(Spe, r0):
    n=100
    P=np.zeros((n,n))
    F=np.zeros((n,n))

    P[0,1:]=0.5*(1+erf((r0-np.arange(1,n))/(np.sqrt(2*np.arange(1,n))*Spe)))
    P[1,1:]=0.5*(erf((1.5-np.arange(1,n))/(np.sqrt(2*np.arange(1,n))*Spe))-erf((r0-np.arange(1,n))/(np.sqrt(2*np.arange(1,n))*Spe)))
    for i in range(2,n):
        P[i,1:]=0.5*(erf((i+0.5-np.arange(1,n))/(np.sqrt(2*np.arange(1,n))*Spe))-erf((i-0.5-np.arange(1,n))/(np.sqrt(2*np.arange(1,n))*Spe)))

    F[0,0]=1/(1+np.sum(P[0,1:]))
    F[1,0]=(1-F[0,0])/(2-F[0,0])
    F[2:,0]=F[1,0]**np.arange(2,n)

    for j in range(1,n):
        for i in range(n):
            F[i,j]=np.sum(F[:i+1,0]*np.flip(P[:i+1,j]))
    return F




fig = plt.figure()
spe=1.5
P=make_P(spe, 0.2)
plt.plot(np.sum(P, axis=1), 'k.')
plt.plot(np.sum(P, axis=0), 'r.')

# Spe=np.linspace(0,2)
# plt.figure()
# plt.plot(Spe, 0.5*(1+erf(0.5/(np.sqrt(2)*Spe))), 'k.')
# plt.axhline(0.73, xmin=0, xmax=1)
plt.show()
