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


def make_P(Spe):
    n=100
    P=np.zeros((n,n))
    F=np.zeros((n,n))
    S=0.5*(1+erf(0.5/(np.sqrt(2)*Spe)))
    P[0,1]=1/S-1
    P[1,1]=(1-P[0,1]-P[0,1]**2)/(1+P[0,1])
    for i in np.arange(2, n):
        P[i,1]=0.5*(erf((i-0.5)/(np.sqrt(2)*Spe))-erf((i-1.5)/(np.sqrt(2)*Spe)))
    for i in range(n):
        for j in range(2,n):
            P[i,j]=np.sum(P[:i+1,1]*np.flip(P[:i+1,j-1]))
    F[0,0]=1-P[0,1]
    F[1,0]=P[0,1]/(1+P[0,1])
    F[2:,0]=F[1,0]**np.arange(2,n)

    for j in range(1,n):
        for i in range(n):
            F[i,j]=np.sum(F[:i+1,0]*np.flip(P[:i+1,j]))
    return F, P




fig = plt.figure()
spe=np.sqrt(1.507**2+0.537**2)
spe=0.5
P, P0=make_P(spe)
plt.plot(P0[:10,1], 'k.')
plt.plot(P[:10,1], 'r.')

# Spe=np.linspace(0,2)
# plt.figure()
# plt.plot(Spe, 0.5*(1+erf(0.5/(np.sqrt(2)*Spe))), 'k.')
# plt.axhline(0.73, xmin=0, xmax=1)
plt.show()
