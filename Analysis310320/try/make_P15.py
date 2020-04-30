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


r0=[]
SPE=[]
for spe in np.linspace(1.1804, 1.181, 500):
    def L(*p):
        P=make_P(spe, p[0])
        print(spe, p[0], np.abs(np.sum(P, axis=1)[1]-1))
        return np.abs(np.sum(P, axis=1)[1]-1)

    p=0.1
    p=minimize(L, p, method='Nelder-Mead', options={'disp':True, 'maxfev':1000})
    r0.append(p.x[0])
    SPE.append(spe)
np.savez('fit4', spe=SPE, r0=r0)
