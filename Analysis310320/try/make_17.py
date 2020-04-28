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

def func(x, m, s):
    dx=x[1]-x[0]
    y=[]
    for i in x:
        r=np.linspace(i-0.5*dx, i+0.5*dx, 100)
        dr=r[1]-r[0]
        y.append(np.sum(dr*np.exp(-0.5*(r-m)**2/s**2)/np.sqrt(2*np.pi*s**2)))
    return y

# def make_P(Spe, s_pad, r0, Q):
#     n=100
#     P=np.zeros((n,n))
#     F=np.zeros((n,n))
#
#     P[0,1:]=0.5*(1+erf((r0-np.arange(1,n))/(np.sqrt(2*(np.arange(1,n)*Spe**2+s_pad**2)))))
#     P[1,1:]=0.5*(erf((1.5-np.arange(1,n))/(np.sqrt(2*(np.arange(1,n)*Spe**2+s_pad**2))))-erf((r0-np.arange(1,n))/(np.sqrt(2*(np.arange(1,n)*Spe**2+s_pad**2)))))
#     for i in range(2,n):
#         P[i,1:]=0.5*(erf((i+0.5-np.arange(1,n))/(np.sqrt(2*(np.arange(1,n)*Spe**2+s_pad**2))))-erf((i-0.5-np.arange(1,n))/(np.sqrt(2*(np.arange(1,n)*Spe**2+s_pad**2)))))
#
#     F[1:,0]=binom.pmf(np.arange(1,n), np.arange(1,n), Q)
#     for j in range(1,n):
#         for i in range(1,n):
#             F[i,j]=np.sum(binom.pmf(np.arange(i+1), i, Q)*np.flip(P[:i+1,j]))
#     F[0,0]=1-np.sum(F[1:,0])
#     F[0,1:]=F[0,0]*P[0,1:]
#     for j in range(n):
#         F[:,j]=F[:,j]/np.sum(F[:,j])
#     return F

def make_P(Spe, s_pad, r0, Q):
    n=100
    P=np.zeros((n,n))
    F=np.zeros((n,n))

    P[0,1:]=0.5*(1+erf((r0-np.arange(1,n))/(np.sqrt(2*(np.arange(1,n)*Spe**2+s_pad**2)))))
    P[1,1:]=0.5*(erf((1.5-np.arange(1,n))/(np.sqrt(2*(np.arange(1,n)*Spe**2+s_pad**2))))-erf((r0-np.arange(1,n))/(np.sqrt(2*(np.arange(1,n)*Spe**2+s_pad**2)))))
    for i in range(2,n):
        P[i,1:]=0.5*(erf((i+0.5-np.arange(1,n))/(np.sqrt(2*(np.arange(1,n)*Spe**2+s_pad**2))))-erf((i-0.5-np.arange(1,n))/(np.sqrt(2*(np.arange(1,n)*Spe**2+s_pad**2)))))

    F[1:,0]=binom.pmf(np.arange(1,n), np.arange(1,n), Q)
    for j in range(1,n):
        for i in range(1,n):
            F[i,j]=np.sum(binom.pmf(np.arange(i+1), i, Q)*np.flip(P[:i+1,j]))
    F[0,0]=1-np.sum(F[1:,0])
    F[0,1:]=F[0,0]*P[0,1:]
    for j in range(n):
        F[:,j]=F[:,j]/np.sum(F[:,j])
    return F

P=make_P(0.15, 0.3, 0, 0.01)

ns=np.arange(100)
p0=[50, 25]
p, cov=curve_fit(func, ns, P[:,36], p0=p0)
plt.plot(P[:,36], 'k.', label='{}'.format(np.sqrt(51*0.3**2)))
plt.plot(ns, func(ns, *p), '--', label='{}, {}'.format(p[0], p[1]))
plt.legend()
plt.show()
