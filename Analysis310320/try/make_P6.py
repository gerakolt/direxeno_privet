import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import poisson, binom
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.special import erf as erf
import sys
import random
from datetime import datetime
random.seed(datetime.now())


def make_P(Spe, p01, p00):
    n=100
    P=np.zeros((n,n))
    F=np.zeros((n,n))
    P[0,1]=p01
    P[1,1]=0.5*(1+erf(0.5/(np.sqrt(2)*Spe)))-p01

    for i in np.arange(2, n):
        P[i,1]=0.5*(erf((i-0.5)/(np.sqrt(2)*Spe))-erf((i-1.5)/(np.sqrt(2)*Spe)))

    for i in range(n):
        for j in range(2,n):
            P[i,j]=np.sum(P[:i+1,1]*np.flip(P[:i+1,j-1], axis=0))

    p00=1/(1+(p01-p01**n)/(1-p01))
    print(p00)
    F[0,0]=p00
    F[1:,0]=((1-p00)/(2-p00))**(np.arange(1, n))
    for j in range(1,n):
        for i in range(n):
            F[i,j]=np.sum(F[:i+1,0]*np.flip(P[:i+1,j]))

    return F


p01=0.25
p00=0.9
r=(p01-p01**100)/(1-p01)

P=make_P(0.8, p01, p00)
plt.plot(np.sum(P, axis=0), '.', label='axis 0')
plt.plot(np.sum(P, axis=1), '.', label='axis 1')
# plt.plot(np.arange(100), (np.log(P[0,:])-np.log(p00))/np.log(p01), 'ko')
# plt.plot(np.arange(100), np.arange(100), 'r.-')
plt.axhline(1, xmin=0, xmax=1)

plt.legend()
# plt.yscale('log')
plt.show()
