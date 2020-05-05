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
    n=150
    P=np.zeros((n,n))
    F=np.zeros((n,n))
    FS=0.5*(1+erf(0.5/(np.sqrt(2)*Spe)))
    p01=1-1/FS
    P[0,1]=p01
    P[1,1]=FS-p01

    for i in np.arange(2, n):
        P[i,1]=0.5*(erf((i-0.5)/(np.sqrt(2)*Spe))-erf((i-1.5)/(np.sqrt(2)*Spe)))

    for i in range(n):
        for j in range(2,n):
            P[i,j]=np.sum(P[:i+1,1]*np.flip(P[:i+1,j-1], axis=0))

    p00=1/(1+(p01-p01**n)/(1-p01))
    F[0,0]=p00
    F[1:,0]=((1-p00)/(2-p00))**(np.arange(1, n))
    for j in range(1,n):
        for i in range(n):
            F[i,j]=np.sum(F[:i+1,0]*np.flip(P[:i+1,j]))

    return F



fig = plt.figure()
counter=0
i=0
while counter<50:
    i+=1
    spe=np.random.uniform(0,1)
    FS=0.5*(1+erf(0.5/(np.sqrt(2)*spe)))
    p01=1-1/FS
    print(i, counter)
    P=make_P(spe)
    # if np.sum(P,axis=1)[1]>0.999 and np.sum(P,axis=1)[1]<1.001:
    counter+=1
    plt.scatter(spe, np.sum(P, axis=1)[1], color='k')

plt.show()
