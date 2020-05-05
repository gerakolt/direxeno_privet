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
    return F



counter=0
i=0
while counter<30:
    fig = plt.figure()
    spe=np.random.uniform(0,1)
    FS=0.5*(1+erf(0.5/(np.sqrt(2)*spe)))
    p01=1-1/FS
    print(counter)
    F=make_P(spe)
    counter+=1
    plt.plot(np.sum(F,axis=1), 'k.')
    plt.plot(np.sum(F,axis=0), 'r.')
    plt.show()
