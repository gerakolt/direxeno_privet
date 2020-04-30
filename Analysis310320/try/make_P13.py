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



SPE=[]
P_1_0=[]
P_0_1=[]
P=[]
C=0
while True:
    C+=1
    print(C)
    spe=np.random.uniform(0.1, 2)
    p10=np.random.uniform(0.1, 0.5)
    p01=np.random.uniform(0.01, 0.2)
    P01=np.zeros(100)
    P01[1:]=p01**np.arange(1,100)
    P01[0]=1-np.sum(P01[1:])
    J=1000
    P0=np.zeros((100,100))
    for n in np.arange(100):
        counter=0
        while counter<J:
            counter+=1
            area=np.sum(np.random.normal(1, spe, n))
            n0=np.random.choice(np.arange(100), p=P01, size=1)[0]
            if erf((area-1)/(np.sqrt(2)*spe))<2*p10-1:
                P0[n0,n]+=1
            elif int(np.round(area))+n0<100:
                P0[int(np.round(area))+n0,n]+=1
    P0=P0/J



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

    I=np.arange(100*100)
    def func(I, spe):
        return np.ravel(make_P(spe))

    p,cov = curve_fit(func, I, np.ravel(P0), p0=spe)
    SPE.append(spe)
    P_1_0.append(p10)
    P_0_1.append(p01)
    P.append(p)
    if C%10==0:
        np.savez('fit', SPE=SPE, P10=P_1_0, P01=P_0_1, P=P)




fig = plt.figure()
F=make_P(spe)
plt.plot(P0[:10,2], 'r.')
plt.plot(F[:10,2], 'k.')
plt.show()
