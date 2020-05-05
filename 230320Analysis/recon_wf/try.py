import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, binom
import sys
def try_spectra(x, a, m, Spe):
    if Spe==0:
        return a*poisson.pmf(x, m)
    H=np.zeros(len(x))
    dx=x[1]-x[0]
    def make_P(Spe):
        P=np.zeros((400, 400))
        P[0,0]=1
        for i in range(len(P[:,0])):
            r=np.linspace(i-0.5,i+0.5,1000)
            dr=r[1]-r[0]
            P[i,1]=dr*np.sum(np.exp(-0.5*(r-1)**2/Spe**2))/(np.sqrt(2*np.pi)*Spe)
        for j in range(2, len(P[0,:])):
            for i in range(len(P[:,0])):
                P[i,j]=np.sum(P[:i+1,1]*np.flip(P[:i+1,j-1]))
        return P/np.sum(P, axis=0)
    P=make_P(Spe)
    ns=np.arange(np.shape(P)[0])
    h=np.matmul(P, poisson.pmf(ns, m))
    for i in range(len(x)):
        for n in range(int(np.ceil(x[i]-0.5*dx)), int(np.ceil(x[i]+0.5*dx))):
            H[i]+=a*h[n]
    return H


def try_spectra(x, a, m, Spe, p0):
    q=(1-p0)
    if Spe==0:
        return a*poisson.pmf(x, m)
    H=np.zeros(len(x))
    dx=x[1]-x[0]
    def make_P(Spe):
        P=np.zeros((400, 400))
        P[0,0]=1
        for i in range(len(P[:,0])):
            r=np.linspace(i-0.5,i+0.5,1000)
            dr=r[1]-r[0]
            P[i,1]=dr*np.sum(np.exp(-0.5*(r-1)**2/Spe**2))/(np.sqrt(2*np.pi)*Spe)
        for j in range(2, len(P[0,:])):
            for i in range(len(P[:,0])):
                P[i,j]=np.sum(P[:i+1,1]*np.flip(P[:i+1,j-1]))
        for i in range(1, len(P[:,0])):
            P[i]=np.sum(binom.pmf(np.arange(i+1), i, q)*P.T[:,:i+1], axis=1)
        return P/np.sum(P, axis=0)
    P=make_P(Spe)
    ns=np.arange(np.shape(P)[0])
    h=np.matmul(P, poisson.pmf(ns, m))
    for i in range(len(x)):
        for n in range(int(np.ceil(x[i]-0.5*dx)), int(np.ceil(x[i]+0.5*dx))):
            H[i]+=a*h[n]
    return H


x=np.arange(0,100)
p0=[1, 50, 0, 0]
p1=[1, 50, 0.5, 0.01]

h0=try_spectra(x, *p0)
h1=try_spectra(x, *p1)

plt.plot(x, h0, 'ko')
plt.plot(x, h1, 'ro')
plt.show()
