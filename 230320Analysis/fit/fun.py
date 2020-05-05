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

params=['NQ', 'F', 'Tf', 'Ts', 'St', 'Mpad', 'Spad', ' Mpe', 'Spe', 'a_pad', 'a_spe', 'a_dpe', 'p01', 'a_spec', 'BG_r']

def make_initial_simplex(p):
    bounds_dn=[0,0,0,0,0,-10000,0,0,0,0,0,0,0,0,0]
    bounds_up=[100, 1, 50, 200, 5, 10000, 10000, 10000, 10000, 1e5, 1e5, 1e5, 1, 1e5, 2]
    n=len(p)
    N=np.zeros((n+1, n))
    N[0]=p
    N[1]=bounds_dn
    N[2]-bounds_up
    for i in range(3,n+1):
        N[i]=np.random.uniform(bounds_dn, bounds_up)


def make_P(Spe, p01):
    n=300
    P=np.zeros((n,n))
    P[0,1]=p01
    P[1,1]=0.5*(1+erf(0.5/(np.sqrt(2)*Spe)))-p01

    for i in np.arange(2, n):
        P[i,1]=0.5*(erf((i-0.5)/(np.sqrt(2)*Spe))-erf((i-1.5)/(np.sqrt(2)*Spe)))

    for i in range(n):
        for j in range(2,n):
            P[i,j]=np.sum(P[:i+1,1]*np.flip(P[:i+1,j-1]))

    P[1,0]=1-np.sum(P[1,1:])
    if P[1,0]>0.5:
        return 1+P[1,0]
    if P[1,0]<0:
        return 1-P[1,0]
    P[2:,0]=P[1,0]**(np.arange(2,n))
    P[0,0]=1-np.sum(P[1:,0])
    if np.any(P<0):
        print('P<0')
        print('Spe=', Spe, 'P01=',p01)
        print(np.nonzero(P<0))
        print(P[:2,:2])
        sys.exit()

    if np.any(P>=1):
        print('P=1')
        print('Spe=', Spe, 'P01=',p01)
        print(np.nonzero(P>=1))
        print(P[:2,:2])
        sys.exit()
    return P

def model_spec(ns, NQ, P):
    h=poisson.pmf(ns, NQ)
    return np.ravel(np.matmul(P[:,ns], h.reshape(len(ns),1))[ns])

def model_area(areas, Mpad, Spad, Mpe, Spe, a_pad, a_spe, a_dpe):
    return a_pad*np.exp(-0.5*(areas-Mpad)**2/(Spad)**2)+a_spe*np.exp(-0.5*(areas-(Mpe+Mpad))**2/((Mpe*Spe)**2+Spad**2))+a_dpe*np.exp(-0.5*(areas-(Mpe+2*Mpad))**2/(2*(Mpe*Spe)**2+Spad**2))


def Const(tau,T,s):
    C=1/(1-erf(s/(np.sqrt(2)*tau)+T/(np.sqrt(2)*s))+np.exp(-s**2/(2*tau**2)-T/tau)*(1+erf(T/(np.sqrt(2)*s))))
    return C

def Int(t, tau, T, s):
    dt=t[1]-t[0]
    return np.exp(-t/tau)*(1-erf(s/(np.sqrt(2)*tau)-(t-T)/(np.sqrt(2)*s)))*dt*Const(tau,T,s)/tau

def make_z(h):
    z=np.zeros((np.shape(h)[0], 1000))
    p0=make_P0(h)
    C=0
    for k in range(np.shape(z)[1]):
        C+=p0[k]*(1-h[0,k])
    for i in range(1, np.shape(z)[0]):
        for k in range(np.shape(z)[1]):
            z[i,0]+=p0[k]*h[i,k]
    for i in range(1, np.shape(z)[0]):
        for j in range(1, np.shape(z)[1]):
            for k in range(np.shape(z)[1]):
                z[i,j]+=p0[k]*(1-h[0,k])*h[i,j+k]
    return z/C


def make_P0(h):
    p0=np.zeros(1000)
    for i in range(1000):
        p0[i]=np.prod(h[0,:i])
    return p0



def model2(NQ, F, Tf, Ts, St, P):
    T=50
    n=10
    t=np.arange(2000)/5
    I=np.arange(2*n*len(t))
    m=NQ*(F*Int(t,Tf,T,St)+(1-F)*Int(t,Ts,T,St))
    M=poisson.pmf(np.floor(I/len(t)), m[I%len(t)])
    h=make_z(M.reshape((2*n, len(t))))
    return np.matmul(P[:n,:np.shape(h)[0]],h)
