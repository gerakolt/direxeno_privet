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


def Const(tau,T,s):
    C=1/(1-erf(s/(np.sqrt(2)*tau)+T/(np.sqrt(2)*s))+np.exp(-s**2/(2*tau**2)-T/tau)*(1+erf(T/(np.sqrt(2)*s))))
    return C

def Int(t, tau, T, s):
    dt=t[1]-t[0]
    return np.exp(-t/tau)*(1-erf(s/(np.sqrt(2)*tau)-(t-T)/(np.sqrt(2)*s)))*dt*Const(tau,T,s)/tau


def make_T0(Ms):
    p0=np.append(1, np.prod(np.cumprod(Ms, axis=0), axis=1))
    t0=p0[:-1]*(np.shape(Ms)[1]-np.sum(Ms, axis=1))
    return t0/np.sum(t0), p0/np.sum(p0)


def Model(NQ, T, F, Tf, Ts, St):
    n=15
    t=np.arange(2000)/5
    dt=t[1]-t[0]
    I=np.arange(2*n*len(t))
    Ms=np.zeros((2*n, len(t), len(NQ)))
    for i in range(len(NQ)):
        m=NQ[i]*(F[i]*Int(t, Tf[i], T[i], St[i])+(1-F[i])*Int(t, Ts[i], T[i], St[i]))
        Ms[:,:,i]=(poisson.pmf(np.floor(I/len(t)), m[I%len(t)]).reshape((2*n, len(t))))
    T0, P0=make_T0(Ms[0,:1000,:])
    temporal=np.zeros((n, len(T0), len(NQ)))
    for i in range(len(NQ)):
        h=np.zeros((np.shape(Ms)[0], len(T0)))
        h[0,0]=np.sum(P0[:-1]*(np.shape(Ms)[2]-1-np.sum(Ms[0,:len(P0[:-1])], axis=1)+Ms[0,:len(P0[:-1]),i])*Ms[0,:len(P0[:-1]),i])
        h[1:,0]=np.sum(P0[:-1]*Ms[1:,:len(P0[:-1]),i], axis=1)
        for j in range(1,len(T0)):
            h[:,j]=np.sum(T0*Ms[:,j:j+len(T0), i], axis=1)
        temporal[:,:,i]=h[:n]/np.sum(h[:n], axis=0)
    return temporal


def Sim(NQ, T, Strig, F, Tf, Ts, St):
    N_events=10000
    d=np.zeros((10000, 1000, len(NQ)))
    d0=np.zeros((10000, 1000, len(NQ)))
    H=np.zeros((15, 1000, len(NQ)))
    for i in range(N_events):
        t0=np.zeros(len(NQ))
        trig=np.random.normal(0, Strig*5, 1)
        for j in range(len(NQ)):
            n=np.random.poisson(NQ[j])
            ch=np.random.choice(2, size=n, replace=True, p=[F[j], (1-F[j])])
            nf=len(np.nonzero(ch==0)[0])
            ns=len(np.nonzero(ch==1)[0])
            tf=np.random.normal(trig+5*T[j]+np.random.exponential(Tf[j]*5, nf), St[j]*5, nf)
            ts=np.random.normal(trig+5*T[j]+np.random.exponential(Ts[j]*5, ns), St[j]*5, ns)
            t=np.append(tf, ts)
            h, bins=np.histogram(t, bins=np.arange(1001)-0.5)
            t0[j]=np.amin(np.nonzero(h>0)[0])
            d[i,:,j]=h
            d0[i,:,j]=h
        for j in range(len(NQ)):
            d[i,:,j]=np.roll(d[i,:,j], -int(np.amin(t0)))
    for j in range(len(NQ)):
        for k in range(1000):
            H[:,k,j]=np.histogram(d[:,k,j], bins=np.arange(16)-0.5)[0]
    return H/N_events
