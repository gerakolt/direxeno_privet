from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf
from scipy.special import comb

# def Sim(NQ, T, Strig, R, F, Tf, Ts, St, q0):
#     N_events=10000
#     d=np.zeros((10000, 1000, len(NQ)))
#     H=np.zeros((15, 1000, len(NQ)))
#     for i in range(N_events):
#         t0=np.zeros(len(NQ))
#         trig=np.random.normal(0, Strig*5, 1)
#         for j in range(len(NQ)):
#             n=np.random.poisson(NQ[j])
#             ch0=np.random.choice(3, size=1000, replace=True, p=[1-q0[j]-q0[j]**2, q0[j], q0[j]**2])
#             ch=np.random.choice(3, size=n, replace=True, p=[R[j], (1-R[j])*F, (1-R[j])*(1-F)])
#             nd=len(np.nonzero(ch==0)[0])
#             nf=len(np.nonzero(ch==1)[0])
#             ns=len(np.nonzero(ch==2)[0])
#             td=np.random.normal(trig+5*T[j], St[j]*5, nd)
#             tf=np.random.normal(trig+5*T[j]+np.random.exponential(Tf*5, nf), St[j]*5, nf)
#             ts=np.random.normal(trig+5*T[j]+np.random.exponential(Ts*5, ns), St[j]*5, ns)
#             t=np.append(td, np.append(tf, ts))
#             h, bins=np.histogram(t, bins=np.arange(1001)-0.5)
#             h+=ch0
#             t0[j]=np.amin(np.nonzero(h>0)[0])
#             d[i,:,j]=h
#         for j in range(len(NQ)):
#             d[i,:,j]=np.roll(d[i,:,j], -int(np.amin(t0)))
#     for j in range(len(NQ)):
#         for k in range(1000):
#             H[:,k,j]=np.histogram(d[:,k,j], bins=np.arange(16)-0.5)[0]
#     return H/N_events

def Sim(NQ, T, Strig, R, F, Tf, Ts, St, q0):
    N_events=10000
    d=np.zeros((10000, 1000, len(NQ)))
    H=np.zeros((15, 1000, len(NQ)))
    for i in range(N_events):
        t0=np.zeros(len(NQ))
        trig=np.random.normal(0, Strig*5, 1)
        for j in range(len(NQ)):
            n=np.random.poisson(NQ[j])
            # ch0=np.random.choice(3, size=1000, replace=True, p=[1-q0[j]-q0[j]**2, q0[j], q0[j]**2])
            ch0=np.random.choice(15, size=1000, replace=True, p=np.append(1-q0[j]/(1-q0[j]), q0[j]**np.arange(1, 15)))
            ch=np.random.choice(3, size=n, replace=True, p=[R[j], (1-R[j])*F, (1-R[j])*(1-F)])
            nd=len(np.nonzero(ch==0)[0])
            nf=len(np.nonzero(ch==1)[0])
            ns=len(np.nonzero(ch==2)[0])
            td=np.random.normal(trig+5*T[j], St[j]*5, nd)
            tf=np.random.normal(trig+5*T[j]+np.random.exponential(Tf*5, nf), St[j]*5, nf)
            ts=np.random.normal(trig+5*T[j]+np.random.exponential(Ts*5, ns), St[j]*5, ns)
            t=np.append(td, np.append(tf, ts))
            h, bins=np.histogram(t, bins=np.arange(1001)-0.5)
            h+=ch0
            t0[j]=np.amin(np.nonzero(h>0)[0])
            d[i,:,j]=h
        for j in range(len(NQ)):
            d[i,:,j]=np.roll(d[i,:,j], -int(np.amin(t0)))
    for j in range(len(NQ)):
        for k in range(1000):
            H[:,k,j]=np.histogram(d[:,k,j], bins=np.arange(16)-0.5)[0]
    return H/N_events



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


def Model(NQ, T, R, F, Tf, Ts, St):
    n=15
    t=np.arange(2000)/5
    dt=t[1]-t[0]
    I=np.arange(2*n*len(t))
    Ms=np.zeros((2*n, len(t), len(NQ)))
    for i in range(len(NQ)):
        m=NQ[i]*((1-R[i])*(F*Int(t, Tf, T[i], St[i])+(1-F)*Int(t, Ts, T[i], St[i]))+R[i]*dt*np.exp(-0.5*(t-T[i])**2/St[i]**2)/(np.sqrt(2*np.pi)*St[i]))
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
