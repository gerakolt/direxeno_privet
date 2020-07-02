import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf
import warnings
from minimize import minimize, make_ps
from scipy.signal import convolve2d, convolve


# def Recomb(t, F, Tf, Ts, T, St, a, eta):
#     dt=t[1]-t[0]
#     y=make_recomb(t, a, eta)
#     Y=np.zeros(len(t))
#     Z=F*np.exp(-t/Tf)/Tf+(1-F)*np.exp(-t/Ts)/Ts
#     for i in range(1,len(t)):
#         # Y[i]=np.sum(y[:i]*Z[i-1::-1])*dt
#         Y[i]=np.sum(y[:i+1]*Z[i::-1])*dt
#     I=np.arange(len(t)*len(T))
#     B=(np.exp(-0.5*(t[I//len(T)]-T[I%len(T)])**2/St[I%len(T)]**2)/np.sqrt(2*np.pi*St[I%len(T)]**2)).reshape(len(t), len(T))
#     C=convolve2d(B, Y.reshape(len(Y), 1), mode='full')*dt
#     return np.ravel(C[:len(t),:])

def Recomb(t, F, Tf, Ts, a, eta, T, St):
    dt=t[1]-t[0]
    y=make_recomb(t, a, eta)
    Y=np.zeros(len(t))
    Z=F*np.exp(-t/Tf)/Tf+(1-F)*np.exp(-t/Ts)/Ts
    for i in range(1,len(t)):
        Y[i]=np.sum(y[:i+1]*Z[i::-1])*dt
    B=np.exp(-0.5*(t-T)**2/St**2)/np.sqrt(2*np.pi*St**2)
    C=convolve(B, Y, mode='full')*dt

    # I=np.arange(len(t)*len(T))
    # B=(np.exp(-0.5*(t[I//len(T)]-T[I%len(T)])**2/St[I%len(T)]**2)/np.sqrt(2*np.pi*St[I%len(T)]**2)).reshape(len(t), len(T))
    # C=convolve2d(B, Y.reshape(len(Y), 1), mode='full')*dt
    return C[:len(t)]

def make_recomb(t, a, eta):
    dt=t[1]-t[0]
    ni=np.ones(len(t))
    ne=np.ones(len(t))*(1-eta)
    for i in range(1, len(t)):
        ni[i]=ni[i-1]*(1-dt*a*ne[i-1])
        ne[i]=ne[i-1]*(1-dt*a*ni[i-1])
    return dt*a*(ni*ne)/(1-eta)


a=0.5
eta=0.1
N=1000
n=100
F=0.1
Tf=2
Ts=30
T=40
St=0.8
y=np.sum(Recomb(np.arange(1000*20)/100, F, Tf, Ts, a, eta, T, St).reshape(1000, 20), axis=1)
f=np.sum(make_recomb(np.arange(1000*20)/100, a, eta).reshape(1000, 20), axis=1)
f[-1]=1-np.sum(f[:-1])
d=np.zeros((N, 1000))
for i in range(N):
    t=np.random.choice(np.arange(1000)/5, size=n,  replace=True, p=f)
    ch=np.random.choice(2, size=n, replace=True, p=[F, 1-F])
    t[ch==0]+=np.random.exponential(Tf, len(t[ch==0]))
    t[ch==1]+=np.random.exponential(Ts, len(t[ch==1]))
    t=np.random.normal(T+t, St, size=len(t))
    d[i]=np.histogram(t, bins=np.arange(1001)/5-0.1)[0]

print(np.sum(y), np.sum(np.mean(d, axis=0)/n))

plt.figure()
plt.plot(np.mean(d, axis=0)/n, 'k.')
plt.plot(y, 'r.')
plt.show()
#     N_events=2000
#     Strig=0.001
#     d=np.zeros((N_events, 200, len(Q)))
#     H=np.zeros((50, 200, len(Q)))
#     G=np.zeros((250,200))
#     trp=np.zeros((N_events, 200, len(Q)))
#     sng=np.zeros((N_events, 200, len(Q)))
#     Rtrp=np.zeros((N_events, 200, len(Q)))
#     Rsng=np.zeros((N_events, 200, len(Q)))
#     Gtrp=np.zeros((250,200))
#     Gsng=np.zeros((250,200))
#     GRtrp=np.zeros((250,200))
#     GRsng=np.zeros((250,200))
#     for i in range(N_events):
#         print('in sim', i)
#         t0=np.zeros(len(Q))
#         trig=np.random.normal(0, 5*Strig, 1)
#         N_glob=np.random.poisson(N)
#         ex=np.random.binomial(N_glob, 1-R)
#         recomb=np.random.binomial(N_glob-ex, 1-eta)
#         t=np.zeros(recomb+ex)
#         # u=np.random.uniform(size=recomb)
#         # I=np.arange(1000*recomb)
#         # t[:recomb]+=np.arange(1000)[np.argmin(np.abs(u[I%recomb]-F_recomb[I//recomb]).reshape(1000, recomb), axis=0)]
#         t[:recomb]+=np.random.choice(np.arange(1000), size=recomb,  replace=True, p=f)
#         ch=np.random.choice(2, size=recomb+ex, replace=True, p=[F, 1-F])
#         t[ch==0]+=np.random.exponential(5*Tf, len(t[ch==0]))
#         t[ch==1]+=np.random.exponential(5*Ts, len(t[ch==1]))
#         slow_i=np.nonzero(ch==1)[0]
#         fast_i=np.nonzero(ch==0)[0]
#         for j in range(len(Q)):
#             ind=np.nonzero(1==np.random.choice(2, size=len(t), replace=True, p=[1-Q[j]*dS[j], Q[j]*dS[j]]))[0]
#             tj=np.random.normal(trig+5*T[j]+t[ind], 5*St[j], len(ind))
#             # tj=t[ind]
#             h, bins=np.histogram(tj, bins=np.arange(201)*5)
#             # if np.any(h>0):
#             #     t0[j]=np.amin(np.nonzero(h>0)[0])
#             d[i,:,j]=h
#             trp[i,:,j]=np.histogram(tj[np.nonzero(np.logical_and(ind>=recomb, np.isin(ind, slow_i)))[0]], bins=np.arange(201)*5)[0]
#             sng[i,:,j]=np.histogram(tj[np.nonzero(np.logical_and(ind>=recomb, np.isin(ind, fast_i)))[0]], bins=np.arange(201)*5)[0]
#             Rtrp[i,:,j]=np.histogram(tj[np.nonzero(np.logical_and(ind<recomb, np.isin(ind, slow_i)))[0]], bins=np.arange(201)*5)[0]
#             Rsng[i,:,j]=np.histogram(tj[np.nonzero(np.logical_and(ind<recomb, np.isin(ind, fast_i)))[0]], bins=np.arange(201)*5)[0]
#         # for j in range(len(Q)):
#         #     d[i,:,j]=np.roll(d[i,:,j], -int(np.amin(t0)))
#         #     trp[i,:,j]=np.roll(trp[i,:,j], -int(np.amin(t0)))
#         #     sng[i,:,j]=np.roll(sng[i,:,j], -int(np.amin(t0)))
#         #     Rtrp[i,:,j]=np.roll(Rtrp[i,:,j], -int(np.amin(t0)))
#         #     Rsng[i,:,j]=np.roll(Rsng[i,:,j], -int(np.amin(t0)))
#
#     spectrum=np.histogram(np.sum(np.sum(d, axis=2), axis=1), bins=np.arange(1000)-0.5)[0]
#     for k in range(200):
#         G[:,k]=np.histogram(np.sum(d[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
#
#         Gtrp[:,k]=np.histogram(np.sum(trp[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
#         Gsng[:,k]=np.histogram(np.sum(sng[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
#         GRtrp[:,k]=np.histogram(np.sum(Rtrp[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
#         GRsng[:,k]=np.histogram(np.sum(Rsng[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
#
#         for j in range(len(Q)):
#             H[:,k,j]=np.histogram(d[:,k,j], bins=np.arange(np.shape(H)[0]+1)-0.5)[0]
#     return H/N_events
#
#
# t=np.arange(200*100)/100
# R=np.sum(Recomb(t, 1, 1, 30, np.array([40]), np.array([0.8]), 0.5, 0.9).reshape(200,100,1), axis=1)
# S=Sim(t, 10, 1, 1, 30, 1, 0.5, 0.9, np.array([1]), np.array([40]), np.array([0.8]))
# plt.plot(R, 'k.')
# plt.plot(np.sum(S[:,:,0].T*np.arange(np.shape(S)[0]), axis=1), 'r.')
# print(np.sum(np.sum(S[:,:,0].T*np.arange(np.shape(S)[0]), axis=1)), np.sum(R))
# plt.show()
