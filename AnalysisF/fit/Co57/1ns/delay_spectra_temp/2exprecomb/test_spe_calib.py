from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf, expi
from scipy.special import comb
from PMTgiom import make_pmts
from scipy.signal import convolve2d




def Sim(t, N, F, Tf, Ts, R, a, eta, Q, T, St):
    f=np.sum(make_recomb(np.arange(1000*20)/100, a, eta).reshape(1000,20), axis=1)
    f[-1]=1-np.sum(f[:-1])
    N_events=200
    Strig=2
    d=np.zeros((N_events, 200, len(Q)))
    H=np.zeros((30, 200, len(Q)))
    G=np.zeros((250,200))
    trp=np.zeros((N_events, 200, len(Q)))
    sng=np.zeros((N_events, 200, len(Q)))
    Rtrp=np.zeros((N_events, 200, len(Q)))
    Rsng=np.zeros((N_events, 200, len(Q)))
    Gtrp=np.zeros((250,200))
    Gsng=np.zeros((250,200))
    GRtrp=np.zeros((250,200))
    GRsng=np.zeros((250,200))
    for i in range(N_events):
        print('in sim', i)
        t0=np.zeros(len(Q))
        trig=np.random.normal(0, Strig, 1)
        N_glob=np.random.poisson(N)
        ex=np.random.binomial(N_glob, 1-R)
        recomb=np.random.binomial(N_glob-ex, 1-eta)
        t=np.zeros(recomb+ex)
        t[:recomb]+=np.random.choice(np.arange(1000)/5, size=recomb,  replace=True, p=f)
        ch=np.random.choice(2, size=recomb+ex, replace=True, p=[F, 1-F])
        t[ch==0]+=np.random.exponential(Tf, len(t[ch==0]))
        t[ch==1]+=np.random.exponential(Ts, len(t[ch==1]))
        slow_i=np.nonzero(ch==1)[0]
        fast_i=np.nonzero(ch==0)[0]
        for j in range(len(Q)):
            ind=np.nonzero(1==np.random.choice(2, size=len(t), replace=True, p=[1-Q[j]*dS[j], Q[j]*dS[j]]))[0]
            tj=np.random.normal(trig+T[j]+t[ind], St[j], len(ind))
            h, bins=np.histogram(tj, bins=np.arange(201))
            for k in range(len(h)):
                h[k]=
            if np.any(h>0):
                t0[j]=np.amin(np.nonzero(h>0)[0])
            d[i,:,j]=h
            trp[i,:,j]=np.histogram(tj[np.nonzero(np.logical_and(ind>=recomb, np.isin(ind, slow_i)))[0]], bins=np.arange(201))[0]
            sng[i,:,j]=np.histogram(tj[np.nonzero(np.logical_and(ind>=recomb, np.isin(ind, fast_i)))[0]], bins=np.arange(201))[0]
            Rtrp[i,:,j]=np.histogram(tj[np.nonzero(np.logical_and(ind<recomb, np.isin(ind, slow_i)))[0]], bins=np.arange(201))[0]
            Rsng[i,:,j]=np.histogram(tj[np.nonzero(np.logical_and(ind<recomb, np.isin(ind, fast_i)))[0]], bins=np.arange(201))[0]
        for j in range(len(Q)):
            d[i,:,j]=np.roll(d[i,:,j], -int(np.amin(t0)))
            trp[i,:,j]=np.roll(trp[i,:,j], -int(np.amin(t0)))
            sng[i,:,j]=np.roll(sng[i,:,j], -int(np.amin(t0)))
            Rtrp[i,:,j]=np.roll(Rtrp[i,:,j], -int(np.amin(t0)))
            Rsng[i,:,j]=np.roll(Rsng[i,:,j], -int(np.amin(t0)))

    spectrum=np.histogram(np.sum(np.sum(d, axis=2), axis=1), bins=np.arange(1000)-0.5)[0]
    for k in range(200):
        G[:,k]=np.histogram(np.sum(d[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]

        Gtrp[:,k]=np.histogram(np.sum(trp[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        Gsng[:,k]=np.histogram(np.sum(sng[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        GRtrp[:,k]=np.histogram(np.sum(Rtrp[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        GRsng[:,k]=np.histogram(np.sum(Rsng[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]

        for j in range(len(Q)):
            H[:,k,j]=np.histogram(d[:,k,j], bins=np.arange(np.shape(H)[0]+1)-0.5)[0]
    return H/N_events, G/N_events, spectrum, Gtrp/N_events, Gsng/N_events, GRtrp/N_events, GRsng/N_events
