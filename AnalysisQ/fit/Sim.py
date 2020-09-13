import multiprocessing
import numpy as np
import sys
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf, expi
from scipy.special import comb
from scipy.signal import convolve2d
import time
from admin import make_iter
from PMTgiom import whichPMT
import matplotlib.pyplot as plt




def Sim_fit(x1, x2, left, right, gamma, Q, W, bins):
    spectra=np.zeros((len(bins)-1, len(Q)))
    N_events=1000
    N=np.random.poisson(1000*gamma/W, N_events)
    v=make_v(N_events, 0, x1, x2)
    s=make_d(N, 0.85*np.array(Q), v)
    up=s[:,0]+s[:,0]
    dn=s[:,-1]+s[:,-2]+s[:,-3]
    ind=np.nonzero(np.logical_and(np.logical_and(np.sum(s, axis=1)>left, np.sum(s, axis=1)<right), dn<3*up+18))[0]
    if len(ind)==0:
        w=np.abs(np.mean(np.sum(s, axis=1))-(left+right)/2)
        return spectra-w
    s=np.array(s)[ind]
    spectrum=np.histogram(np.sum(s, axis=1), bins=bins)[0]
    for i in range(len(Q)):
        spectra[:,i]=np.histogram(s[:,i], bins=bins)[0]
    return spectrum/N_events



def make_v(N, mu, x1, x2):
    k=0
    V=np.zeros(3)
    while len(V)<=N:
        k+=1
        costheta=np.random.uniform(-1,1)
        phi=np.random.uniform(0,2*np.pi)
        r3=np.random.uniform(0,(10/40)**3)
        r=r3**(1/3)
        v=np.array([r*np.sin(np.arccos(costheta))*np.cos(phi), r*np.sin(np.arccos(costheta))*np.sin(phi), r*costheta])
        # V=np.vstack((V,v))
        d=v[x1]+np.sqrt((10/40)**2-v[x2]**2-v[-1]**2)
        # P=np.exp(-d/mu)
        P=1
        if 1==np.random.choice([0,1], size=1,  p=[1-P, P]):
            V=np.vstack((V,v))
    return V[1:]

def make_d(N,Q,v):
    s=np.zeros((len(N), len(Q)))
    for i in range(len(N)):
        costheta=np.random.uniform(-1,1, N[i])
        phi=np.random.uniform(0,2*np.pi, N[i])
        us=np.vstack((np.vstack((np.sin(np.arccos(costheta))*np.cos(phi), np.sin(np.arccos(costheta))*np.sin(phi))), costheta))
        pmt_hit=whichPMT(v[i], us)
        for j in range(len(Q)):
            s[i, j]=np.random.binomial(len(np.nonzero(pmt_hit==j)[0]), Q[j])
    return s

def make_d_show(iter_array):
    [Q, T, St, Sa, N, F, Tf, Ts, R, a, v]=iter_array
    Strig=2
    d=np.zeros((5, 200, len(Q)))
    t0=np.zeros(len(Q))
    trig=np.random.normal(0, Strig, 1)
    ex=np.random.binomial(N, 1-R)
    recomb=N-ex
    t=np.zeros(recomb+ex)
    u=np.random.uniform(size=recomb)
    t[:recomb]+=1/a*(u/(1-u))
    ch=np.random.choice(2, size=recomb+ex, replace=True, p=[F, 1-F])
    t[ch==0]+=np.random.exponential(Tf, len(t[ch==0]))
    t[ch==1]+=np.random.exponential(Ts, len(t[ch==1]))
    slow_i=np.nonzero(ch==1)[0]
    fast_i=np.nonzero(ch==0)[0]

    costheta=np.random.uniform(-1,1, len(t))
    phi=np.random.uniform(0,2*np.pi, len(t))
    us=np.vstack((np.vstack((np.sin(np.arccos(costheta))*np.cos(phi), np.sin(np.arccos(costheta))*np.sin(phi))), costheta))
    pmt_hit=whichPMT(v, us)
    for j in range(len(Q)):
        hits=np.nonzero(pmt_hit==j)[0]
        ind=np.nonzero(1==np.random.choice(2, size=len(hits), replace=True, p=[1-Q[j], Q[j]]))[0]
        tj=np.random.normal(trig+T[j]+t[hits[ind]], St[j], len(ind))
        h0, bins=np.histogram(tj, bins=np.arange(201))
        As=np.random.normal(loc=h0[h0>0], scale=np.sqrt(h0[h0>0]*Sa[j]))
        d[0,h0>0,j]+=np.round(As).astype(int)

        h=np.histogram(tj[np.nonzero(np.logical_and(hits[ind]>=recomb, np.isin(hits[ind], slow_i)))[0]], bins=np.arange(201))[0]
        As=np.random.normal(loc=h[h>0], scale=np.sqrt(h[h>0]*Sa[j]))
        d[1,h>0,j]+=np.round(As).astype(int)

        h=np.histogram(tj[np.nonzero(np.logical_and(hits[ind]>=recomb, np.isin(hits[ind], fast_i)))[0]], bins=np.arange(201))[0]
        As=np.random.normal(loc=h[h>0], scale=np.sqrt(h[h>0]*Sa[j]))
        d[2,h>0,j]+=np.round(As).astype(int)

        h=np.histogram(tj[np.nonzero(np.logical_and(hits[ind]<recomb, np.isin(hits[ind], slow_i)))[0]], bins=np.arange(201))[0]
        As=np.random.normal(loc=h[h>0], scale=np.sqrt(h[h>0]*Sa[j]))
        d[3,h>0,j]+=np.round(As).astype(int)

        h=np.histogram(tj[np.nonzero(np.logical_and(hits[ind]<recomb, np.isin(hits[ind], fast_i)))[0]], bins=np.arange(201))[0]
        As=np.random.normal(loc=h[h>0], scale=np.sqrt(h[h>0]*Sa[j]))
        d[4,h>0,j]+=np.round(As).astype(int)

        if np.any(h0>0):
            t0[j]=np.amin(np.nonzero(h0>0)[0])
    for i in range(5):
        for j in range(len(Q)):
            d[i,:,j]=np.roll(d[i,:,j], -int(np.amin(t0)))
    return d


def make_recomb(t, a, eta):
    dt=t[1]-t[0]
    ni=np.ones(len(t))
    ne=np.ones(len(t))*(1-eta)
    for i in range(1, len(t)):
        ni[i]=ni[i-1]*(1-dt*a*ne[i-1])
        ne[i]=ne[i-1]*(1-dt*a*ni[i-1])
    return dt*a*(ni*ne)/(1-eta)




def Sim_show(x1, x2, left, right, gamma, Q, T, St, Sa, mu, W, F, Tf, Ts, R, a, PEs):

    N_events=10000
    H=np.zeros((50, 200, len(Q)))
    G=np.zeros((250,200))
    Gtrp=np.zeros((250,200))
    Gsng=np.zeros((250,200))
    GRtrp=np.zeros((250,200))
    GRsng=np.zeros((250,200))
    spectra=np.zeros((len(PEs),len(Q)))
    cov=np.zeros((11, 15))

    v=make_v(N_events, mu, x1, x2)
    N=np.random.poisson(1000*gamma/W, N_events)
    p=multiprocessing.Pool(processes=2)
    ds=p.map(make_d_show, make_iter(N, Q, T, St, Sa, F, Tf, Ts, R, a, v))
    p.close()
    p.join()

    ds=np.array(ds)
    s=np.sum(np.sum(ds[:,0,:100],axis=1), axis=1)
    ind=np.nonzero(np.logical_and(s>left, s<right))[0]
    if len(ind)==0:
        w=np.abs(np.mean(s)-(left+right)/2)
        print('No events in the left-right band')
        return H-w, spectra-w, cov-w, G-w, G-w, G-w, G-w, G-w, 0
    ds=ds[ind]
    d=ds[:,0]
    trp=ds[:,1]
    sng=ds[:,2]
    Rtrp=ds[:,3]
    Rsng=ds[:,4]

    for i in range(len(Q)):
        spectra[:,i]=np.histogram(np.sum(d[:,:100,i], axis=1), bins=np.arange(len(PEs)+1)-0.5)[0]

    # k=0
    # for i in range(5):
    #     Si=np.sum(d[:,:,i], axis=1)
    #     Mi=np.mean(Si)
    #     for j in range(i+1,6):
    #         Sj=np.sum(d[:,:,j], axis=1)
    #         Mj=np.mean(Sj)
    #         cov[:,k]=np.histogram((Si-Mi)*(Sj-Mj)/(Mi*Mj), bins=11, range=[-0.5,0.5])[0]
    #         k+=1

    for k in range(200):
        G[:,k]=np.histogram(np.sum(d[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        Gtrp[:,k]=np.histogram(np.sum(trp[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        Gsng[:,k]=np.histogram(np.sum(sng[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        GRtrp[:,k]=np.histogram(np.sum(Rtrp[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        GRsng[:,k]=np.histogram(np.sum(Rsng[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        for j in range(len(Q)):
            H[:,k,j]=np.histogram(d[:,k,j], bins=np.arange(np.shape(H)[0]+1)-0.5)[0]
    return H/N_events, spectra/N_events, cov/N_events,  G/N_events, Gtrp/N_events, Gsng/N_events, GRtrp/N_events, GRsng/N_events, N_events
