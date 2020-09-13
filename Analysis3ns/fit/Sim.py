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



# @profile
def Sim_fit(x1, x2, left, right, gamma, Q, T, St, mu, W, F, Tf, Ts, R, a, PEbins):
    N_events=10000
    d=np.zeros((N_events, 33, len(Q)))
    H=np.zeros((50, 33, len(Q)))
    spectra=np.zeros((len(PEbins)-1,len(Q)))
    cov=np.zeros((11, 15))
    v=make_v(N_events, mu, x1, x2)
    N=np.random.poisson(1000*gamma/W, N_events)
    p=multiprocessing.Pool(processes=2)
    d=p.map(make_d, make_iter(N, Q, T, St, F, Tf, Ts, R, a, v))
    p.close()
    p.join()

    s=np.sum(np.sum(d,axis=1), axis=1)
    ind=np.nonzero(np.logical_and(s>left, s<right))[0]
    if len(ind)==0:
        w=np.abs(np.mean(s)-(left+right)/2)
        return H-w, spectra-w, cov-w
    d=np.array(d)[ind]

    for i in range(len(Q)):
        spectra[:,i]=np.histogram(np.sum(d[:,:,i], axis=1), bins=PEbins)[0]

    k=0
    for i in range(5):
        Si=np.sum(d[:,:,i], axis=1)
        Mi=np.mean(Si)
        for j in range(i+1,6):
            Sj=np.sum(d[:,:,j], axis=1)
            Mj=np.mean(Sj)
            cov[:,k]=np.histogram((Si-Mi)*(Sj-Mj)/(Mi*Mj), bins=11, range=[-0.1,0.1])[0]
            k+=1

    for k in range(33):
        for j in range(len(Q)):
            H[:,k,j]=np.histogram(np.sum(d[:,3*k:3*k+3,j], axis=1), bins=np.arange(np.shape(H)[0]+1)-0.5)[0]
    return H/len(ind), spectra/len(ind), cov/len(ind)



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
        d=v[x1]+np.sqrt((10/40)**2-v[x2]**2-v[-1]**2)
        P=np.exp(-d/mu)
        if 1==np.random.choice([0,1], size=1,  p=[1-P, P]):
            V=np.vstack((V,v))
    return V[1:]

def make_d(iter_array):
    [Q, T, St, N, F, Tf, Ts, R, a, v]=iter_array
    Strig=2
    d=np.zeros((200, len(Q)))
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
        h, bins=np.histogram(tj, bins=np.arange(201))
        d[:,j]=h
        if np.any(h>0):
            t0[j]=np.amin(np.nonzero(h>0)[0])
    for j in range(len(Q)):
        d[:,j]=np.roll(d[:,j], -int(np.amin(t0)))
    return d[:99]

def make_d_show(iter_array):
    [Q, T, St, N, F, Tf, Ts, R, a, v]=iter_array
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
        h, bins=np.histogram(tj, bins=np.arange(201))
        d[0,:,j]=h
        d[1,:,j]=np.histogram(tj[np.nonzero(np.logical_and(hits[ind]>=recomb, np.isin(hits[ind], slow_i)))[0]], bins=np.arange(201))[0]
        d[2,:,j]=np.histogram(tj[np.nonzero(np.logical_and(hits[ind]>=recomb, np.isin(hits[ind], fast_i)))[0]], bins=np.arange(201))[0]
        d[3,:,j]=np.histogram(tj[np.nonzero(np.logical_and(hits[ind]<recomb, np.isin(hits[ind], slow_i)))[0]], bins=np.arange(201))[0]
        d[4,:,j]=np.histogram(tj[np.nonzero(np.logical_and(hits[ind]<recomb, np.isin(hits[ind], fast_i)))[0]], bins=np.arange(201))[0]
        if np.any(h>0):
            t0[j]=np.amin(np.nonzero(h>0)[0])
    for i in range(5):
        for j in range(len(Q)):
            d[i,:,j]=np.roll(d[i,:,j], -int(np.amin(t0)))
    return d[:99]


def make_recomb(t, a, eta):
    dt=t[1]-t[0]
    ni=np.ones(len(t))
    ne=np.ones(len(t))*(1-eta)
    for i in range(1, len(t)):
        ni[i]=ni[i-1]*(1-dt*a*ne[i-1])
        ne[i]=ne[i-1]*(1-dt*a*ni[i-1])
    return dt*a*(ni*ne)/(1-eta)




def Sim_show(x1, x2, left, right, gamma, Q, T, St, mu, W, F, Tf, Ts, R, a, bins):

    N_events=10000
    H=np.zeros((50, 33, len(Q)))
    G=np.zeros((250,33))
    Gtrp=np.zeros((250,33))
    Gsng=np.zeros((250,33))
    GRtrp=np.zeros((250,33))
    GRsng=np.zeros((250,33))
    spectra=np.zeros((len(bins)-1,len(Q)))
    cov=np.zeros((11, 15))

    v=make_v(N_events, mu, x1, x2)
    N=np.random.poisson(1000*gamma/W, N_events)
    p=multiprocessing.Pool(processes=2)
    ds=p.map(make_d_show, make_iter(N, Q, T, St, F, Tf, Ts, R, a, v))
    p.close()
    p.join()

    ds=np.array(ds)
    s=np.sum(np.sum(ds[:,0,:],axis=1), axis=1)
    ind=np.nonzero(np.logical_and(s>left, s<right))[0]
    if len(ind)==0:
        w=np.abs(np.mean(s)-(left+right)/2)
        return H-w, spectra-w, cov-w, G-w, G-w, G-w, G-w, G-w, 0
    ds=ds[ind]
    d=ds[:,0]
    trp=ds[:,1]
    sng=ds[:,2]
    Rtrp=ds[:,3]
    Rsng=ds[:,4]

    for i in range(len(Q)):
        spectra[:,i]=np.histogram(np.sum(d[:,:,i], axis=1), bins=bins)[0]

    k=0
    for i in range(5):
        Si=np.sum(d[:,:,i], axis=1)
        Mi=np.mean(Si)
        for j in range(i+1,6):
            Sj=np.sum(d[:,:,j], axis=1)
            Mj=np.mean(Sj)
            cov[:,k]=np.histogram((Si-Mi)*(Sj-Mj)/(Mi*Mj), bins=11, range=[-0.5,0.5])[0]
            k+=1

    for k in range(33):
        G[:,k]=np.histogram(np.sum(np.sum(d[:,3*k:3*(k+1),:], axis=1), axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        Gtrp[:,k]=np.histogram(np.sum(np.sum(trp[:,3*k:3*(k+1),:], axis=1), axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        Gsng[:,k]=np.histogram(np.sum(np.sum(sng[:,3*k:3*(k+1),:], axis=1), axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        GRtrp[:,k]=np.histogram(np.sum(np.sum(Rtrp[:,3*k:3*(k+1),:], axis=1), axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        GRsng[:,k]=np.histogram(np.sum(np.sum(Rsng[:,3*k:3*(k+1),:], axis=1), axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        for j in range(len(Q)):
            H[:,k,j]=np.histogram(np.sum(d[:,3*k:3*(k+1),j], axis=1), bins=np.arange(np.shape(H)[0]+1)-0.5)[0]
    return H/len(ind), spectra/len(ind), cov/len(ind),  G/len(ind), Gtrp/len(ind), Gsng/len(ind), GRtrp/len(ind), GRsng/len(ind), len(ind)
