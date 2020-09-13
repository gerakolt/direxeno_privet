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





def Sim_fit(x1, x2, left, right, gamma, Q, W, Bins, bins):
    spectra=np.zeros((len(bins)-1, len(Q)))
    cov=np.zeros(15)
    N_events=10000
    N=np.random.poisson(1000*gamma/W, N_events)
    v=make_v(N_events, 0, x1, x2)
    p=multiprocessing.Pool(processes=2)
    s=p.map(make_d, make_iter(N, Q, v))
    p.close()
    p.join()
    s=np.array(s)
    up=s[:,0]+s[:,1]
    dn=s[:,-1]+s[:,-2]+s[:,-3]
    ind=np.nonzero(np.logical_and(np.logical_and(np.sum(s, axis=1)>left, np.sum(s, axis=1)<right),True))[0]
    if len(ind)==0:
        print('len in<0')
        w=np.abs(np.mean(np.sum(s, axis=1))-(left+right)/2)
        return np.zeros(len(Bins)-1)-w, spectra-w, cov-w

    s=np.array(s)[ind]
    spectrum=np.histogram(np.sum(s, axis=1), bins=Bins)[0]
    for i in range(len(Q)):
        spectra[:,i]=np.histogram(s[:,i], bins=bins)[0]

    M=np.mean(s, axis=0)
    k=0
    for i in range(5):
        for j in range(i+1,6):
            cov[k]=np.mean((s[:,i]-M[i])*(s[:,j]-M[j]))
            k+=1

    return spectrum/len(ind), spectra/len(ind), cov


def make_v(N, mu, x1, x2):
    k=0
    V=np.zeros(3)
    while np.shape(V)[0]<=N:
        k+=1
        costheta=np.random.uniform(-1,1)
        phi=np.random.uniform(0,2*np.pi)
        r3=np.random.uniform(0,(10/40)**3)
        r=r3**(1/3)
        v=np.array([r*np.sin(np.arccos(costheta))*np.cos(phi), r*np.sin(np.arccos(costheta))*np.sin(phi), r*costheta])
        d=v[x1]+np.sqrt((10/40)**2-v[x2]**2-v[-1]**2)
        P=1
        # P=np.exp(-d/mu)
        if 1==np.random.choice([0,1], size=1,  p=[1-P, P]):
            V=np.vstack((V,v))
    return V[1:]


def make_d(iter):
    Q, N, v, i=iter
    np.random.seed(int(i*time.time()%(2**32)))
    s=np.zeros(len(Q))
    costheta=np.random.uniform(-1,1, N)
    phi=np.random.uniform(0,2*np.pi, N)
    us=np.vstack((np.vstack((np.sin(np.arccos(costheta))*np.cos(phi), np.sin(np.arccos(costheta))*np.sin(phi))), costheta))
    vs=np.repeat(v, N).reshape(3, N)
    # count=np.zeros(N)
    # while np.any(np.sqrt(np.sum(vs**2, axis=0))<0.75):
    #
    #     absorb=np.nonzero(count>10)[0]
    #     vs[0,absorb]=2
    #     us[:,absorb]=vs[:,absorb]
    #
    #     ind_LXe=np.nonzero(np.sqrt(np.sum(vs**2, axis=0))<=0.25)[0]
    #     ind_toLXe=np.nonzero(np.logical_and(np.sqrt(np.sum(vs**2, axis=0))>0.25, np.sum(vs*us, axis=0)<=0))[0]
    #     ind_toVac=np.nonzero(np.logical_and(np.logical_and(np.sqrt(np.sum(vs**2, axis=0))<0.75, np.sqrt(np.sum(vs**2, axis=0))>0.25), np.sum(vs*us, axis=0)>0))[0]
    #
    #     count[ind_LXe]+=1
    #     count[ind_toLXe]+=1
    #     count[ind_toVac]+=1
    #
    #     if len(ind_LXe)>0:
    #         vs[:,ind_LXe], us[:,ind_LXe]=traceLXe(vs[:,ind_LXe], us[:,ind_LXe], nLXe, sigma_smr)
    #     if len(ind_toLXe)>0:
    #         vs[:,ind_toLXe], us[:,ind_toLXe]=tracetoLXe(vs[:,ind_toLXe], us[:,ind_toLXe], nLXe, sigma_smr)
    #     if len(ind_toVac)>0:
    #         vs[:,ind_toVac], us[:,ind_toVac]=tracetoVac(vs[:,ind_toVac], us[:,ind_toVac], nLXe, sigma_smr)

    pmt_hit=whichPMT(v, us)
    for j in range(len(Q)):
        #area=np.sum(np.random.normal(1, 1, np.random.binomial(len(np.nonzero(pmt_hit==j)[0]), Q[j])))
        s[j]=np.random.binomial(len(np.nonzero(pmt_hit==j)[0]), Q[j])
        #s[j]=int(np.round(area))
    return s



def smr(vs, sigma_smr):
    x=np.random.uniform(-1,1, len(vs[0]))
    y=np.random.uniform(-1,1, len(vs[0]))
    z=-(vs[0]*x+vs[1]*y)/vs[2]
    rot=np.vstack((np.vstack((x, y)), z))
    rot=rot/np.sqrt(np.sum(rot**2, axis=0))
    theta=np.random.normal(0, sigma_smr, len(x))
    return rot*np.sum(rot*vs, axis=0)+np.cos(theta)*np.cross(np.cross(rot, vs, axis=0), rot, axis=0)+np.sin(theta)*np.cross(rot, vs, axis=0)



def traceLXe(vs, us, nLXe, sigma):
    nHPFS=1.6
    # us and vs is an (3,N) array
    a=(np.sqrt(np.sum(vs*us, axis=0)**2+(0.25**2-np.sum(vs**2, axis=0)))-np.sum(us*vs, axis=0)) # N len array

    vmin=np.amin(np.sqrt(np.sum(vs**2, axis=0)))
    ind=np.argmin(np.sqrt(np.sum(vs**2, axis=0)))
    vs=vs+us*a
    rot=np.cross(us,smr(vs, sigma), axis=0)
    rot=rot/np.sqrt(np.sum(rot**2, axis=0))
    inn=np.arccos(np.sum(vs*us, axis=0)/np.sqrt(np.sum(vs**2, axis=0))) # N len array
    TIR=np.nonzero(np.sin(inn)*nLXe/nHPFS>1)[0]
    Rif=np.nonzero(np.sin(inn)*nLXe/nHPFS<=1)[0]

    if len(Rif)>0:
        out=np.arcsin(np.sin(inn[Rif])*nLXe/nHPFS) # N len array
        theta=inn[Rif]-out
        us[:,Rif]=rot[:,Rif]*np.sum(rot[:,Rif]*us[:,Rif], axis=0)+np.cos(theta)*np.cross(np.cross(rot[:,Rif], us[:,Rif], axis=0), rot[:,Rif], axis=0)+np.sin(theta)*np.cross(rot[:,Rif], us[:,Rif], axis=0)

    if len(TIR)>0:
        theta=-(np.pi-2*inn[TIR])
        us[:,TIR]=rot[:, TIR]*np.sum(rot[:,TIR]*us[:,TIR], axis=0)+np.cos(theta)*np.cross(np.cross(rot[:,TIR], us[:,TIR], axis=0), rot[:,TIR], axis=0)+np.sin(theta)*np.cross(rot[:,TIR], us[:,TIR], axis=0)

    us=us/np.sqrt(np.sum(us*us, axis=0))
    return vs+1e-6*us, us

def tracetoLXe(vs, us, nLXe, sigma):
    nHPFS=1.6
    # us and vs is an (3,N) array
    d=np.sum(vs*us, axis=0)**2+(0.25**2-np.sum(vs**2, axis=0))
    toLXe=np.nonzero(d>=0)[0]
    toHPFS=np.nonzero(d>=0)[0]
    if len(toLXe)>0:
        a=(-np.sqrt(d[:, toLXe])-np.sum(us[:, toLXe]*vs[:, toLXe], axis=0)) # N len array
        vs[:, toLXe]=vs[:, toLXe]+us[:, toLXe]*a
        rot=np.cross(us[:, toLXe], smr(-vs[:, toLXe], sigma), axis=0)
        rot=rot/np.sqrt(np.sum(rot**2, axis=0))
        inn=np.pi-np.arccos(np.sum(vs[:, toLXe]*us[:, toLXe], axis=0)/np.sqrt(np.sum(vs[:, toLXe]**2, axis=0))) # N len array
        TIR=np.nonzero(np.sin(inn)*nHPFS/nLXe>1)[0]
        Rif=np.nonzero(np.sin(inn)*nHPFS/nLXe<=1)[0]

        if len(Rif)>0:
            out=np.arcsin(np.sin(inn[Rif])*nHPFS/nLXe) # N len array
            theta=inn[Rif]-out
            us[:,toLXe[Rif]]=rot[:,Rif]*np.sum(rot[:,Rif]*us[:,toLXe[Rif]], axis=0)+np.cos(theta)*np.cross(np.cross(rot[:,Rif], us[:,toLXe[Rif]], axis=0),
                rot[:,Rif], axis=0)+np.sin(theta)*np.cross(rot[:,Rif], us[:,toLXe[Rif]], axis=0)

        if len(TIR)>0:
            theta=-(np.pi-2*inn[TIR])
            us[:,ToLXe[TIR]]=rot[:, TIR]*np.sum(rot[:,TIR]*us[:,ToLXe[TIR]], axis=0)+np.cos(theta)*np.cross(np.cross(rot[:,TIR], us[:,TIR], axis=0),
                rot[:,TIR], axis=0)+np.sin(theta)*np.cross(rot[:,TIR], us[:,ToLXe[TIR]], axis=0)
    if len(toHPFS)>0:
        vs[:,toHPFS], us[:,toHPFS]=tracetoVac(vs[:,toHPFS], us[:,toHPFS])
    us=us/np.sqrt(np.sum(us*us, axis=0))
    return vs+1e-6*us, us


def tracetoVac(vs, us, nLXe, sigma):
    nHPFS=1.6
    # us and vs is an (3,N) array
    a=(np.sqrt(np.sum(vs*us, axis=0)**2+(0.75**2-np.sum(vs**2, axis=0)))-np.sum(us*vs, axis=0)) # N len array
    vs=vs+us*a
    rot=np.cross(us, smr(vs, sigma), axis=0)
    rot=rot/np.sqrt(np.sum(rot**2, axis=0))
    inn=np.arccos(np.sum(vs*us, axis=0)/np.sqrt(np.sum(vs**2, axis=0))) # N len array
    TIR=np.nonzero(np.sin(inn)*nHPFS>1)[0]
    Rif=np.nonzero(np.sin(inn)*nHPFS<=1)[0]

    if len(Rif)>0:
        out=np.arcsin(np.sin(inn[Rif])*nHPFS) # N len array
        theta=inn[Rif]-out
        us[:,Rif]=rot[:,Rif]*np.sum(rot[:,Rif]*us[:,Rif], axis=0)+np.cos(theta)*np.cross(np.cross(rot[:,Rif], us[:,Rif], axis=0), rot[:,Rif], axis=0)+np.sin(theta)*np.cross(rot[:,Rif], us[:,Rif], axis=0)

    if len(TIR)>0:
        theta=-(np.pi-2*inn[TIR])
        us[:,TIR]=rot[:,TIR]*np.sum(rot[:,TIR]*us[:,TIR], axis=0)+np.cos(theta)*np.cross(np.cross(rot[:,TIR], us[:,TIR], axis=0), rot[:,TIR], axis=0)+np.sin(theta)*np.cross(rot[:,TIR], us[:,TIR], axis=0)
    us=us/np.sqrt(np.sum(us*us, axis=0))

    return vs+1e-6*us, us





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
