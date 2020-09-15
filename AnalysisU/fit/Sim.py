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
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors



def Sim_fit(x1, x2, left, right, gamma, Q, T ,St, W, std, nLXe, sigma_smr, mu, R, a, F, Tf, Ts, Bins, bins):
    spectra=np.zeros((len(bins)-1, len(Q)))
    N_events=1000
    H=np.zeros((15,100,len(Q)))
    G=np.zeros((50, 100))
    N=np.array([]).astype(int)
    while len(N)<N_events:
        n=np.round(np.random.normal(1000*gamma/W, std, N_events-len(N))).astype(int)
        N=np.append(N, n[n>0])
    v=make_v(N_events, mu, x1, x2)
    p=multiprocessing.Pool(processes=2)
    ret=p.map(make_d, make_iter(N, Q, T, St, nLXe, sigma_smr, R, a, F, Tf, Ts, v))
    p.close()
    p.join()
    Ret=np.array(ret)
    s=np.sum(Ret, axis=1)
    up=s[:,0]+s[:,1]
    dn=s[:,-1]+s[:,-2]+s[:,-3]
    ind=np.nonzero(np.logical_and(np.logical_and(np.sum(s, axis=1)>=left, np.sum(s, axis=1)<=right), dn<3*up+18))[0]
    if len(ind)==0:
        print('len in<0')
        w=np.abs(np.mean(np.sum(s, axis=1))-(left+right)/2)
        return np.zeros(len(Bins)-1)-w, spectra-w, G, H

    Ret=Ret[ind]
    for i in range(np.shape(Ret)[1]):
        G[:,i]=np.histogram(np.sum(Ret[:,i,:], axis=1), bins=np.arange((np.shape(G)[0]+1))-0.5)[0]
        for j in range(len(Q)):
            H[:,i,j]=np.histogram(Ret[:,i,j], bins=np.arange((np.shape(H)[0]+1))-0.5)[0]

    spectrum=np.histogram(np.sum(np.sum(Ret, axis=1), axis=1), bins=Bins)[0]
    for i in range(len(Q)):
        spectra[:,i]=np.histogram(np.sum(Ret[:,:,i], axis=1), bins=bins)[0]
    return spectrum/len(ind), spectra/len(ind), G/len(ind), H/len(ind)


def make_v(N, mu, x1, x2):
    vs=np.zeros((3, N))
    count=0
    while count<N:
        d=np.random.exponential(mu, N-count)
        d=d[d<0.5]
        c=len(d)
        x=d-0.25
        r=np.random.uniform(0, np.sqrt(0.25**2-x**2), c)
        phi=np.random.uniform(0,2*np.pi, c)
        y=r*np.cos(phi)
        z=r*np.sin(phi)
        vs[x1, count:count+c]=x
        vs[x2, count:count+c]=y
        vs[2, count:count+c]=z
        count+=c
    # costheta=np.random.uniform(-1,1, N)
    # phi=np.random.uniform(0,2*np.pi, N)
    # r3=np.random.uniform(0,(10/40)**3, N)
    # r=r3**(1/3)
    #v=np.array([r*np.sin(np.arccos(costheta))*np.cos(phi), r*np.sin(np.arccos(costheta))*np.sin(phi), r*costheta])
    return vs


def make_d(iter):
    N, Q, T, St, nLXe, sigma_smr, R, a, F, Tf, Ts, v, i=iter
    np.random.seed(int(i*time.time()%(2**32)))
    h=np.zeros((200,len(Q)))

    Strig=2
    trig=np.random.normal(0, Strig, 1)
    recomb=np.random.binomial(N, R)
    ex=N-recomb
    t=np.zeros(recomb+ex)
    u=np.random.uniform(size=recomb)
    t[:recomb]+=1/a*(u/(1-u))
    sng_ind=np.random.choice(2, size=N, replace=True, p=[F, 1-F])
    t[sng_ind==0]+=np.random.exponential(Tf, len(t[sng_ind==0]))
    t[sng_ind==1]+=np.random.exponential(Ts, len(t[sng_ind==1]))

    costheta=np.random.uniform(-1,1, N)
    phi=np.random.uniform(0,2*np.pi, N)
    us=np.vstack((np.vstack((np.sin(np.arccos(costheta))*np.cos(phi), np.sin(np.arccos(costheta))*np.sin(phi))), costheta))
    vs=np.repeat(v, N).reshape(3, N)
    count=np.zeros(N)
    while np.any(np.sqrt(np.sum(vs**2, axis=0))<0.75):
        absorb=np.nonzero(count>10)[0]
        vs[0,absorb]=2
        us[:,absorb]=vs[:,absorb]
        ind_LXe=np.nonzero(np.sqrt(np.sum(vs**2, axis=0))<=0.25)[0]
        ind_toLXe=np.nonzero(np.logical_and(np.sqrt(np.sum(vs**2, axis=0))>0.25, np.sum(vs*us, axis=0)<=0))[0]
        ind_toVac=np.nonzero(np.logical_and(np.logical_and(np.sqrt(np.sum(vs**2, axis=0))<0.75, np.sqrt(np.sum(vs**2, axis=0))>0.25), np.sum(vs*us, axis=0)>0))[0]

        count[ind_LXe]+=1
        count[ind_toLXe]+=1
        count[ind_toVac]+=1

        if len(ind_LXe)>0:
            vs[:,ind_LXe], us[:,ind_LXe]=traceLXe(vs[:,ind_LXe], us[:,ind_LXe], nLXe, sigma_smr)
        if len(ind_toLXe)>0:
            vs[:,ind_toLXe], us[:,ind_toLXe]=tracetoLXe(vs[:,ind_toLXe], us[:,ind_toLXe], nLXe, sigma_smr)
        if len(ind_toVac)>0:
            vs[:,ind_toVac], us[:,ind_toVac]=tracetoVac(vs[:,ind_toVac], us[:,ind_toVac], nLXe, sigma_smr)
    pmt_hit=whichPMT(vs, us)
    t0=np.zeros(len(Q))+1000
    for j in range(len(Q)):
        hit_ind=np.nonzero(pmt_hit==j)[0]
        PE_extrct=hit_ind[np.nonzero(np.random.choice(2, size=len(hit_ind), replace=True, p=[Q[j], 1-Q[j]])==0)[0]]
        tj=np.random.normal(trig+T[j]+t[PE_extrct], St[j], len(PE_extrct))
        h[:,j]=np.histogram(tj, bins=np.arange(np.shape(h)[0]+1)-0.5)[0]
        if np.any(h[:,j]>0):
            t0[j]=np.amin(np.nonzero(h[:,j]>0)[0])

    for j in range(len(Q)):
        h[:,j]=np.roll(h[:,j], -int(np.amin(t0)))
    return h[:100]



def smr(vs, sigma_smr):
    # return vs
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
