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

pmts=[0,1,4,7,8,14]


path='/home/gerak/Desktop/DireXeno/190803/Co57/EventRecon/'
data=np.load(path+'H.npz')
H=data['H'][:30,:,:]
G=data['G']
spectrum=data['spectrum']
spectra=data['spectra']
left=data['left']
right=data['right']
PEs=np.arange(len(spectra[:,0]))
N=65*122


def make_dS(d, pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn):
    dS=np.zeros(len(pmt_mid))
    ETA=np.linspace(-1,1,100, endpoint=True)
    zeta=np.linspace(-1,1,100, endpoint=True)
    deta=ETA[1]-ETA[0]
    for i in range(len(dS)):
        for eta in ETA:
            dS[i]+=np.sum(pmt_r[i]*pmt_r[i])*deta**2*np.sum((1-np.sum(d*pmt_mid[i]))/((1-2*np.sum(d*pmt_mid[i])+np.sum(d*d)-2*eta*np.sum(d*pmt_r[i])-2*zeta*np.sum(d*pmt_up[i])
                            +eta**2*np.sum(pmt_r[i]*pmt_r[i])+zeta**2*np.sum(pmt_up[i]*pmt_up[i]))**(3/2)))
    return dS/(4*np.pi)

pmts=[0,1,4,7,8,14]

pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn, r=make_pmts(pmts)
dS=make_dS(np.array([0,0,0]),  pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn)

def make_recomb(t, a, eta):
    dt=t[1]-t[0]
    ni=np.ones(len(t))
    ne=np.ones(len(t))*(1-eta)
    for i in range(1, len(t)):
        ni[i]=ni[i-1]*(1-dt*a*ne[i-1])
        ne[i]=ne[i-1]*(1-dt*a*ni[i-1])
    return dt*a*(ni*ne)/(1-eta)

def Sim(PEs, N, F, Tf, Ts, R, a, eta, Q, T, St, sigma, a0):
    f=np.sum(make_recomb(np.arange(1000*20)/100, a, eta).reshape(1000,20), axis=1)
    f[-1]=1-np.sum(f[:-1])
    N_events=2000
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
    spectra=np.zeros((len(PEs),len(Q)))
    for i in range(N_events):
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
            if sigma>0:
                for k in range(len(h)):
                    if h[k]>0:
                        area=np.sum(np.random.normal(1, sigma, size=h[k]))
                        if area<a0:
                            h[k]=0
                        elif area<1:
                            h[k]=1
                        else:
                            h[k]=int(np.round(area))
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
    for i in range(len(Q)):
        spectra[:,i]=np.histogram(np.sum(d[:,:,i], axis=1), bins=np.arange(len(PEs)+1)-0.5)[0]
    for k in range(200):
        G[:,k]=np.histogram(np.sum(d[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]

        Gtrp[:,k]=np.histogram(np.sum(trp[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        Gsng[:,k]=np.histogram(np.sum(sng[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        GRtrp[:,k]=np.histogram(np.sum(Rtrp[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        GRsng[:,k]=np.histogram(np.sum(Rsng[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]

        for j in range(len(Q)):
            H[:,k,j]=np.histogram(d[:,k,j], bins=np.arange(np.shape(H)[0]+1)-0.5)[0]
    return H/N_events, G/N_events, spectrum, spectra, Gtrp/N_events, Gsng/N_events, GRtrp/N_events, GRsng/N_events

rec=np.recarray(1, dtype=[
    ('Q', 'f8', len(pmts)),
    ('T', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('F', 'f8', 1),
    ('Tf', 'f8', 1),
    ('Ts', 'f8', 1),
    ('R', 'f8', 1),
    ('a', 'f8', 1),
    ('eta', 'f8', 1),
    ])

rec[0]=([0.21747557,  0.1648596,   0.13315825,  0.21394634,  0.2189225,   0.34596169],
 [42.13272964, 42.08280408, 42.35549289, 42.03699013, 42.19650918, 42.40075324],
  [0.83201371,  0.52560077,  1.59897881,  0.72090362,  0.88857833,  0.83077755],
  0.07721134,  1.98666512, 32.80634845,  0.40617278,  0.46942869,  0.21018914)

t=np.arange(200)
fig1, ax1=plt.subplots(2,3)
fig1.text(0.04, 0.5, r'$N_{events}\sum_n nH_{ni}$', va='center', rotation='vertical', fontsize=15)
for i in range(len(pmts)):
    np.ravel(ax1)[i].plot(t, np.sum(H[:,:,i].T*np.arange(np.shape(H)[0]), axis=1)/np.sum(H[:,0,i]), 'ko', label='Data - PMT{}'.format(pmts[i]))
    np.ravel(ax1)[i].set_xlabel('Time [ns]', fontsize='15')

fig2, ax2=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax2)[i].plot(PEs, spectra[:,i], 'ko', label='spectrum - PMT{}'.format(pmts[i]))
    np.ravel(ax2)[i].legend()

fig3, ax3=plt.subplots(1,1)
ax3.plot(t, np.sum(G.T*np.arange(np.shape(G)[0]), axis=1), 'ko', label='Global Data')
ax3.legend()

sigma=0
a0=0
s, GS, GS_spectrum, Sspectra, Gtrp, Gsng, GRtrp, GRsng=Sim(PEs, N, rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0], rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0], sigma, a0)

for i in range(len(pmts)):
    np.ravel(ax1)[i].plot(t, np.sum(s[:,:,i].T*np.arange(np.shape(s)[0]), axis=1), '.-', label='{}, {}'.format(sigma, a0), linewidth=3)


Sspectra=Sspectra/np.amax(Sspectra, axis=0)*np.amax(spectra, axis=0)
for i in range(len(pmts)):
    np.ravel(ax2)[i].plot(PEs, Sspectra[:,i], '.-', label='{}, {}'.format(sigma, a0))

ax3.plot(t, np.sum(G[:,0])*np.sum(GS.T*np.arange(np.shape(GS)[0]), axis=1), '-.', label='{}, {}'.format(sigma, a0))

for i, sigma in enumerate(np.linspace(0.3,0.9,5)):
    for j, a0 in enumerate(np.linspace(0.1,0.35,5)):
        print(i,j)
        s, GS, GS_spectrum, Sspectra, Gtrp, Gsng, GRtrp, GRsng=Sim(PEs, N, rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0], rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0], sigma, a0)

        for i in range(len(pmts)):
            np.ravel(ax1)[i].plot(t, np.sum(s[:,:,i].T*np.arange(np.shape(s)[0]), axis=1), '.-', label='{}, {}'.format(sigma, a0), linewidth=3)
            np.ravel(ax1)[i].legend()

        Sspectra=Sspectra/np.amax(Sspectra, axis=0)*np.amax(spectra, axis=0)
        for i in range(len(pmts)):
            np.ravel(ax2)[i].plot(PEs, Sspectra[:,i], '.-', label='{}, {}'.format(sigma, a0))
            np.ravel(ax2)[i].legend()

        ax3.plot(t, np.sum(G[:,0])*np.sum(GS.T*np.arange(np.shape(GS)[0]), axis=1), '-.', label='{}, {}'.format(sigma, a0))

plt.show()
