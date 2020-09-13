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
from make_mash import mash

pmts=[0,1,4,7,8,14]
r_mash, V_mash, dS=mash(pmts)

path='/home/gerak/Desktop/DireXeno/190803/Cs137B/EventRecon/'
data=np.load(path+'H.npz')
H=data['H'][:30,:,:]
G=data['G']
spectrum=data['spectrum']
spectra=data['spectra']
left=data['left']
right=data['right']
t=np.arange(200)
dt=t[1]-t[0]

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


rec[0]=([0.19657476,  0.13754033,  0.12743771,  0.18797336,  0.17835696,  0.3510241],
 [42.00312603, 42.07819445, 42.03561186, 42.05469875, 42.03181254 ,42.13596326],
  [0.94580769,  0.61208912,  0.84663691,  1.25148529,  0.78060014,  0.59422144],
  0.09440092,  2.06581567, 37.59474049,  0.68417731,  0.454177,    0.19242887)


def make_recomb(t, a, eta):
    dt=t[1]-t[0]
    ni=np.ones(len(t))
    ne=np.ones(len(t))*(1-eta)
    for i in range(1, len(t)):
        ni[i]=ni[i-1]*(1-dt*a*ne[i-1])
        ne[i]=ne[i-1]*(1-dt*a*ni[i-1])
    return dt*a*(ni*ne)/(1-eta)

def whichPMT(v, us, mid, rt, up):
    hits=np.zeros(len(us[0]))-1
    for i in range(len(mid)):
        a=(1-np.sum(mid[i]*v, axis=0))/np.sum(us.T*mid[i], axis=1)
        r=v+(a*us).T-mid[i]
        hits[np.nonzero(np.logical_and(a>0, np.logical_and(np.abs(np.sum(r*rt[i], axis=1))<np.sum(rt[i]**2), np.abs(np.sum(r*up[i], axis=1))<np.sum(up[i]**2))))[0]]=i
    return hits
pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn=make_pmts(pmts)

def Sim2(N, F, Tf, Ts, R, a, eta, Q, T, St, PEs, prm):
    f=np.sum(make_recomb(np.arange(1000*20)/100, a, eta).reshape(1000,20), axis=1)
    f[-1]=1-np.sum(f[:-1])
    N_events=10000
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
        print('in sim', i, 'prm=',prm)
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
        if prm==0:
            costheta=np.random.uniform(-1,1)
            phi=np.random.uniform(0,2*np.pi)
            r3=np.random.uniform(0,(10/40)**3)
            r=r3**(1/3)
            v=np.array([r*np.sin(np.arccos(costheta))*np.cos(phi), r*np.sin(np.arccos(costheta))*np.sin(phi), r*costheta])
            # v=r_mash[np.argmin(np.sum((r_mash-v)**2, axis=1))]
        elif prm==1:
            costheta=np.random.uniform(-1,1)
            phi=np.random.uniform(0,2*np.pi)
            while phi>np.pi/2 and phi<3*np.pi/2:
                phi=np.random.uniform(0,2*np.pi)
            r3=np.random.uniform(0,(10/40)**3)
            r=r3**(1/3)
            v=np.array([r*np.sin(np.arccos(costheta))*np.cos(phi), r*np.sin(np.arccos(costheta))*np.sin(phi), r*costheta])
        elif prm==2:
            costheta=np.random.uniform(-1,1)
            phi=np.random.uniform(0,2*np.pi)
            while phi>np.pi:
                phi=np.random.uniform(0,2*np.pi)
            r3=np.random.uniform(0,(10/40)**3)
            r=r3**(1/3)
            v=np.array([r*np.sin(np.arccos(costheta))*np.cos(phi), r*np.sin(np.arccos(costheta))*np.sin(phi), r*costheta])
        elif prm==3:
            costheta=np.random.uniform(-1,1)
            phi=np.random.uniform(0,2*np.pi)
            while phi<np.pi/2 or phi>3*np.pi/2:
                phi=np.random.uniform(0,2*np.pi)
            r3=np.random.uniform(0,(10/40)**3)
            r=r3**(1/3)
            v=np.array([r*np.sin(np.arccos(costheta))*np.cos(phi), r*np.sin(np.arccos(costheta))*np.sin(phi), r*costheta])
        elif prm==4:
            costheta=np.random.uniform(-1,1)
            phi=np.random.uniform(0,2*np.pi)
            while phi<np.pi:
                phi=np.random.uniform(0,2*np.pi)
            r3=np.random.uniform(0,(10/40)**3)
            r=r3**(1/3)
            v=np.array([r*np.sin(np.arccos(costheta))*np.cos(phi), r*np.sin(np.arccos(costheta))*np.sin(phi), r*costheta])

        costheta=np.random.uniform(-1,1, len(t))
        phi=np.random.uniform(0,2*np.pi, len(t))
        us=np.vstack((np.vstack((np.sin(np.arccos(costheta))*np.cos(phi), np.sin(np.arccos(costheta))*np.sin(phi))), costheta))
        pmt_hit=whichPMT(v, us, pmt_mid, pmt_r, pmt_up)
        for j in range(len(Q)):
            hits=np.nonzero(pmt_hit==j)[0]
            ind=np.nonzero(1==np.random.choice(2, size=len(hits), replace=True, p=[1-Q[j], Q[j]]))[0]
            tj=np.random.normal(trig+T[j]+t[hits[ind]], St[j], len(ind))
            h, bins=np.histogram(tj, bins=np.arange(201))
            if np.any(h>0):
                t0[j]=np.amin(np.nonzero(h>0)[0])
            d[i,:,j]=h
            trp[i,:,j]=np.histogram(tj[np.nonzero(np.logical_and(hits[ind]>=recomb, np.isin(hits[ind], slow_i)))[0]], bins=np.arange(201))[0]
            sng[i,:,j]=np.histogram(tj[np.nonzero(np.logical_and(hits[ind]>=recomb, np.isin(hits[ind], fast_i)))[0]], bins=np.arange(201))[0]
            Rtrp[i,:,j]=np.histogram(tj[np.nonzero(np.logical_and(hits[ind]<recomb, np.isin(hits[ind], slow_i)))[0]], bins=np.arange(201))[0]
            Rsng[i,:,j]=np.histogram(tj[np.nonzero(np.logical_and(hits[ind]<recomb, np.isin(hits[ind], fast_i)))[0]], bins=np.arange(201))[0]
        for j in range(len(Q)):
            d[i,:,j]=np.roll(d[i,:,j], -int(np.amin(t0)))
            trp[i,:,j]=np.roll(trp[i,:,j], -int(np.amin(t0)))
            sng[i,:,j]=np.roll(sng[i,:,j], -int(np.amin(t0)))
            Rtrp[i,:,j]=np.roll(Rtrp[i,:,j], -int(np.amin(t0)))
            Rsng[i,:,j]=np.roll(Rsng[i,:,j], -int(np.amin(t0)))

    spectrum=np.histogram(np.sum(np.sum(d, axis=2), axis=1), bins=np.arange(1000)-0.5)[0]
    for i in range(len(Q)):
        spectra[:,i]=np.histogram(np.sum(d[:,:100,i], axis=1), bins=np.arange(len(PEs)+1)-0.5)[0]
    for k in range(200):
        G[:,k]=np.histogram(np.sum(d[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]

        Gtrp[:,k]=np.histogram(np.sum(trp[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        Gsng[:,k]=np.histogram(np.sum(sng[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        GRtrp[:,k]=np.histogram(np.sum(Rtrp[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        GRsng[:,k]=np.histogram(np.sum(Rsng[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]

        for j in range(len(Q)):
            H[:,k,j]=np.histogram(d[:,k,j], bins=np.arange(np.shape(H)[0]+1)-0.5)[0]
    return H/N_events, G/N_events, spectrum, spectra, Gtrp/N_events, Gsng/N_events, GRtrp/N_events, GRsng/N_events

#
# PEs=np.arange(len(spectra[:,0]))
# s, GS, GS_spectrum, Sspectra0, Gtrp, Gsng, GRtrp, GRsng=Sim2(662*60, rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0], rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0], PEs, 0)
# s, GS, GS_spectrum, Sspectra1, Gtrp, Gsng, GRtrp, GRsng=Sim2(662*60, rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0], rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0], PEs, 1)
# s, GS, GS_spectrum, Sspectra2, Gtrp, Gsng, GRtrp, GRsng=Sim2(662*60, rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0], rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0], PEs, 2)
# s, GS, GS_spectrum, Sspectra3, Gtrp, Gsng, GRtrp, GRsng=Sim2(662*60, rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0], rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0], PEs, 3)
# s, GS, GS_spectrum, Sspectra4, Gtrp, Gsng, GRtrp, GRsng=Sim2(662*60, rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0], rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0], PEs, 4)
#
# fig, ax=plt.subplots(2,3)
# for i in range(len(pmts)):
#     np.ravel(ax)[i].plot(PEs, spectra[:,i], 'ko', label='spectrum - PMT{}'.format(pmts[i]))
#     np.ravel(ax)[i].plot(PEs, np.sum(spectra[:,0])*Sspectra0[:,i]/np.sum(Sspectra0[:,0]), '.-', label='sim0')
#     np.ravel(ax)[i].plot(PEs, np.sum(spectra[:,0])*Sspectra1[:,i]/np.sum(Sspectra1[:,0]), '.-', label='sim1')
#     np.ravel(ax)[i].plot(PEs, np.sum(spectra[:,0])*Sspectra2[:,i]/np.sum(Sspectra2[:,0]), '.-', label='sim2')
#     np.ravel(ax)[i].plot(PEs, np.sum(spectra[:,0])*Sspectra3[:,i]/np.sum(Sspectra3[:,0]), '.-', label='sim3')
#     np.ravel(ax)[i].plot(PEs, np.sum(spectra[:,0])*Sspectra4[:,i]/np.sum(Sspectra4[:,0]), '.-', label='sim4')
#     np.ravel(ax)[i].legend()
#
# plt.show()
