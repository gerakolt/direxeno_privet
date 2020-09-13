import numpy as np
import time
import os
import sys
from scipy.stats import poisson, binom
from scipy.special import erf as erf
from minimize import rec_to_p
from admin import make_glob_array
import multiprocessing
from Sim import Sim_show
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from make_3D import make_3D
from L import L


pmts=[0,1,4,7,8,14]

path='/home/gerak/Desktop/DireXeno/190803/pulser/DelayRecon/'
delay_hs=[]
names=[]
delays=[]
for i in range(len(pmts)-1):
    for j in range(i+1, len(pmts)):
        data=np.load(path+'delay_hist{}-{}.npz'.format(pmts[i], pmts[j]))
        delays.append(data['x']-data['m'])
        delay_hs.append(data['h'])
        names.append('{}_{}'.format(pmts[i], pmts[j]))

source='Co57'
type=''
path='/home/gerak/Desktop/DireXeno/190803/'+source+type+'/EventRecon/'
data=np.load(path+'H3ns.npz')
H=data['H'][:50,:,:]
G=data['G']
spectrum=data['spectrum']
spectra=data['spectra']
left=data['left']
right=data['right']
cov=data['cov']
Xcov=data['Xcov']

type='B'
path='/home/gerak/Desktop/DireXeno/190803/'+source+type+'/EventRecon/'
data=np.load(path+'H3ns.npz')
HB=data['H'][:50,:,:]
GB=data['G']
spectrumB=data['spectrum']
spectraB=data['spectra']
left=data['left']
right=data['right']
covB=data['cov']
XcovB=data['Xcov']
PEbins=data['PEbins']

t=np.arange(33)*3
dt=3
PEs=0.5*(PEbins[1:]+PEbins[:-1])

if type=='B':
    x1=1
    x2=0
elif type=='':
    x1=0
    x2=1
if source=='Co57':
    gamma=122
elif source=='Cs137':
    gamma=662

Rec=np.recarray(1, dtype=[
    ('Q', 'f8', len(pmts)),
    ('T', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('mu', 'f8', 1),
    ('W', 'f8', 1),
    ('F', 'f8', 1),
    ('Tf', 'f8', 1),
    ('Ts', 'f8', 1),
    ('R', 'f8', 1),
    ('a', 'f8', 1),
    ])


# Rec[0]=([0.26645133, 0.1852529 , 0.15451606, 0.23476867, 0.24536202, 0.40333365], [43.05346962, 42.873252  , 42.94007243, 43.05813256, 42.9075503 , 42.7691549 ],
#  [1.11972955, 0.6902668 , 0.73902005, 1.29871286, 0.97512363, 0.76713902], 0.2, (gamma*1000)/8088.8245986, 0.08845413, 1.2170795, 32.7050479, 0.54599945, 0.35477914)

p=[0.21479492,  0.14538561,  0.13073778,  0.1969609,   0.17663971,  0.27376075,
 44.14892478, 43.60143576, 44.66321839, 44.11787161, 44.09023266, 44.10159648,
  0.73568363,  0.79358755,  0.83428189,  0.73767781,  0.7787808,   0.74064659,
  2.36092996, 12.46929755,  0.06050573,  1.03140709, 35.63065966,  0.2260836,
  0.94817856]

#Q, T, St, mu, W, F, Tf, Ts, R, a=make_glob_array(rec_to_p(Rec))
Q, T, St, mu, W, F, Tf, Ts, R, a=make_glob_array(p)
# M, Mspectra, Mcov=make_3D(x1, x2, gamma,np.array( Q[:]), np.array(T[:]), np.array(St[:]), mu[0], W[0], F[0], Tf[0], Ts[0], R[0], a[0], eta[0], PEs, Xcov)

S, Sspectra, Scov, SG, Gtrp, Gsng, GRtrp, GRsng, SN=Sim_show(x1, x2, left, right, gamma, Q[:], T[:], St[:], mu[0], W[0], F[0], Tf[0], Ts[0], R[0], a[0], PEbins)


N=np.sum(H[:,0,0])
fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].step(t+1, np.sum(H[:,:,i].T*np.arange(np.shape(H)[0]), axis=1)/dt, label='Data A - PMT{}'.format(pmts[i]), linewidth=3, where='post')
    # np.ravel(ax)[i].plot(t[:100]+0.5, N*np.sum(M[:,:,i].T*np.arange(np.shape(M)[0]), axis=1)/dt, 'ro', label='model', linewidth=3)
    np.ravel(ax)[i].plot(t+1, N*np.sum(S[:,:,i].T*np.arange(np.shape(S)[0]), axis=1)/dt, 'r.', label='sim', linewidth=3)
    np.ravel(ax)[i].errorbar(t+1, N*np.sum(S[:,:,i].T*np.arange(np.shape(S)[0]), axis=1)/dt, N*np.sqrt(np.sum(S[:,:,i].T*np.arange(np.shape(S)[0]), axis=1)/(dt*SN)), fmt='r.')
    np.ravel(ax)[i].set_xlabel('Time [ns]', fontsize='15')
    np.ravel(ax)[i].legend()
fig.text(0.04, 0.5, r'$N_{events}\sum_n nH_{ni}$', va='center', rotation='vertical', fontsize=15)

fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].step(PEs, spectra[:,i], linewidth=3, where='mid', label='spectrum A - PMT{}'.format(pmts[i]))
    # np.ravel(ax)[i].plot(PEs, N*Mspectra[:,i], 'r.', label='model')
    np.ravel(ax)[i].plot(PEs, N*Sspectra[:,i], 'g.', label='sim')
    np.ravel(ax)[i].errorbar(PEs, N*Sspectra[:,i], N*np.sqrt(Sspectra[:,i]/SN), fmt='g.')
    np.ravel(ax)[i].legend()

fig, ax=plt.subplots(3,5)
k=0
for i in range(len(pmts)-1):
    for j in range(i+1, len(pmts)):
        x=delays[names=='{}_{}'.format(pmts[i], pmts[j])]
        data=delay_hs[names=='{}_{}'.format(pmts[i], pmts[j])]
        rng=np.nonzero(np.logical_and(x>x[np.argmax(data)]-7, x<x[np.argmax(data)]+7))
        # model=(x[1]-x[0])*np.exp(-0.5*(x-T[j]+T[i])**2/(St[i]**2+St[j]**2))/np.sqrt(2*np.pi*(St[i]**2+St[j]**2))
        model=(x[1]-x[0])*np.exp(-0.5*(x)**2/(St[i]**2+St[j]**2))/np.sqrt(2*np.pi*(St[i]**2+St[j]**2))
        model=model/np.amax(model)*np.amax(data)
        np.ravel(ax)[k].step(x, data, label='Delays {}_{}'.format(pmts[i], pmts[j]), linewidth=3, where='mid')
        # np.ravel(ax)[k].plot(x[rng], model[rng], 'r-.')
        np.ravel(ax)[k].set_xlabel('Delay [ns]', fontsize='15')
        np.ravel(ax)[k].legend(fontsize=15)
        k+=1


fig, bx=plt.subplots(3,5)
k=0
for k in range(15):
    np.ravel(bx)[k].step(Xcov, cov[:,k], where='mid', label='full A')
    # np.ravel(bx)[k].step(Xcov, cov10[:,k], where='mid', label='10 ns')
    # np.ravel(bx)[k].plot(Xcov, Mcov[:,k]*N, 'r.', label='model')
    np.ravel(bx)[k].plot(Xcov, Scov[:,k]*N, 'g.', label='sim')
    np.ravel(bx)[k].errorbar(Xcov, Scov[:,k]*N, N*np.sqrt(Scov[:,k]/SN), fmt='g.')
    np.ravel(bx)[k].legend()


plt.figure()
plt.step(t, np.sum(G.T*np.arange(np.shape(G)[0]), axis=1)/dt, where='mid', label='A')
plt.plot(t, np.sum(G[:,0])*np.sum(SG.T*np.arange(np.shape(SG)[0]), axis=1)/dt, 'r.-', label='')
plt.plot(t, np.sum(G[:,0])*np.sum(Gtrp.T*np.arange(np.shape(SG)[0]), axis=1)/dt, 'g.-', label='trp')
plt.plot(t, np.sum(G[:,0])*np.sum(Gsng.T*np.arange(np.shape(SG)[0]), axis=1)/dt, 'b.-', label='sng')
plt.plot(t, np.sum(G[:,0])*np.sum(GRsng.T*np.arange(np.shape(SG)[0]), axis=1)/dt, 'c.-', label='Rsng')
plt.plot(t, np.sum(G[:,0])*np.sum(GRtrp.T*np.arange(np.shape(SG)[0]), axis=1)/dt, 'y.-', label='Rtrp')
plt.legend()


plt.show()
