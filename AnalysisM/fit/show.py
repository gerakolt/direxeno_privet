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
from make_3D import make_3D


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
type='B'
path='/home/gerak/Desktop/DireXeno/190803/'+source+'/EventRecon/'
data=np.load(path+'H.npz')
H=data['H'][:50,:,:]
G=data['G']
spectrum=data['spectrum']
spectra=data['spectra']
left=data['left']
right=data['right']
cov=data['cov']
# cov10=data['cov10']
Xcov=data['Xcov']

t=np.arange(200)
dt=1
PEs=np.arange(np.shape(spectra)[0])

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
    ('eta', 'f8', 1),
    ])


Rec[0]=([0.26645133, 0.1852529 , 0.15451606, 0.23476867, 0.24536202, 0.40333365], [43.05346962, 42.873252  , 42.94007243, 43.05813256, 42.9075503 , 42.7691549 ],
 [1.11972955, 0.6902668 , 0.73902005, 1.29871286, 0.97512363, 0.76713902], 0.2, (gamma*1000)/8088.8245986, 0.08845413, 1.2170795, 32.7050479, 0.54599945, 0.42470853, 0.35477914)

p=[0.32378283,  0.19660243,  0.16057405,  0.234686 ,   0.25145273 , 0.41114211,
 44.45625697, 45.03407608, 44.66588412, 44.28857476, 45.08497647, 44.7183324,
  0.85008913,  1.12283851,  1.06490646 , 1.13497033 , 1.14844984,  0.83412358,
  0.65719119 ,12.4121034 ,  0.30306365  ,0.94478687 ,26.42918234 , 0.87484769,
  0.10085858,  0.40190071]
Q, T, St, mu, W, F, Tf, Ts, R, a, eta=make_glob_array(rec_to_p(Rec))
# Q, T, St, mu, W, F, Tf, Ts, R, a, eta=make_glob_array(p)
# M, Mspectra, Mcov=make_3D(x1, x2, gamma,np.array( Q[:]), np.array(T[:]), np.array(St[:]), mu[0], W[0], F[0], Tf[0], Ts[0], R[0], a[0], eta[0], PEs, Xcov)
S, Sspectra, Scov, G, Gtrp, Gsng, GRtrp, GRsng, SN=Sim_show(x1, x2, left, right, gamma, Q[:], T[:], St[:], mu[0], W[0], F[0], Tf[0], Ts[0], R[0], a[0], eta[0], PEs)


N=np.sum(H[:,0,0])
fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].step(t[:100], np.sum(H[:,:100,i].T*np.arange(np.shape(H)[0]), axis=1)/(dt), label='Data - PMT{}'.format(pmts[i]), linewidth=3, where='post')
    # np.ravel(ax)[i].plot(t[:100]+0.5, N*np.sum(M[:,:,i].T*np.arange(np.shape(M)[0]), axis=1)/dt, 'ro', label='model', linewidth=3)
    np.ravel(ax)[i].plot(t[:100]+0.5, N*np.sum(S[:,:100,i].T*np.arange(np.shape(S)[0]), axis=1)/dt, 'r.', label='sim', linewidth=3)
    np.ravel(ax)[i].errorbar(t[:100]+0.5, N*np.sum(S[:,:100,i].T*np.arange(np.shape(S)[0]), axis=1)/dt, N*np.sqrt(np.sum(S[:,:100,i].T*np.arange(np.shape(S)[0]), axis=1)/(dt*SN)), fmt='r.')
    np.ravel(ax)[i].set_xlabel('Time [ns]', fontsize='15')
fig.text(0.04, 0.5, r'$N_{events}\sum_n nH_{ni}$', va='center', rotation='vertical', fontsize=15)

fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].step(PEs, spectra[:,i], linewidth=3, where='mid', label='spectrum - PMT{}'.format(pmts[i]))
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
        model=(x[1]-x[0])*np.exp(-0.5*(x-Rec['T'][0,j]+Rec['T'][0,i])**2/(Rec['St'][0,i]**2+Rec['St'][0,j]**2))/np.sqrt(2*np.pi*(Rec['St'][0,i]**2+Rec['St'][0,j]**2))
        model=model/np.amax(model)*np.amax(data)
        np.ravel(ax)[k].step(x, data, label='Delays {}_{}'.format(pmts[i], pmts[j]), linewidth=3, where='mid')
        np.ravel(ax)[k].plot(x[rng], model[rng], 'r-.')
        np.ravel(ax)[k].set_xlabel('Delay [ns]', fontsize='15')
        np.ravel(ax)[k].legend(fontsize=15)
        k+=1


fig, bx=plt.subplots(3,5)
k=0
for k in range(15):
    np.ravel(bx)[k].step(Xcov, cov[:,k], where='mid', label='full')
    # np.ravel(bx)[k].step(Xcov, cov10[:,k], where='mid', label='10 ns')
    # np.ravel(bx)[k].plot(Xcov, Mcov[:,k]*N, 'r.', label='model')
    np.ravel(bx)[k].plot(Xcov, Scov[:,k]*N, 'g.', label='sim')
    np.ravel(bx)[k].errorbar(Xcov, Scov[:,k]*N, N*np.sqrt(Scov[:,k]/SN), fmt='g.')
    np.ravel(bx)[k].legend()


plt.show()
