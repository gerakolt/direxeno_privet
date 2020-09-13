import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.stats import poisson, binom
from scipy.special import erf as erf
from fun import make_3D2, Sim2
from minimize import minimize, make_ps
from PMTgiom import make_mash
from fun import make_3D, Sim


start_time = time.time()
pmts=[0,1,4,7,8,14]
path='/home/gerak/Desktop/DireXeno/190803/Co57B/EventRecon/'
rec=np.load(path+'recon1ns.npz')['rec']
blw_cut=15
init_cut=20
chi2_cut=5000
left=170
right=230

rec=rec[np.all(rec['init_wf']>20, axis=1)]
rec=rec[np.sqrt(np.sum(rec['blw']**2, axis=1))<blw_cut]
rec=rec[np.sqrt(np.sum(rec['chi2']**2, axis=1))<chi2_cut]
init=np.sum(np.sum(rec['h'][:,:10,:], axis=2), axis=1)
full=np.sum(np.sum(rec['h'], axis=2), axis=1)
rec=rec[init/full<0.5]

up=np.sum(rec['h'][:,:100,0], axis=1)+np.sum(rec['h'][:,:100,0], axis=1)
dn=np.sum(rec['h'][:,:100,-1], axis=1)+np.sum(rec['h'][:,:100,-2], axis=1)+np.sum(rec['h'][:,:100,-3], axis=1)
rec=rec[dn<3*up+18]

spectrum=np.histogram(np.sum(np.sum(rec['h'], axis=1), axis=1), bins=np.arange(1000)-0.5)[0]
rec=rec[np.sum(np.sum(rec['h'], axis=1), axis=1)>left]
rec=rec[np.sum(np.sum(rec['h'], axis=1), axis=1)<right]

H=rec['h'][:,:100,:]
S=np.sum(H, axis=1)
M=np.mean(S, axis=0)

cov=np.zeros((len(rec), 15))
k=0
for i in range(len(pmts)-1):
    for j in range(i+1, len(pmts)):
        cov[:,k]=(S[:,i]-M[i])*(S[:,j]-M[j])/(M[i]*M[j])
        k+=1

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

path='/home/gerak/Desktop/DireXeno/190803/Co57B/EventRecon/'
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


rec[0]=( [0.22115049,  0.16550132,  0.13854544,  0.22185697,  0.22404989,  0.36315742],
 [42.77823954, 42.63822317, 42.57538105, 42.67443359, 42.67509404, 42.68730998],
  [1.12231578,  0.62974626,  0.55487258,  1.08325085,  0.8779115,   0.8899565],
  0.09593308,  0.78911899, 31.75915812,  0.54248159,  0.35601112,  0.21202624)

# N=60*662
N=122*65
PEs=np.arange(100)
r_mash, V_mash, dS=make_mash(pmts)

fig, ax=plt.subplots(2,3)
fig, bx=plt.subplots(3,5)

for i in range(len(pmts)):
    np.ravel(ax)[i].plot(PEs, spectra[:len(PEs),i], 'ko', label='spectrum - PMT{}'.format(pmts[i]))
for k in range(15):
    h, bins,pa=np.ravel(bx)[k].hist(cov[:,k], bins=7, range=[-0.1,0.1], histtype='step', label='data')
    np.ravel(bx)[k].legend()
x=0.5*(bins[1:]+bins[:-1])
for mu in [0.5]:
    # s, GS, GS_spectrum, Sspectra, Gtrp, Gsng, GRtrp, GRsng, Scov=Sim2(N, rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0], rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0], PEs, mu)
    m, s_model, Mcov=make_3D2(t, N, rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0], rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0], dS, PEs, r_mash, V_mash, x)
    # for i in range(len(pmts)):
        # np.ravel(ax)[i].plot(PEs, np.sum(spectra[:,0])*Sspectra[:,i]/np.sum(Sspectra[:,0]), '.-', label='sim, mu={}'.format(mu))
        # np.ravel(ax)[i].legend()
    for k in range(15):
        # h,bins=np.histogram(Mcov[:,k], bins=7, range=[-0.1,0.1])
        # np.ravel(bx)[k].plot(x, h/len(Scov)*len(cov), 'o')
        np.ravel(bx)[k].plot(x, Mcov[:,k]*len(cov), 'o')
        # np.ravel(bx)[k].legend()

plt.show()
