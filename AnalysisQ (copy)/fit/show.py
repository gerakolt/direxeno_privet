import numpy as np
import time
import os
import sys
from scipy.stats import poisson, binom
from scipy.special import erf as erf
from minimize import rec_to_p
from admin import make_glob_array
import multiprocessing
from Sim import Sim_fit
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from make_3D import make_3D
from L import L
from rebin import rebin_spectra

pmts=[0,1,4,7,8,14]
TB=1564825612162-1564824285761
TBG=1564874707904-1564826183355

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
data=np.load(path+'H.npz')
H=data['H'][:50,:,:]
G=data['G']
spectrum=data['spectrum']
bins, spectra=rebin_spectra(data['spectra'])
left=data['left']
right=data['right']
cov=data['cov']
Xcov=data['Xcov']
N=data['N']

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
    ])


p=[0.23744822,  0.19335455,  0.27713303,  0.2391398 ,  0.2424144 ,  0.27420061,
 14.68521394 , 0.93384985]

Q, W, mu=make_glob_array(p)

Sspectra=Sim_fit(x1, x2, left, right, gamma, Q[:], W[0], mu[0], bins)

# PEs, spectra=rebin_spectra(spectra)
# PEs, Sspectra=rebin_spectra(Sspectra)

fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].bar(0.5*(bins[1:]+bins[:-1]), spectra[:,i], linewidth=5, width=bins[1:]-bins[:-1], align='center', label='spectrum - PMT{}'.format(pmts[i]), alpha=0.5)
    np.ravel(ax)[i].plot(0.5*(bins[1:]+bins[:-1]), N*Sspectra[:,i], 'r.', label='sim')
    np.ravel(ax)[i].errorbar(0.5*(bins[1:]+bins[:-1]), N*Sspectra[:,i], N*np.sqrt(Sspectra[:,i]/10000), fmt='r.', linewidth=3)
    np.ravel(ax)[i].legend()
plt.show()
