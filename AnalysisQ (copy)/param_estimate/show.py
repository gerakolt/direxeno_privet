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
spectra=data['spectra']
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


p=[0.22,       0.19041186, 0.22063317, 0.20403208, 0.20959739, 0.24661499]

Q=make_glob_array(p)

Sspectra=Sim_fit(x1, x2, left, right, gamma, Q[:], 13.7, np.arange(np.shape(spectra)[0]))

PEs, spectra=rebin_spectra(spectra)
PEs, Sspectra=rebin_spectra(Sspectra)

fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].bar(PEs, spectra[:,i], linewidth=5, width=PEs[1]-PEs[0], align='edge', label='spectrum - PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].plot(PEs+0.5*(PEs[1]-PEs[0]), N*Sspectra[:,i], 'g.', label='sim')
    np.ravel(ax)[i].errorbar(PEs+0.5*(PEs[1]-PEs[0]), N*Sspectra[:,i], N*np.sqrt(Sspectra[:,i]/10000), fmt='g.', linewidth=3)
    np.ravel(ax)[i].legend()
plt.show()
