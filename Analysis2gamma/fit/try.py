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

source='Co57TEMP'
type=''
path='/home/gerak/Desktop/DireXeno/190803/'+source+type+'/EventRecon/'
data=np.load(path+'H.npz')
HB=data['H'][:50,:,:]
GB=data['G']
spectrumB=data['spectrum']
spectraB=data['spectra']
left=data['left']
right=data['right']
covB=data['cov']
XcovB=data['Xcov']

t=np.arange(200)
dt=1
PEs=np.arange(np.shape(spectra)[0])



N=np.sum(H[:,0,0])
fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].step(t[:100], np.sum(H[:,:100,i].T*np.arange(np.shape(H)[0]), axis=1), label='Data A - PMT{}'.format(pmts[i]), linewidth=3, where='post')
    np.ravel(ax)[i].step(t[:100], np.sum(HB[:,:100,i].T*np.arange(np.shape(HB)[0]), axis=1), label='Data B - PMT{}'.format(pmts[i]), linewidth=3, where='post')

    np.ravel(ax)[i].legend()
fig.text(0.04, 0.5, r'$N_{events}\sum_n nH_{ni}$', va='center', rotation='vertical', fontsize=15)

fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].step(PEs, spectra[:,i], linewidth=3, where='mid', label='spectrum A - PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].step(PEs, spectraB[:,i], linewidth=3, where='mid', label='spectrum B - PMT{}'.format(pmts[i]))

fig, ax=plt.subplots(3,5)
k=0
for i in range(len(pmts)-1):
    for j in range(i+1, len(pmts)):
        x=delays[names=='{}_{}'.format(pmts[i], pmts[j])]
        data=delay_hs[names=='{}_{}'.format(pmts[i], pmts[j])]

        np.ravel(ax)[k].step(x, data, label='Delays {}_{}'.format(pmts[i], pmts[j]), linewidth=3, where='mid')
        np.ravel(ax)[k].set_xlabel('Delay [ns]', fontsize='15')
        np.ravel(ax)[k].legend(fontsize=15)
        k+=1


fig, bx=plt.subplots(3,5)
k=0
for k in range(15):
    np.ravel(bx)[k].step(Xcov, cov[:,k], where='mid', label='full A')
    np.ravel(bx)[k].step(XcovB, covB[:,k], where='mid', label='full B')

    np.ravel(bx)[k].legend()


plt.show()
