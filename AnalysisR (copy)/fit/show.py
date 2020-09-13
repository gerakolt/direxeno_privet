import numpy as np
import time
import os
import sys
from scipy.stats import poisson, binom
from scipy.special import erf as erf
from admin import make_glob_array
import multiprocessing
from Sim import Sim_fit
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from make_3D import make_3D
# from L import L
from rebin import rebin_spectrum

pmts=[0,1,4,7,8,14]
TB=1564926608911-1564916365644
TA=1564916315672-1564886605156
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
binsSpec=data['binsSpec']
spectra=data['spectra']
bins=data['bins']
left=data['left']
right=data['right']
cov=data['cov']

path='/home/gerak/Desktop/DireXeno/190803/BG/EventRecon/'
data=np.load(path+'H.npz')
BGH=data['H'][:50,:,:]
BGG=data['G']
BGspectrum=data['spectrum']
BGspectra=data['spectra']
cov=data['cov']


t=np.arange(200)
dt=1
if type=='B':
    x1=1
    x2=0
    T=TB
elif type=='':
    x1=0
    x2=1
    T=TA
if source=='Co57':
    gamma=122
elif source=='Cs137':
    gamma=662


p=([0.19838883,  0.13568442,  0.10828627,  0.17653182,  0.17463287,  0.28635258,
  1.39186269,  1.30583897, 14.15082216])
Q, nLXe, mu, W=make_glob_array(p)
Sspectrum=np.zeros((10, len(binsSpec)-1))
Sspectra=np.zeros((10, len(bins)-1, 6))

fig, ax=plt.subplots(3,5)
for i in range(10):
    print(i)
    s, ss, Scov=Sim_fit(x1, x2, left, right, gamma, Q, 13.7, nLXe, mu, binsSpec, bins)
    Sspectrum[i]=T/TBG*BGspectrum+(np.sum(spectrum)-T/TBG*np.sum(BGspectrum))*s
    Sspectra[i]=T/TBG*BGspectra+(np.sum(spectra, axis=0)-T/TBG*np.sum(BGspectra, axis=0))*ss
    for j in range(15):
        np.ravel(ax)[j].plot(i, Scov[j], 'ko')

j=0
for i in range(5):
    for k in range(i+1,6):
        np.ravel(ax)[j].axhline(cov[j], 0,1, color='r', label='Cov\n PMT{}-PMT{}'.format(i, k), linewidth=5)
        np.ravel(ax)[j].legend(fontsize=15)
        # np.ravel(ax)[j].set_ylim(-70, 40)
        j+=1


plt.figure()
plt.bar(0.5*(binsSpec[1:]+binsSpec[:-1]), spectrum, width=binsSpec[1:]-binsSpec[:-1], label='data')
plt.bar(0.5*(binsSpec[1:]+binsSpec[:-1]), BGspectrum*T/TBG, width=binsSpec[1:]-binsSpec[:-1], label='Background')
plt.plot(0.5*(binsSpec[1:]+binsSpec[:-1]), np.mean(Sspectrum, axis=0), 'k.', label='Simulation')
plt.errorbar(0.5*(binsSpec[1:]+binsSpec[:-1]), np.mean(Sspectrum, axis=0), np.std(Sspectrum, axis=0), fmt='k.', linewidth=3)
plt.xlabel('PEs', fontsize=25)


fig, ax=plt.subplots(2,3)
for i in range(6):
    np.ravel(ax)[i].bar(0.5*(bins[1:]+bins[:-1]), spectra[:,i], width=(bins[1:]-bins[:-1]), label='Data PMT {}'.format(i))
    np.ravel(ax)[i].bar(0.5*(bins[1:]+bins[:-1]), T/TBG*BGspectra[:,i], width=(bins[1:]-bins[:-1]), label='BG'.format(i))
    np.ravel(ax)[i].errorbar(0.5*(bins[1:]+bins[:-1]), np.mean(Sspectra[:,:,i], axis=0), np.std(Sspectra[:,:,i], axis=0), fmt='k.', linewidth=3)
    fig.text(0.45, 0.04, 'PEs in PMT', fontsize=25)


plt.legend()
plt.show()
