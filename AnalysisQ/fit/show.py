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
from rebin import rebin_spectra, rebin_spectrum

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
type='B'
path='/home/gerak/Desktop/DireXeno/190803/'+source+type+'/EventRecon/'
data=np.load(path+'H.npz')
H=data['H'][:50,:,:]
G=data['G']
bins, spectrum=rebin_spectrum(data['spectrum'])
# bins, spectra=rebin_spectra(data['spectra'])
left=data['left']
right=data['right']
cov=data['cov']
Xcov=data['Xcov']
N=data['N']

# path='/home/gerak/Desktop/DireXeno/190803/BG/EventRecon/'
# data=np.load(path+'H.npz')
# BGH=data['H'][:50,:,:]
# BGG=data['G']
# bins, BGspectrum=rebin_spectrum(data['spectrum'])
# # bins, spectra=rebin_spectra(data['spectra'])
# left=data['left']
# right=data['right']
# cov=data['cov']
# Xcov=data['Xcov']
# BGN=data['N']

t=np.arange(200)
dt=1
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


p=np.array([0.22863552,  0.26201005,  0.26584989,  0.20042843,  0.19058325,  0.25078971])*0.9

Q=make_glob_array(p)
Sspectrum=Sim_fit(x1, x2, left, right, gamma, Q, 13.7, bins)

plt.figure()
plt.bar(0.5*(bins[1:]+bins[:-1]), spectrum, width=bins[1:]-bins[:-1])
plt.plot(0.5*(bins[1:]+bins[:-1]), Sspectrum*N, 'k.')
plt.errorbar(0.5*(bins[1:]+bins[:-1]), Sspectrum*N, N*np.sqrt(Sspectrum/1000), fmt='k.')

plt.show()
