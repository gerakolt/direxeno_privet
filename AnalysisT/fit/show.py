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
from L import L
from rebin import rebin_spectrum

pmts=[0,1,4,7,8,14]



# path='/home/gerak/Desktop/DireXeno/190803/pulser/DelayRecon/'
# delay_hs=[]
# names=[]
# delays=[]
# for i in range(len(pmts)-1):
#     for j in range(i+1, len(pmts)):
#         data=np.load(path+'delay_hist{}-{}.npz'.format(pmts[i], pmts[j]))
#         delays.append(data['x']-data['m'])
#         delay_hs.append(data['h'])
#         names.append('{}_{}'.format(pmts[i], pmts[j]))

Abins=[]
Areas=np.zeros((len(pmts), 14))
for i, pmt in enumerate(pmts):
    path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
    data=np.load(path+'areas.npz')
    Areas[i]=data['Areas']
    Abins.append(data['Abins'])


source='Co57'
type=''
path='/home/gerak/Desktop/DireXeno/190803/'+source+type+'/EventRecon/'
data=np.load(path+'H.npz')
H=data['H']
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
BGH=data['H']
BGG=data['G']
BGspectrum=data['spectrum']
BGspectra=data['spectra']
cov=data['cov']


t=np.arange(200)
dt=1
TBG=1564874707904-1564826183355
if source=='Co57':
    gamma=122
    if type=='B':
        x1=1
        x2=0
        I=1
        T=1564926608911-1564916365644
    elif type=='':
        I=0
        x1=0
        x2=1
        T=1564916315672-1564886605156
elif source=='Cs137':
    BGspectrum=spectrum*0
    BGspectra=spectra*0
    gamma=662
    if type=='B':
        x1=1
        x2=0
        T=1564825612162-1564824285761
    elif type=='':
        x1=0
        x2=1
        T=1564823506349-1564820774226


p=([1.99595565e-01, 1.10855491e-01, 1.16931933e-01, 1.64855939e-01,
 1.71385413e-01, 1.75494553e-01, 1.10843989e+00, 7.60654309e-01,
 4.98808286e-01, 6.27130735e-01, 9.39779196e-01, 8.96174105e-01,
 4.77687664e-01, 2.80100432e-01 ,3.90831974e+01, 6.87363235e+02])

N=10
Q, Sa, W, std, Nbg=make_glob_array(p)
Sspectrum=np.zeros((N, len(binsSpec)-1))
Sspectra=np.zeros((N, len(bins)-1, 6))
SAreas=np.zeros((N, len(Q), len(Abins[0])-1))
# fig, ax=plt.subplots(3,5)
for i in range(N):
    l=0
    # l=L(np.array(p), -100, -100, np.zeros((1, len(p))), [], 1000)
    print(i, l)
    s, ss, sa=Sim_fit(x1, x2, left, right, gamma, Q, Sa, W, std, binsSpec, bins, Abins)
    Sspectrum[i]=Nbg[I]*BGspectrum+(np.sum(spectrum)-Nbg[I]*np.sum(BGspectrum))*s
    Sspectra[i]=Nbg[I]*BGspectra+(np.sum(spectra, axis=0)-Nbg[I]*np.sum(BGspectra, axis=0))*ss
    SAreas[i]=(np.sum(Areas, axis=1)*sa).T



plt.figure()
plt.bar(0.5*(binsSpec[1:]+binsSpec[:-1]), spectrum, width=binsSpec[1:]-binsSpec[:-1], label='data')
plt.bar(0.5*(binsSpec[1:]+binsSpec[:-1]), BGspectrum*Nbg[I], width=binsSpec[1:]-binsSpec[:-1], label='Background', alpha=0.5)
plt.errorbar(0.5*(binsSpec[1:]+binsSpec[:-1]), np.mean(Sspectrum, axis=0), np.std(Sspectrum, axis=0), fmt='k.', linewidth=3)
plt.xlabel('PEs', fontsize=25)

fig, ax=plt.subplots(2,3)
for i in range(6):
    np.ravel(ax)[i].bar(0.5*(bins[1:]+bins[:-1]), spectra[:,i], width=(bins[1:]-bins[:-1]), label='Data PMT {}'.format(i))
    np.ravel(ax)[i].bar(0.5*(bins[1:]+bins[:-1]), Nbg[I]*BGspectra[:,i], width=(bins[1:]-bins[:-1]), label='BG'.format(i), alpha=0.5)
    np.ravel(ax)[i].errorbar(0.5*(bins[1:]+bins[:-1]), np.mean(Sspectra[:,:,i], axis=0), np.std(Sspectra[:,:,i], axis=0), fmt='k.', linewidth=3)
    fig.text(0.45, 0.04, 'PEs in PMT', fontsize=25)

fig, ax=plt.subplots(2,3)
for i in range(6):
    np.ravel(ax)[i].bar(0.5*(Abins[i][1:]+Abins[i][:-1]), Areas[i], width=(Abins[i][1:]-Abins[i][:-1]), label='Areas PMT {}'.format(i))
    np.ravel(ax)[i].errorbar(0.5*(Abins[i][1:]+Abins[i][:-1]), np.mean(SAreas[:,i,:], axis=0), np.std(SAreas[:,i,:], axis=0), fmt='k.', linewidth=3)
    np.ravel(ax)[i].legend()
# fig, ax=plt.subplots(2,3)
# for i in range(6):
#     np.ravel(ax)[i].bar(areas[i], H_areas[i], label='Areas PMT {}'.format(i))
#     # np.ravel(ax)[i].plot(areas[i], H_areas[i], label='Areas PMT {}'.format(i))
#     np.ravel(ax)[i].set_yscale('log')


plt.legend()
plt.show()
