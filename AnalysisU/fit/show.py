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
Spectrum=data['spectrum']
binsSpec=data['binsSpec']
Spectra=data['spectra']
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


p=([2.14999828e-01, 1.23153368e-01, 1.29865201e-01, 1.89851059e-01*27/29,
 1.90290844e-01*26/33, 1.73430939e-01, 4.01650009e+01, 4.00359737e+01,
 4.17480666e+01, 4.01341080e+01, 3.96873314e+01, 4.19519739e+01,
 4.67867077e-01, 5.80644915e-01, 3.96173931e-01, 4.79202121e-01,
 9.95341087e-01, 6.07327970e-01, 0 ,0,
 1.36778815e+01*75/170 ,8.31860880e+01 ,1.58733152e+00, 9.79290651e-01,
 5.58119499e-01, 8.17402810e-01, 1.79443937e-06, 2.98232484e-01,
 9.30616752e+00, 2.97927685e+01])

N=3
Q, T, St, Nbg, W, std, nLXe, sigma_smr, mu, R, a, F, Tf, Ts=make_glob_array(p)
Sspectrum=np.zeros((N, len(binsSpec)-1))
Sspectra=np.zeros((N, len(bins)-1, 6))
SH=np.zeros((N, np.shape(H)[0], np.shape(H)[1], np.shape(H)[2]))
SG=np.zeros((N, np.shape(G)[0], np.shape(G)[1]))

for i in range(N):
    l=0
    # l=L(np.array(p), -100, -100, np.zeros((1, len(p))), [], 1000)
    print(i, l)
    spectrum, spectra, g, h=Sim_fit(x1, x2, left, right, gamma, Q, T, St, W, std, nLXe, sigma_smr, mu, R, a, F, Tf, Ts, binsSpec, bins)
    Sspectrum[i]=Nbg[I]*BGspectrum+(np.sum(Spectrum)-Nbg[I]*np.sum(BGspectrum))*spectrum
    Sspectra[i]=Nbg[I]*BGspectra+(np.sum(Spectra, axis=0)-Nbg[I]*np.sum(BGspectra, axis=0))*spectra
    SH[i]=Nbg[I]*BGH+(np.sum(H[:,0,:], axis=0)-Nbg[I]*np.sum(BGH[:,0,:], axis=0))*h
    SG[i]=Nbg[I]*BGG+(np.sum(G[:,0])-Nbg[I]*np.sum(BGG[:,0]))*g


plt.figure()
plt.bar(0.5*(binsSpec[1:]+binsSpec[:-1]), Spectrum, width=binsSpec[1:]-binsSpec[:-1], label='data')
plt.bar(0.5*(binsSpec[1:]+binsSpec[:-1]), BGspectrum*Nbg[I], width=binsSpec[1:]-binsSpec[:-1], label='Background', alpha=0.5)
plt.errorbar(0.5*(binsSpec[1:]+binsSpec[:-1]), np.mean(Sspectrum, axis=0), np.std(Sspectrum, axis=0), fmt='k.', linewidth=3)
plt.xlabel('PEs', fontsize=25)

fig, ax=plt.subplots(2,3)
for i in range(6):
    np.ravel(ax)[i].bar(0.5*(bins[1:]+bins[:-1]), Spectra[:,i], width=(bins[1:]-bins[:-1]), label='Data PMT {}'.format(i))
    np.ravel(ax)[i].bar(0.5*(bins[1:]+bins[:-1]), Nbg[I]*BGspectra[:,i], width=(bins[1:]-bins[:-1]), label='BG'.format(i), alpha=0.5)
    np.ravel(ax)[i].errorbar(0.5*(bins[1:]+bins[:-1]), np.mean(Sspectra[:,:,i], axis=0), np.std(Sspectra[:,:,i], axis=0), fmt='k.', linewidth=3)
    fig.text(0.45, 0.04, 'PEs in PMT', fontsize=25)


fig, ax=plt.subplots(2,3)
for i in range(6):
    np.ravel(ax)[i].bar(np.arange(np.shape(H)[1]), np.sum(H[:,:,i].T*np.arange(np.shape(H)[0]), axis=1), width=1, label='PMT {}'.format(i))
    np.ravel(ax)[i].bar(np.arange(np.shape(H)[1]), Nbg[I]*np.sum(BGH[:,:,i].T*np.arange(np.shape(H)[0]), axis=1), width=1)
    np.ravel(ax)[i].errorbar(np.arange(np.shape(H)[1]), np.mean(np.sum(np.transpose(SH[:,:,:,i], (0,2,1))*np.arange(np.shape(H)[0]), axis=-1), axis=0), np.std(np.sum(np.transpose(SH[:,:,:,i], (0,2,1))*np.arange(np.shape(H)[0]), axis=-1), axis=0), fmt='k.', linewidth=3)
    np.ravel(ax)[i].legend()

plt.figure()
plt.bar(np.arange(np.shape(G)[1]), np.sum(G.T*np.arange(np.shape(G)[0]), axis=1), width=1, label='Global')
plt.bar(np.arange(np.shape(G)[1]), Nbg[I]*np.sum(BGG.T*np.arange(np.shape(G)[0]), axis=1), width=1)
plt.errorbar(np.arange(np.shape(G)[1]), np.mean(np.sum(np.transpose(SG, (0,2,1))*np.arange(np.shape(G)[0]), axis=-1), axis=0), np.std(np.sum(np.transpose(SG, (0,2,1))*np.arange(np.shape(G)[0]), axis=-1), axis=0), fmt='k.', linewidth=3)

plt.legend()
plt.show()
