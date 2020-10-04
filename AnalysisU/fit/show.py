import numpy as np
import time
import os
import sys
from scipy.stats import poisson, binom
from scipy.special import erf as erf
from admin import make_glob_array
import multiprocessing
from Sim_show import Sim_fit
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from make_3D import make_3D
from L import L
from rebin import rebin_spectrum

pmts=[0,1,4,7,8,14]



# path='/home/gerak/Desktop/DireXeno/190803/pulser/DelayRecon/'
# delays=[]
# for pmt in pmts:
#     if pmt!=14:
#         data=np.load(path+'delay_hist{}-14.npz'.format(pmt))
#         delays.append(data['m'])
#     else:
#         delays.append(0)
# path='/home/gerak/Desktop/DireXeno/190803/pulser/EventRecon/'
# data=np.load(path+'delay_list.npz')
# HDelay=data['HDelay']
# BinsDelay=data['BinsDelay']
# DelayNames=data['names']
#
# Abins=[]
# Areas=np.zeros((len(pmts), 14))
# for i, pmt in enumerate(pmts):
#     path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
#     data=np.load(path+'areas.npz')
#     Areas[i]=data['Areas']
#     Abins.append(data['Abins'])


source='Co57'
type='B'
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


p=([2.42113876e-01, 1.67631209e-01, 1.58076323e-01, 2.16066806e-01,
 2.00606154e-01, 2.25645993e-01, 4.07734489e+01, 4.13619837e+01,
 4.05362765e+01, 4.04138555e+01, 4.07258581e+01, 4.11255294e+01,
 5.46874571e-01, 8.21390422e-01, 4.26925194e-01, 7.19220196e-01,
 5.05166708e-01, 7.48014047e-01, 1000, 1000, 1000, 1000, 1000, 1000, 3.71239620e-01, 2.09395868e-01,
 1.79299737e+01, 1.06977242e+02, 1.67719833e+00, 4.77128877e-01,
 4.24762658e-01, 3.41243978e-01, 2.82956642e-01, 5.06594842e-02,
 9.21813118e+00, 2.78242776e+01])

N=10
Q, T, St, AmpS, Nbg, W, std, nLXe, sigma_smr, mu, R, a, F, Tf, Ts=make_glob_array(p)
Sspectrum=np.zeros((N, len(binsSpec)-1))
Sspectra=np.zeros((N, len(bins)-1, 6))
SH=np.zeros((N, np.shape(H)[0], np.shape(H)[1], np.shape(H)[2]))
SG=np.zeros((N, np.shape(G)[0], np.shape(G)[1]))
SD=np.zeros((N, 4, np.shape(G)[0], np.shape(G)[1]))

for i in range(N):
    l=0
    # l=L(np.array(p), -100, -100, np.zeros((1, len(p))), [], 1000)
    print(i, l)
    # spectrum, spectra, g, h=Sim_fit(x1, x2, left, right, gamma, Q, T, St, W, std, nLXe, sigma_smr, mu, R, a, F, Tf, Ts, binsSpec, bins)
    # Sspectrum[i]=Nbg[I]*BGspectrum+(np.sum(Spectrum)-Nbg[I]*np.sum(BGspectrum))*spectrum
    # Sspectra[i]=Nbg[I]*BGspectra+(np.sum(Spectra, axis=0)-Nbg[I]*np.sum(BGspectra, axis=0))*spectra
    # SH[i]=Nbg[I]*BGH+(np.sum(H[:,0,:], axis=0)-Nbg[I]*np.sum(BGH[:,0,:], axis=0))*h
    # SG[i]=Nbg[I]*BGG+(np.sum(G[:,0])-Nbg[I]*np.sum(BGG[:,0]))*g[0]
    # SD[i]=np.sum(G[:,0])*g[1:]


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
    np.ravel(ax)[i].errorbar(np.arange(np.shape(H)[1]), np.mean(np.sum(np.transpose(SH[:,:,:,i], (0,2,1))*np.arange(np.shape(H)[0]), axis=-1), axis=0),
        np.std(np.sum(np.transpose(SH[:,:,:,i], (0,2,1))*np.arange(np.shape(H)[0]), axis=-1), axis=0), fmt='k.', linewidth=3)
    np.ravel(ax)[i].legend()

plt.figure()
plt.bar(np.arange(np.shape(G)[1]), np.sum(G.T*np.arange(np.shape(G)[0]), axis=1), width=1, label='Global')
plt.bar(np.arange(np.shape(G)[1]), Nbg[I]*np.sum(BGG.T*np.arange(np.shape(G)[0]), axis=1), width=1)
plt.errorbar(np.arange(np.shape(G)[1]), np.mean(np.sum(np.transpose(SG, (0,2,1))*np.arange(np.shape(G)[0]), axis=-1), axis=0),
    np.std(np.sum(np.transpose(SG, (0,2,1))*np.arange(np.shape(G)[0]), axis=-1), axis=0), fmt='k.', linewidth=3)
label=['Ex 1', 'Ex 3', 'Recomb 1', 'Recomb 3']
for i in range(4):
    plt.errorbar(np.arange(np.shape(G)[1]), np.mean(np.sum(np.transpose(SD[:,i], (0,2,1))*np.arange(np.shape(G)[0]), axis=-1), axis=0),
        np.std(np.sum(np.transpose(SD[:,i], (0,2,1))*np.arange(np.shape(G)[0]), axis=-1), axis=0), fmt='.', linewidth=3, label=label[i])
plt.legend()

# fig, ax=plt.subplots(3,5)
# r=np.linspace(BinsDelay[0], BinsDelay[1], 100)
# I=np.arange(len(r)*(len(BinsDelay)-1))
# i=0
# for j in range(len(pmts)-1):
#     for k in range(j, len(pmts)):
#         np.ravel(ax)[i].bar(0.5*(BinsDelay[1:]+BinsDelay[:-1]), HDelay[i], width=BinsDelay[1:]-BinsDelay[:-1])
#         model=np.sum(AmpS[i]*np.exp(-0.5*(BinsDelay[I%(len(BinsDelay)-1)]+r[I//(len(BinsDelay)-1)]-T[i]+delays[i])**2/St[i]**2).reshape((len(r), len(BinsDelay)-1)), axis=0)*(BinsDelay[1]-BinsDelay[0])/len(r)
#         np.ravel(ax)[i].errorbar(0.5*(BinsDelay[1:]+BinsDelay[:-1]), model, np.sqrt(model), fmt='.', linewidth=3, label=DelayNames[i])
# plt.show()
