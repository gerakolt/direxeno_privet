from admin import make_glob_array
import numpy as np
import time
import os
import sys
from scipy.stats import poisson, binom
from scipy.special import erf as erf
from Sim import Sim_fit
import multiprocessing
from rebin import rebin_spectrum


pmts=[0,1,4,7,8,14]
note='Only spectrum only Q, 60 bins of width 1, only one side, 10k events per sim'
#path='/home/gerak/Desktop/DireXeno/190803/pulser/DelayRecon/'
#path='/storage/xenon/gerak/pulser/DelayRecon/'

# delay_hs=[]
# names=[]
# delays=[]
# for i in range(len(pmts)-1):
#     for j in range(i+1, len(pmts)):
#         data=np.load(path+'delay_hist{}-{}.npz'.format(pmts[i], pmts[j]))
#         delays.append(data['x']-data['m'])
#         delay_hs.append(data['h'])
#         names.append('{}_{}'.format(pmts[i], pmts[j]))
#

Abins=[]
Areas=np.zeros((len(pmts), 14))
for i, pmt in enumerate(pmts):
    try:
        path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
        data=np.load(path+'areas.npz')
    except:
        path='/storage/xenon/gerak/pulser/PMT{}/'.format(pmt)
        data=np.load(path+'areas.npz')
    Areas[i]=data['Areas']
    Abins.append(data['Abins'])

source='Co57'
try:
    path='/home/gerak/Desktop/DireXeno/190803/'+source+'/EventRecon/'
    data=np.load(path+'H.npz')
except:
    path='/storage/xenon/gerak/'+source+'/'
    data=np.load(path+'H.npz')

HA=data['H'][:50,:,:]
GA=data['G']
spectrumA=data['spectrum']
binsSpec=data['binsSpec']
spectraA=data['spectra']
bins=data['bins']
left=data['left']
right=data['right']
cov=data['cov']
try:
    path='/home/gerak/Desktop/DireXeno/190803/'+source+'B/EventRecon/'
    data=np.load(path+'H.npz')

except:
    path='/storage/xenon/gerak/'+source+'B/'
    data=np.load(path+'H.npz')

HB=data['H'][:50,:,:]
GB=data['G']
spectrumB=data['spectrum']
binsSpec=data['binsSpec']
spectraB=data['spectra']
covB=data['cov']

try:
    path='/home/gerak/Desktop/DireXeno/190803/BG/EventRecon/'
    data=np.load(path+'H.npz')
except:
    path='/storage/xenon/gerak/BG/'
    data=np.load(path+'H.npz')

BGH=data['H'][:50,:,:]
BGG=data['G']
BGspectrum=data['spectrum']
BGspectra=data['spectra']
cov=data['cov']

TBG=1564874707904-1564826183355
if source=='Co57':
    gamma=122
    TB=1564926608911-1564916365644
    TA=1564916315672-1564886605156
elif source=='Cs137':
    gamma=662
    TB=1564825612162-1564824285761
    TA=1564823506349-1564820774226

t=np.arange(200)
dt=1

counter=0
ls=[]
ps=np.zeros((15000, 6))
t0=time.time()
t1=time.time()
def L(p, pmt, q, ps, ls, ID):
    global t0, t1
    if np.any(p<0):
        print('p<0:', np.nonzero(p<0)[0])
        return 1e20*(1-np.amin(p))
    Q, Sa, W, std, Nbg=make_glob_array(p)
    if np.any(np.array(Q)[:]>1):
        print('Q>1')
        return 1e20*np.amax(Q)
    if (np.sum(spectrumA)-Nbg[0]*np.sum(BGspectrum))<0 or np.any((np.sum(spectraA, axis=0)-Nbg[0]*np.sum(BGspectra, axis=0))<0):
        print('Nbg0 to big')
        return 1e20*(1+Nbg[0])
    if (np.sum(spectrumB)-Nbg[1]*np.sum(BGspectrum))<0 or np.any((np.sum(spectraB, axis=0)-Nbg[1]*np.sum(BGspectra, axis=0))<0):
        print('Nbg1 to big')
        return 1e20*(1+Nbg[1])

    SAreas = multiprocessing.Array("d", np.ravel(np.zeros(np.shape(Areas))))
    la=multiprocessing.Value('d', 0.0)
    lb=multiprocessing.Value('d', 0.0)
    B=multiprocessing.Process(target=make_l, args=(len(ls)+1, lb, SAreas, HB, spectrumB, spectraB, TB, binsSpec, bins, Abins, left, right, 1, 0, gamma, Q, Sa, W, std, Nbg[1]))
    A=multiprocessing.Process(target=make_l, args=(len(ls)+1, la, SAreas, HA, spectrumA, spectraA, TA, binsSpec, bins, Abins, left, right, 0, 1, gamma, Q, Sa, W, std, Nbg[0]))

    B.start()
    A.start()
    B.join()
    A.join()
    l=la.value+lb.value

    data=np.ravel(Areas)
    model=np.ravel(((np.sum(Areas, axis=1)*(np.array(SAreas).reshape(np.shape(Areas))).T)).T)
    l-=np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

    ps[len(ls)]=p
    ls.append(l)
    print(len(ls), time.time()-t1)
    t1=time.time()
    np.savez('Q{}_{}'.format(pmt, ID), param=q, ps=ps, ls=ls, T=time.time()-t0, note=note)
    return l



def make_l(i, l, SAreas, H, spectrum, spectra, T, Bins, bins, Abins, left, right, x1, x2, gamma, Q, Sa, W, std, Nbg):
    np.random.seed(int(i*time.time()%2**32))
    Sspectrum, Sspectra, Sareas=Sim_fit(x1, x2, left, right, gamma, Q, Sa, W, std, binsSpec, bins, Abins)
    if np.any(Sspectrum<0):
        l.value=1e20*(1-np.amin(Sspectrum))
    else:
        data=spectrum
        model=Nbg*BGspectrum+(np.sum(spectrum)-Nbg*np.sum(BGspectrum))*Sspectrum
        l.value=-np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

        data=np.ravel(spectra)
        model=np.ravel(Nbg*BGspectra+(np.sum(spectra, axis=0)-Nbg*np.sum(BGspectra, axis=0))*Sspectra)
        l.value=-np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

        SAreas+=np.ravel(Sareas.T)
