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
source='Co57'
try:
    path='/home/gerak/Desktop/DireXeno/190803/'+source+'/EventRecon/'
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
    BGspectrum=spectrumA*0
    BGspectra=spectraA*0
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
    Q, Sa, W,g=make_glob_array(p)
    if np.any(np.array(Q)[:]>1):
        print('Q>1')
        return 1e20*np.amax(Q)


    la=multiprocessing.Value('d', 0.0)
    lb=multiprocessing.Value('d', 0.0)
    B=multiprocessing.Process(target=make_l, args=(len(ls)+1, lb, HB, spectrumB, spectraB, TB, binsSpec, bins, covB, left, right, 1, 0, gamma, Q, W, Sa, g))
    A=multiprocessing.Process(target=make_l, args=(len(ls)+1, la, HA, spectrumA, spectraA, TA, binsSpec, bins, covB, left, right, 0, 1, gamma, Q, W, Sa, g))

    B.start()
    A.start()
    B.join()
    A.join()
    l=la.value+lb.value
    ps[len(ls)]=p
    ls.append(l)
    print(len(ls), time.time()-t1)
    t1=time.time()
    np.savez('Q{}_{}'.format(pmt, ID), param=q, ps=ps, ls=ls, T=time.time()-t0, note=note)
    return l



def make_l(i, l, H, spectrum, spectra, T, Bins, bins, cov, left, right, x1, x2, gamma, Q, W, Sa, g):
    np.random.seed(int(i*time.time()%2**32))
    Sspectrum, Sspectra, Scov=Sim_fit(x1, x2, left, right, gamma, Q, W, Sa, g , binsSpec, bins)
    if np.any(Sspectrum<0):
        l.value=1e20*(1-np.amin(Sspectrum))
    else:
        data=spectrum
        model=T/TBG*BGspectrum+(np.sum(spectrum)-T/TBG*np.sum(BGspectrum))*Sspectrum
        l.value=-np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

        data=np.ravel(spectra)
        model=np.ravel(T/TBG*BGspectra+(np.sum(spectra, axis=0)-T/TBG*np.sum(BGspectra, axis=0))*Sspectra)
        l.value=-np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)
