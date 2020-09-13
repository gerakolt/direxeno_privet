from admin import make_glob_array
import numpy as np
import time
import os
import sys
from scipy.stats import poisson, binom
from scipy.special import erf as erf
from Sim import Sim_fit
import multiprocessing
from rebin import rebin_spectra


pmts=[0,1,4,7,8,14]
note='Only spectrum only Q'
path='/home/gerak/Desktop/DireXeno/190803/pulser/DelayRecon/'
# path='/storage/xenon/gerak/pulser/DelayRecon/'

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
path='/home/gerak/Desktop/DireXeno/190803/'+source+'/EventRecon/'
#path='/storage/xenon/gerak/'+source+'/EventRecon/'

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

path='/home/gerak/Desktop/DireXeno/190803/'+source+'B/EventRecon/'
#path='/storage/xenon/gerak/'+source+'B/EventRecon/'

data=np.load(path+'H.npz')
HB=data['H'][:50,:,:]
GB=data['G']
spectrumB=data['spectrum']
spectraB=data['spectra']
leftB=data['left']
rightB=data['right']
covB=data['cov']
XcovB=data['Xcov']
NB=data['N']


if source=='Co57':
    gamma=122
elif source=='Cs137':
    gamma=662


t=np.arange(200)
dt=1


t0=time.time()


def L(p, param, ind, param_name, ps, ls):
    global t0, NB, N
    if np.any(p<0):
        print('p<0:', np.nonzero(p<0)[0])
        return 1e10*(1-np.amin(p))
    Q=make_glob_array(p)
    if np.any(np.array(Q)[:]>1):
        print('Q>1')
        return 1e10*np.amax(Q)

    la=multiprocessing.Value('d', 0.0)
    lb=multiprocessing.Value('d', 0.0)

    #B=multiprocessing.Process(target=make_l, args=(lb, HB, spectraB, covB, leftB, rightB, XcovB, 1, 0, gamma, Q[:], T[:], St[:], Sa[:], mu[0], W[0], F[0], Tf[0], Ts[0], R[0], a[0], NB))
    A=multiprocessing.Process(target=make_l, args=(la, H, spectra, cov, left, right, Xcov, 0, 1, gamma, Q[:], N))

    #B.start()
    A.start()
    #B.join()
    A.join()

    l=0
    l=-(l+la.value+lb.value)
    ls.append(l)
    ps[len(ls)-1]=p
    if len(ls)%1==0:
        np.savez(source+''+param_name, ps=ps[:len(ls)], ls=ls, T=time.time()-t0, note=note, ind=ind, param=param)
    return l



def make_l(l, H, spectra, cov, left, right, Xcov, x1, x2, gamma, Q, N):
    Sspectra=Sim_fit(x1, x2, left, right, gamma, Q, 13.7, np.arange(np.shape(spectra)[0]))
    if np.any(Sspectra<0):
        l.value=-1e10*(1-np.amin(Sspectra))
    else:
        # model=np.sum(H[:,0,0])*np.ravel(S)
        # data=np.ravel(H[:,:100,:])
        # model=np.sum(H[:,0,0])*np.ravel(np.sum(S, axis=-1))
        # data=np.ravel(np.sum(H[:,:100,:], axis=-1))
        # l.value+=np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)


        spectra=rebin_spectra(spectra)[1]
        Sspectra=rebin_spectra(Sspectra)[1]
        data=np.ravel(spectra)
        model=N*np.ravel(Sspectra)
        l.value+=np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

        # model=np.sum(H[:,0,0])*np.ravel(Scov)
        # data=np.ravel(cov)
        # l.value+=np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)
