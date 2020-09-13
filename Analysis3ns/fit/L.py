from admin import make_glob_array
import numpy as np
import time
import os
import sys
from scipy.stats import poisson, binom
from scipy.special import erf as erf
from Sim import Sim_fit
import multiprocessing



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
path='/home/gerak/Desktop/DireXeno/190803/'+source+'/EventRecon/'
data=np.load(path+'H3ns.npz')
H=data['H'][:50,:,:]
G=data['G']
spectrum=data['spectrum']
spectra=data['spectra']
left=data['left']
right=data['right']
cov=data['cov']
Xcov=data['Xcov']


path='/home/gerak/Desktop/DireXeno/190803/'+source+'B/EventRecon/'
data=np.load(path+'H3ns.npz')
HB=data['H'][:50,:,:]
GB=data['G']
spectrumB=data['spectrum']
spectraB=data['spectra']
leftB=data['left']
rightB=data['right']
covB=data['cov']
XcovB=data['Xcov']

if source=='Co57':
    gamma=122
elif source=='Cs137':
    gamma=662

PEbins=data['PEbins']
t=np.arange(200)
dt=1

counter=0
ls=[]
ps=np.zeros((10000, 3*len(pmts)+7))
t0=time.time()

def L(p):
    global counter, ls, ps, t0
    if np.any(p<0):
        print('p<0')
        return 1e10*(1-np.amin(p))
    Q, T, St, mu, W, F, Tf, Ts, R, a=make_glob_array(p)
    if np.any(np.array(Q)[:]>1):
        print('Q>1')
        return 1e10*np.amax(Q)
    if np.any(np.array(F)[:]>1):
        print('F>1')
        return 1e10*np.amax(F)
    if np.any(np.array(R)[:]>1):
        print('R>1')
        return 1e10*np.amax(R)
    if np.any(np.array(a)[:]>1):
        print('A>1')
        return 1e10*np.amax(a)

    la=multiprocessing.Value('d', 0.0)
    lb=multiprocessing.Value('d', 0.0)

    B=multiprocessing.Process(target=make_l, args=(lb, HB, spectraB, covB, leftB, rightB, XcovB, PEbins, 1, 0, gamma, Q[:], T[:], St[:], mu[0], W[0], F[0], Tf[0], Ts[0], R[0], a[0]))
    A=multiprocessing.Process(target=make_l, args=(la, H, spectra, cov, left, right, Xcov, PEbins, 0, 1, gamma, Q[:], T[:], St[:], mu[0], W[0], F[0], Tf[0], Ts[0], R[0], a[0]))

    B.start()
    A.start()
    B.join()
    A.join()

    l=0
    for i in range(len(pmts)-1):
        for j in range(i+1, len(pmts)):
            x=delays[names=='{}_{}'.format(pmts[i], pmts[j])]
            data=delay_hs[names=='{}_{}'.format(pmts[i], pmts[j])]
            rng=np.nonzero(np.logical_and(x>x[np.argmax(data)]-3, x<x[np.argmax(data)]+3))[0]
            model=(x[1]-x[0])*np.exp(-0.5*(x[rng]-T[j]+T[i])**2/(St[i]**2+St[j]**2))/np.sqrt(2*np.pi*(St[i]**2+St[j]**2))
            # model=(x[1]-x[0])*np.exp(-0.5*(x[rng])**2/(St[i]**2+St[j]**2))/np.sqrt(2*np.pi*(St[i]**2+St[j]**2))
            model=model/np.amax(model)*np.amax(data)
            data=data[rng]
            l+=np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

    l=-(l+la.value+lb.value)
    ps[counter]=p
    ls.append(l)
    print(counter, l)
    counter+=1
    if counter%10==0:
        np.savez(source, ps=ps, ls=ls, T=time.time()-t0)
    return l



def make_l(l, H, spectra, cov, left, right, Xcov, PEbins, x1, x2, gamma, Q, T, St, mu, W, F, Tf, Ts, R, a):
    S, Sspectra, Scov=Sim_fit(x1, x2, left, right, gamma, Q, T, St, mu, W, F, Tf, Ts, R, a, PEbins)
    if np.any(S<0):
        print('S<0')
        l.value=-1e10*(1-np.amin(S))
    else:
        model=np.sum(H[:,0,0])*np.ravel(S)
        data=np.ravel(H[:,:,:])
        l.value+=np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

        data=np.ravel(spectra)
        model=np.sum(H[:,0,0])*np.ravel(Sspectra)
        l.value+=np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

        model=np.sum(H[:,0,0])*np.ravel(Scov)
        data=np.ravel(cov)
        l.value+=np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)
