import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf
from Sim import Sim

pmts=np.array([0,1,4,7,8,14])

path='/home/gerak/Desktop/DireXeno/190803/pulser/EventRecon/'
data=np.load(path+'delays.npz')
H=data['H']
DC=data['DC']
bins=data['BinsDelay']
names=data['names']

note=''
t0=time.time()
def L(p, pmt, q, ps, ls, ID):
    global t0
    T=p[:6]
    St=p[6:]
    if np.any(St<=0):
        return 1e10*(1-np.amin(p))

    S=Sim(T, St, bins)
    if np.any(np.isnan(S)):
        return 1e10*np.amax(np.abs(T))

    l=0
    for i in range(len(T)):
        data=H[i, np.argmax(H[i])-3:np.argmax(H[i])+4]
        model=S[i, np.argmax(H[i])-3:np.argmax(H[i])+4]*np.amax(H[i])
        l-=np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

    ps[len(ls)]=p
    ls.append(l)
    print('PMT: ', pmt, 'ID: ', ID)
    print(len(ls), l)
    np.savez('Q{}_{}'.format(pmt, ID), param=q, ps=ps, ls=ls, T=time.time()-t0, note=note)
    return l
