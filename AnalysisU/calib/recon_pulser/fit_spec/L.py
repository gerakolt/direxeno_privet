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

Sa=np.zeros(len(pmts))
for i, pmt in enumerate(pmts):
    path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
    data=np.load(path+'areas.npz')
    Sa[i]=data['Sa']


blw_cut=20
chi2_cut=2000
left=240
right=380
print('in L')
path='/home/gerak/Desktop/DireXeno/190803/pulser/EventRecon/'
data=np.load(path+'recon1ns.npz')
rec=data['rec']
rec=rec[np.sqrt(np.sum(rec['blw']**2, axis=1))<blw_cut]
rec=rec[np.sqrt(np.sum(rec['chi2']**2, axis=1))<chi2_cut]
h=rec['h']
Abins=data['Abins']
DCAreas=np.sum(rec['DCAreas'], axis=0)
spec=np.zeros((len(pmts), 5))
DCspec=np.zeros((len(pmts), 5))
for i in range(len(pmts)):
    spec[i], bins=np.histogram(np.sum(h[:,left:right,i], axis=1), bins=np.arange(6)-0.5)
    DCspec[i], bins=np.histogram((np.sum(h[:,:left ,i], axis=1)+np.sum(h[:,right: ,i], axis=1))*(right-left)/(1000-right+left), bins=np.arange(6)-0.5)

t0=time.time()
t1=time.time()
th=np.array([0.45, 0.9, 1, 0.7, 0.7, 0.55])
note=''
def L(p, pmt, q, ps, ls, ID):
    global t0, t1
    Ma=p[0]
    Sa=p[1]
    lamb=p[2]

    if np.any(p<=0):
        return 1e10*(1-np.amin(p))

    S , A=Sim(Ma, Sa, lamb, th[pmt], Abins, DCspec[pmt], DCAreas[pmt])
    l=0
    #
    # data=spec[pmt]
    # model=S*np.sum(spec[pmt])
    # l-=np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

    bool=0.5*(Abins[1:]+Abins[:-1])>th[pmt]
    data=np.sum(rec['Areas'][:,pmt], axis=0)[bool]
    model=A[bool]*np.sum(np.sum(rec['Areas'][:,pmt], axis=0)[bool])
    l-=np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

    ps[len(ls)]=p
    ls.append(l)
    print('PMT: ', pmt, 'ID: ', ID)
    print(len(ls), time.time()-t1, l)
    t1=time.time()
    np.savez('Q{}_{}'.format(pmt, ID), param=q, ps=ps, ls=ls, T=time.time()-t0, note=note)
    return l
