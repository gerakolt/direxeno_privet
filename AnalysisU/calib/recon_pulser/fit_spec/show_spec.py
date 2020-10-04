import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf
from Sim import Sim

chn=5
pmts=[0,1,4,7,8,14]
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmts[chn])
data=np.load(path+'areas.npz')
Sa=data['Sa']


blw_cut=20
chi2_cut=2000
left=240
right=380

path='/home/gerak/Desktop/DireXeno/190803/pulser/EventRecon/'
data=np.load(path+'recon1ns.npz')
rec=data['rec']
rec=rec[np.sqrt(np.sum(rec['blw']**2, axis=1))<blw_cut]
rec=rec[np.sqrt(np.sum(rec['chi2']**2, axis=1))<chi2_cut]
h=rec['h']
Abins=data['Abins']
DCAreas=np.sum(rec['DCAreas'], axis=0)
DC=(np.sum(np.sum(h[:,:left], axis=0), axis=0)+np.sum(np.sum(h[:,right:], axis=0), axis=0))/len(h[:,0,0])*1000/(1000-right+left)
spec=np.zeros((len(pmts), 5))
DCspec=np.zeros((len(pmts), 5))
for i in range(len(pmts)):
    spec[i], bins=np.histogram(np.sum(h[:,left:right,i], axis=1), bins=np.arange(6)-0.5)
    DCspec[i], bins=np.histogram((np.sum(h[:,:left ,i], axis=1)+np.sum(h[:,right: ,i], axis=1))*(right-left)/(1000-right+left), bins=np.arange(6)-0.5)

th=np.array([0.45, 0.9, 1, 0.7, 0.7, 0.55])
p=[0.82839096, 0.38661561, 0.27190796]


Ma=p[0]
Sa=p[1]
q=p[2]


iter=10
S=np.zeros((iter, len(spec[chn])))
A=np.zeros((iter, len(DCAreas[chn])))


for j in range(iter):
    print(j)
    S[j], A[j]=Sim(Ma, Sa, q, th[chn], Abins, DCspec[chn], DCAreas[chn])
    l=0
    bool=0.5*(Abins[1:]+Abins[:-1])>th[chn]
    data=np.sum(rec['Areas'][:,chn], axis=0)[bool]
    model=A[j, bool]*np.sum(np.sum(rec['Areas'][:,chn], axis=0)[bool])
    l-=np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)
    print(l)

fig, ax=plt.subplots(1)
np.ravel(ax)[0].step(0.5*(bins[1:]+bins[:-1]), spec[chn], where='mid', label='PMT{}'.format(i), linewidth=3)
np.ravel(ax)[0].step(0.5*(bins[1:]+bins[:-1]), DCspec[chn], where='mid', label='PMT{}'.format(i), linewidth=3)
np.ravel(ax)[0].errorbar(0.5*(bins[1:]+bins[:-1]), np.mean(S, axis=0)*np.sum(spec[chn]), np.std(S, axis=0)*np.sum(spec[chn]), fmt='.', color='k')

np.ravel(ax)[0].legend(fontsize=15)
np.ravel(ax)[0].set_yscale('log')

bool=0.5*(Abins[1:]+Abins[:-1])>th[chn]
fig, ax=plt.subplots(1)
np.ravel(ax)[0].step(0.5*(Abins[1:]+Abins[:-1]), np.sum(rec['Areas'][:,chn,:], axis=0), where='mid', label='PMT{}'.format(chn), linewidth=3)
np.ravel(ax)[0].step(0.5*(Abins[1:]+Abins[:-1]), DCAreas[chn], where='mid', label='PMT{}'.format(chn), linewidth=3)
np.ravel(ax)[0].errorbar(0.5*(Abins[1:]+Abins[:-1])[bool], np.mean(A, axis=0)[bool]*np.sum(np.sum(rec['Areas'][:,chn,:], axis=0)[bool]), np.std(A, axis=0)[bool]*np.sum(np.sum(rec['Areas'][:,chn,:], axis=0)[bool]), fmt='.', color='k')

plt.show()
