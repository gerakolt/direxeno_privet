import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf




pmts=[0,1,4,7,8,14]
path='/home/gerak/Desktop/DireXeno/190803/Co57B/EventRecon/'
rec=np.load(path+'recon1ns.npz')['rec']
blw_cut=15
init_cut=20
chi2_cut=5000
left=170
right=230

rec=rec[np.all(rec['init_wf']>20, axis=1)]
rec=rec[np.sqrt(np.sum(rec['blw']**2, axis=1))<blw_cut]
rec=rec[np.sqrt(np.sum(rec['chi2']**2, axis=1))<chi2_cut]
init=np.sum(np.sum(rec['h'][:,:10,:], axis=2), axis=1)
full=np.sum(np.sum(rec['h'], axis=2), axis=1)
rec=rec[init/full<0.5]

up=np.sum(rec['h'][:,:100,0], axis=1)+np.sum(rec['h'][:,:100,0], axis=1)
dn=np.sum(rec['h'][:,:100,-1], axis=1)+np.sum(rec['h'][:,:100,-2], axis=1)+np.sum(rec['h'][:,:100,-3], axis=1)
rec=rec[dn<3*up+18]

spectrum=np.histogram(np.sum(np.sum(rec['h'], axis=1), axis=1), bins=np.arange(1000)-0.5)[0]
rec=rec[np.sum(np.sum(rec['h'], axis=1), axis=1)>left]
rec=rec[np.sum(np.sum(rec['h'], axis=1), axis=1)<right]

H=rec['h'][:,:100,:]
S=np.sum(H, axis=1)
M=np.mean(S, axis=0)
print(M)

cov=np.zeros((len(rec), 15))
k=0
fig, ax=plt.subplots(3,5)
for i in range(len(pmts)-1):
    for j in range(i+1, len(pmts)):
        cov[:,k]=(S[:,i]-M[i])*(S[:,j]-M[j])/(M[i]*M[j])
        np.ravel(ax)[k].hist(cov[:,k], bins=20, range=[-0.1, 0.1])
        # np.ravel(ax)[k].set_yscale('log')
        k+=1
plt.show()
