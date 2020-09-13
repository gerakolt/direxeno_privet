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
rec=rec[np.sum(np.sum(rec['h'][:,:100,:], axis=2), axis=1)>0]
init=np.sum(np.sum(rec['h'][:,:10,:], axis=2), axis=1)
full=np.sum(np.sum(rec['h'], axis=2), axis=1)
rec=rec[init/full<0.5]

up=np.sum(rec['h'][:,:100,0], axis=1)+np.sum(rec['h'][:,:100,1], axis=1)
dn=np.sum(rec['h'][:,:100,-1], axis=1)+np.sum(rec['h'][:,:100,-2], axis=1)+np.sum(rec['h'][:,:100,-3], axis=1)
rec=rec[dn<3*up+18]
rec=rec[np.sum(np.sum(rec['h'][:,:100,:], axis=1), axis=1)>=left]
rec=rec[np.sum(np.sum(rec['h'][:,:100,:], axis=1), axis=1)<=right]
spectrum, binsSpec=np.histogram(np.sum(np.sum(rec['h'][:,:100,:], axis=1), axis=1), bins=np.arange(left, right+1)-0.5)
# spectrum, binsSpec=np.histogram(np.sum(np.sum(rec['h'][:,:100,:], axis=1), axis=1), bins=np.arange(left,right+2))





H=np.zeros((50, 200, len(pmts)))
G=np.zeros((300, 200))

for j in range(200):
    G[:,j]=np.histogram(np.sum(rec['h'][:,j,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
    # G[:,j]=np.histogram(np.sum(rec['h'][:,j,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]

spectra=np.zeros((100, len(pmts)))
for i, pmt in enumerate(pmts):
    h=rec['h'][:,:,i]
    spectra[:,i], binsspec=np.histogram(np.sum(h[:,:100], axis=1), bins=np.arange(101)-0.5)
    for j in range(200):
        H[:,j,i]=np.histogram(h[:,j], bins=np.arange(np.shape(H)[0]+1)-0.5)[0]


M=np.mean(np.sum(rec['h'][:,:100], axis=1), axis=0)
S=np.sum(rec['h'][:,:100], axis=1)

cov=np.zeros(15)

k=0
# fig, ax=plt.subplots(3,5)
for i in range(len(pmts)-1):
    for j in range(i+1, len(pmts)):
        cov[k]=np.mean((S[:,i]-M[i])*(S[:,j]-M[j]))
        # np.ravel(ax)[k].step(Xcov, cov[:,k], where='mid', label='full')
        # np.ravel(ax)[k].step(Xcov, cov10[:,k], where='mid', label='10')
        # np.ravel(ax)[k].legend()
        k+=1
# plt.show()
np.savez(path+'H', H=H, G=G, left=left, right=right, spectra=spectra, spectrum=spectrum, binsSpec=binsSpec, bins=binsspec, up_dn_cut='dn<3*up+18', cov=cov)
