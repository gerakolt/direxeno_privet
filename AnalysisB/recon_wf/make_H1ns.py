import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf




pmts=[0,1,4,7,8,14]
path='/home/gerak/Desktop/DireXeno/190803/Co57/EventRecon/'
rec=np.load(path+'recon81785.npz')['rec']
blw_cut=15
init_cut=20
chi2_cut=5000
left=105
right=288

rec=rec[np.all(rec['init_wf']>20, axis=1)]
rec=rec[np.sqrt(np.sum(rec['blw']**2, axis=1))<blw_cut]
rec=rec[np.sqrt(np.sum(rec['chi2']**2, axis=1))<chi2_cut]
rec=rec[np.sum(np.sum(rec['h'], axis=1), axis=1)>left]
rec=rec[np.sum(np.sum(rec['h'], axis=1), axis=1)<right]


H=np.zeros((15, 200, len(pmts)))
G=np.zeros((15, 200))



for j in range(200):
    G[:,j]=np.histogram(np.sum(np.sum(rec['h'][:,j*5:5*(j+1),:], axis=2), axis=1), bins=np.arange(np.shape(H)[0]+1)-0.5)[0]

spectra=[]
for i, pmt in enumerate(pmts):
    rc=rec['h'][:,:,i]
    spectra.append(np.histogram(np.sum(rc, axis=1), bins=np.arange(200)-0.5)[0])
    for j in range(200):
        H[:,j,i]=np.histogram(np.sum(rc[:,5*j:5*(j+1)],axis=1), bins=np.arange(np.shape(H)[0]+1)-0.5)[0]
        print(i,j)
np.savez(path+'H1ns', H=H, G=G, left=left, right=right, spectra=spectra)
