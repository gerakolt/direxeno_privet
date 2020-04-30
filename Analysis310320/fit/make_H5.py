import numpy as np
import matplotlib.pyplot as plt
import time
import os
from fun import make_P, model_spec, model_area, rec_to_p, p_to_rec, model_h
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf




pmts=[7,8]
ns=np.arange(30,40)
path='/home/gerak/Desktop/DireXeno/190803/BG/EventRecon/'
rec=np.load(path+'recon.npz')['rec']
blw_cut=6.5
init_cut=20
chi2_cut=1050
left=6
right=30

rec=rec[np.all(rec['init_wf']>init_cut, axis=1)]
rec=rec[np.sqrt(np.sum(rec['blw']**2, axis=1))<blw_cut]
rec=rec[np.sqrt(np.sum(rec['chi2']**2, axis=1))<chi2_cut]
rec=rec[np.sum(np.sum(rec['h'], axis=1), axis=1)>left]
rec=rec[np.sum(np.sum(rec['h'], axis=1), axis=1)<right]


H=np.zeros((15, 1000, 2))
G=np.zeros((15, 1000))



for j in range(1000):
    G[:,j]=np.histogram(np.sum(rec['h'][:,j,:], axis=1), bins=np.arange(np.shape(H)[0]+1)-0.5)[0]

spectra=[]
for i, pmt in enumerate(pmts):
    rc=rec['h'][:,:,i]
    spectra.append(np.histogram(np.sum(rc, axis=1), bins=np.arange(100)-0.5)[0])
    for j in range(1000):
        H[:,j,i]=np.histogram(rc[:,j], bins=np.arange(np.shape(H)[0]+1)-0.5)[0]
        print(i,j)
np.savez(path+'H0', H=H, G=G, left=left, right=right, spectra=spectra)
