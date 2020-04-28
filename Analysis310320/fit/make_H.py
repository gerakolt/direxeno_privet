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
path='/home/gerak/Desktop/DireXeno/190803/Co57/EventRecon/'
rec=np.load(path+'recon.npz')['rec']
blw_cut=4.7
init_cut=20
chi2_cut=500

rec=rec[np.all(rec['init_wf']>init_cut, axis=1)]
rec=rec[np.all(rec['blw']<blw_cut, axis=1)]
rec=rec[np.all(rec['chi2']<chi2_cut, axis=1)]

ns=np.arange(30,40)
H=np.zeros((15, 1000, 2))

for i, pmt in enumerate(pmts):
    rc=rec['h'][:,:,i]
    rc=rc[np.logical_and(np.sum(rc,axis=1)<=ns[-1], np.sum(rc,axis=1)>=ns[0])]
    for j in range(1000):
        H[:,j,i]=np.histogram(rc[:,j], bins=np.arange(np.shape(H)[0]+1)-0.5)[0]
        print(i,j)
np.savez(path+'H', H=H, ns=ns)
