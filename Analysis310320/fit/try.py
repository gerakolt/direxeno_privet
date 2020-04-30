import numpy as np
import matplotlib.pyplot as plt
import time
import os
from fun import make_P, model_spec, model_area, rec_to_p, p_to_rec, model_h, Model2
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf
import warnings

path='/home/gerak/Desktop/DireXeno/190803/Co57/EventRecon/'
rec=np.load(path+'recon.npz')['rec']
H=np.load(path+'H.npz')['H']
ns=np.load(path+'H.npz')['ns']
blw_cut=4.7
init_cut=20
chi2_cut=500

rec=rec[np.all(rec['init_wf']>init_cut, axis=1)]
rec=rec[np.all(rec['blw']<blw_cut, axis=1)]
rec=rec[np.all(rec['chi2']<chi2_cut, axis=1)]
pmts=[7,8]

P=[]
P.append(make_P(0.5, 0.1, 0.01, 0.01))

temporal=Model2([35], 0.1, 5, 45, [0.8], P)

plt.figure()
plt.plot(temporal[:,0], 'k.')

# plt.figure()
# plt.plot(np.mean(H[:,:,0].T*np.arange(np.shape(H)[0]), axis=1), 'r.-')
# plt.plot(np.sum(H[:,0,0])*np.mean(temporal[:,:,0].T*np.arange(np.shape(H)[0]), axis=1), 'k.-')
#
# fig, (ax1, ax2)=plt.subplots(2,1)
# ax1.plot(np.sum(H[:,0,0])*temporal[:,0,0], 'ko')
# ax1.plot(H[:,0,0], 'ro')
# ax1.axhline(np.mean(H[:,0,0]*np.arange(np.shape(H)[0])), color='r')
# ax1.axhline(np.sum(H[:,0,0])*np.mean(temporal[:,0,0]*np.arange(np.shape(H)[0])), color='k')
#
# ax2.plot(np.sum(H[:,1,0])*temporal[:,0,0], 'ko')
# ax2.plot(H[:,1,0], 'ro')
# ax2.axhline(np.mean(H[:,1,0]*np.arange(np.shape(H)[0])), color='r')
# ax2.axhline(np.sum(H[:,0,0])*np.mean(temporal[:,1,0]*np.arange(np.shape(H)[0])), color='k')

plt.show()
