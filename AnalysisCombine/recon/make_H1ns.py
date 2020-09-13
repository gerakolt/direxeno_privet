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
full=np.sum(np.sum(rec['h'][:,:100,:], axis=2), axis=1)
rec=rec[init/full<0.5]

up=np.sum(rec['h'][:,:100,0], axis=1)+np.sum(rec['h'][:,:100,0], axis=1)
dn=np.sum(rec['h'][:,:100,-1], axis=1)+np.sum(rec['h'][:,:100,-2], axis=1)+np.sum(rec['h'][:,:100,-3], axis=1)
rec=rec[dn<3*up+18]

alm=np.recarray(len(rec), dtype=[
    ('a0', 'f8',1),
    ('a1', 'f8',3)
    ])
#
# for i in range(len(rec)):
#     alm['a0'][i], alm['a1'][i]=Ylm(np.sum(rec['h'][i, :100,:], axis=0))
#
# rec=rec[alm['a1'][:,1]/alm['a0']<0.2068]

spectrum=np.histogram(np.sum(np.sum(rec['h'], axis=1), axis=1), bins=np.arange(1000)-0.5)[0]
rec=rec[np.sum(np.sum(rec['h'], axis=1), axis=1)>left]
rec=rec[np.sum(np.sum(rec['h'], axis=1), axis=1)<right]

H=np.zeros((50, 200, len(pmts)))
G=np.zeros((300, 200))

for j in range(200):
    G[:,j]=np.histogram(np.sum(rec['h'][:,j,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]

spectra=np.zeros((350, len(pmts)))
for i, pmt in enumerate(pmts):
    h=rec['h'][:,:,i]
    spectra[:,i]=np.histogram(np.sum(h[:,:100], axis=1), bins=np.arange(351)-0.5)[0]
    for j in range(200):
        H[:,j,i]=np.histogram(h[:,j], bins=np.arange(np.shape(H)[0]+1)-0.5)[0]

np.savez(path+'H', H=H, G=G, left=left, right=right, spectra=spectra, spectrum=spectrum, up_down_cut='dn<3*up+18')
