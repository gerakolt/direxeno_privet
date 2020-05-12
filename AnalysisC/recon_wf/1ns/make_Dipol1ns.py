import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf


mid=np.array([[1/np.sqrt(2)*np.cos(120*np.pi/180), 1/np.sqrt(2)*np.sin(120*np.pi/180), 1/np.sqrt(2)],
                    [-1/np.sqrt(2), 0, 1/np.sqrt(2)],
                    [1/np.sqrt(2)*np.cos(240*np.pi/180), 1/np.sqrt(2)*np.sin(240*np.pi/180), 1/np.sqrt(2)],
                    [0,1,0],
                    [-1/np.sqrt(2), 1/np.sqrt(2), 0],
                    [-1,0,0],
                    [-1/np.sqrt(2), -1/np.sqrt(2),0],
                    [1/np.sqrt(2)*np.cos(120*np.pi/180), 1/np.sqrt(2)*np.sin(120*np.pi/180), -1/np.sqrt(2)],
                    [-1/np.sqrt(2), 0, -1/np.sqrt(2)],
                    [1/np.sqrt(2)*np.cos(240*np.pi/180), 1/np.sqrt(2)*np.sin(240*np.pi/180), -1/np.sqrt(2)],
                    [1/np.sqrt(2)*np.cos(300*np.pi/180), 1/np.sqrt(2)*np.sin(300*np.pi/180), 1/np.sqrt(2)],
                    [1/np.sqrt(2), 0,1/np.sqrt(2)],
                    [1/np.sqrt(2)*np.cos(60*np.pi/180), 1/np.sqrt(2)*np.sin(60*np.pi/180), 1/np.sqrt(2)],
                    [0, -1, 0],
                    [1/np.sqrt(2), -1/np.sqrt(2), 0],
                    [1, 0, 0],
                    [1/np.sqrt(2), 1/np.sqrt(2), 0],
                    [1/np.sqrt(2)*np.cos(300*np.pi/180), 1/np.sqrt(2)*np.sin(300*np.pi/180), -1/np.sqrt(2)],
                    [1/np.sqrt(2), 0, -1/np.sqrt(2)],
                    [1/np.sqrt(2)*np.cos(60*np.pi/180), 1/np.sqrt(2)*np.sin(60*np.pi/180), -1/np.sqrt(2)],
                    ])


pmts=[0,1,4,7,8,14]
path='/home/gerak/Desktop/DireXeno/190803/Co57/EventRecon/'
rec=np.load(path+'recon1ns81785.npz')['rec']
blw_cut=15
init_cut=20
chi2_cut=5000
left=170
right=250

rec=rec[np.all(rec['init_wf']>20, axis=1)]
rec=rec[np.sqrt(np.sum(rec['blw']**2, axis=1))<blw_cut]
rec=rec[np.sqrt(np.sum(rec['chi2']**2, axis=1))<chi2_cut]
init=np.sum(np.sum(rec['h'][:,:10,:], axis=2), axis=1)
full=np.sum(np.sum(rec['h'], axis=2), axis=1)
rec=rec[init/full<0.5]

f=np.zeros(len(rec['h']))
for i, h in enumerate(rec['h']):
    print(i)
    ns=np.sum(h, axis=0)
    cos=np.sum(mid[pmts]*mid[pmts[-1]], axis=1)
    f[i]=np.sum(ns*cos)

plt.figure()
plt.hist(f, bins=100)
plt.show()
