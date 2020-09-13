import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.stats import poisson, binom
from scipy.special import erf as erf
from fun import make_3D, Sim
from minimize import minimize, make_ps
from PMTgiom import make_mash

path='/home/gerak/Desktop/DireXeno/190803/Co57/EventRecon/'
data=np.load(path+'H.npz')
H=data['H']
cov1=data['cov']/np.sum(H[:,0,0])
Xcov1=data['Xcov']


# path='/home/gerak/Desktop/DireXeno/190803/Co57B/EventRecon/'
# data=np.load(path+'H.npz')
# H=data['H']
# cov2=data['cov']/np.sum(H[:,0,0])
# Xcov2=data['Xcov']
#
# path='/home/gerak/Desktop/DireXeno/190803/Cs137/EventRecon/'
# data=np.load(path+'H.npz')
# H=data['H']
# cov3=data['cov']/np.sum(H[:,0,0])
# Xcov3=data['Xcov']
#
# path='/home/gerak/Desktop/DireXeno/190803/Cs137B/EventRecon/'
# data=np.load(path+'H.npz')
# H=data['H']
# cov4=data['cov']/np.sum(H[:,0,0])
# Xcov4=data['Xcov']
#
#
# fig, bx=plt.subplots(3,5)
# k=0
# for k in range(15):
#     np.ravel(bx)[k].step(Xcov1, cov1[:,k], where='mid', label='1')
#     np.ravel(bx)[k].step(Xcov2, cov2[:,k], where='mid', label='2')
#     np.ravel(bx)[k].step(Xcov3, cov3[:,k], where='mid', label='3')
#     np.ravel(bx)[k].step(Xcov4, cov4[:,k], where='mid', label='4')
#     np.ravel(bx)[k].legend()
# plt.show()
