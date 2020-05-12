import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

pmts=np.array([0,5,12,13,14,15,16,18,19,2,3,4,10])

path='/home/gerak/Desktop/DireXeno/050520/DC/EventRecon/'
blw_cut=25
init_cut=20
chi2_cut=10000
left=170
right=250

data=np.load(path+'recon1ns376.npz')
rec=data['rec']
WFs=data['WFs']
recon_WFs=data['recon_WFs']
A=rec[np.all(rec['sat']==0, axis=1)]

print(rec['id'][np.amax(np.sum(rec['h'][:,:10,:],axis=1),axis=1)<10])

plt.figure()
plt.hist(np.amax(np.sum(rec['h'][:,:10,:],axis=1),axis=1), bins=np.arange(100)-0.5, label='PEs that coused trigger', histtype='step')
plt.hist(np.amax(np.sum(A['h'][:,:10,:],axis=1),axis=1), bins=np.arange(100)-0.5, label='PEs that coused trigger without sat', histtype='step')

plt.yscale('log')
plt.legend()

plt.show()
