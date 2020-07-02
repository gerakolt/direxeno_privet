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

pmt=2
data=np.load(path+'recon1ns376.npz')
rec=data['rec']
WFs=data['WFs']
recon_WFs=data['recon_WFs']


for pmt in pmts:
    sat=rec[rec['sat'][:,np.nonzero(pmts==pmt)[0][0]]==1]
    plt.figure()
    plt.title(pmt)
    plt.hist(np.sum(rec['h'][:,:10,np.nonzero(pmts==pmt)[0]], axis=1), bins=np.arange(100)-0.5, label='PEs in 1 ns on each PMT', histtype='step')
    plt.hist(np.sum(sat['h'][:,:10,np.nonzero(pmts==pmt)[0]], axis=1), bins=np.arange(100)-0.5, label='sat', histtype='step')

    plt.yscale('log')
    plt.legend()

    plt.show()
