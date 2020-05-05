import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import poisson, binom
from fun import Recon_WF

pmt=4
Data=np.load('/home/gerak/Desktop/DireXeno/pulser_190803_46211/PMT{}/AllSPEs.npz'.format(pmt))
spe=np.sum(Data['SPE'], axis=0)
spe=(spe-np.median(spe[:150]))/Data['factor']
spe[Data['zeros']]=0
Data=np.load('PMT{}/simWFs.npz'.format(pmt))
WF=Data['WF']
H=Data['H']

h_init=100
for j in range(10000):
    ev=np.random.randint(0,len(WF))
    recon_wf, chi2, recon_H=next(Recon_WF([WF[ev]], spe, 12, 6, h_init))


    fig=plt.figure()
    fig.suptitle('Event {}'.format(ev))
    ax=fig.add_subplot(211)
    ax.plot(WF[ev], 'k.')
    ax.plot(recon_wf, 'r--')


    ax=fig.add_subplot(212)
    ax.plot(H[ev], 'ko', label=np.sum(H[ev]))
    ax.plot(recon_H, 'ro', label=np.sum(recon_H))
    ax.legend()

    plt.show()
