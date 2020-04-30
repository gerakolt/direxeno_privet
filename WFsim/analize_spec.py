import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import poisson, binom
from scipy.optimize import curve_fit



chi2=3e6
pmt=4

Data=np.load('/home/gerak/Desktop/DireXeno/190803/pulser_190803_46211/PMT{}/AllSPEs.npz'.format(pmt))
Spe=Data['Spe']


Data=np.load('PMT{}/spectra.npz'.format(pmt))
spec=Data['spectrum']
WF=Data['mean_WF']
Recon_WF=Data['Recon_wf']
Chi2=Data['Chi2']
ID=Data['ID']
N=len(spec[:,0])

t=np.arange(1000)/5
fig = plt.figure(figsize=(20,10))
fig.suptitle('SIM - PMT {} - {} events'.format(pmt, N), fontsize=25)
ax=fig.add_subplot(221)
ax.plot(t, np.mean(spec, axis=0), 'k.-', label='reconstructed')
ax.legend(fontsize=25)
ax.set_ylabel('PEs', fontsize=30)

def make_P(ns, Spe):
    P=np.zeros((ns[-1]+100, ns[-1]+100))
    P[0,0]=1
    for i in range(len(P[:,0])):
        r=np.linspace(i-0.5,i+0.5,1000)
        dr=r[1]-r[0]
        P[i,1]=dr*np.sum(np.exp(-0.5*(r-1)**2/Spe**2))/(np.sqrt(2*np.pi)*Spe)
    for j in range(2, len(P[0,:])):
        for i in range(len(P[:,0])):
            P[i,j]=np.sum(P[:i+1,1]*np.flip(P[:i+1,j-1]))
    return P[ns,:]

ax=fig.add_subplot(222)
h,bins, pach=ax.hist(np.sum(spec, axis=1), bins=np.arange(30,90)-0.5)
ns=(0.5*(bins[1:]+bins[:-1])).astype(int)
P=make_P(ns, Spe)
ax.plot(ns, P[:,50+1]/np.max(P[:,50+1])*np.max(h), 'ro', label='{}'.format(Spe))
ax.legend()

ax=fig.add_subplot(223)
ax.plot(np.arange(1000)/5, WF/N, 'k.', label='Summed signal')
ax.plot(np.arange(1000)/5, Recon_WF/N, 'r.', label='Summed reconstructed\n signal')
ax.plot(np.arange(1000)/5, WF/N-Recon_WF/N, 'g--', label='Summed reconstructed\n signal')

ax.legend(fontsize=15, loc='best')

ax=fig.add_subplot(224)
ax.hist(Chi2, bins=100, color='k', range=[0,0.4e7], label='Reconstruction Chi2')
# ax.axvline(x=chi2, ymin=0, ymax=1)
ax.legend(fontsize=25)
plt.show()
