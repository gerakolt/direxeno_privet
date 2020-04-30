import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import poisson, binom
from scipy.optimize import curve_fit


file='/home/gerak/Desktop/DireXeno/190803/Co57/Spectra/spectra.npz'
Data=np.load(file)
pmts=Data['pmts']
first_pmt=Data['first_pmt']
BLW=Data['BLW']
Chi2=Data['Chi2']
mean_WF=Data['mean_WF']
Recon_wf=Data['Recon_wf']
spectrum=Data['spectrum']

plt.figure()
plt.plot(pmts, first_pmt, 'o')
plt.title('first pmt')

chi2_cut=2e6
plt.figure()
plt.hist(Chi2, bins=100, range=[0,2e7])
plt.axvline(chi2_cut, ymin=0, ymax=1, color='k')
plt.title('Chi2')

blw_cut=20
for i, pmt in enumerate(pmts):
    plt.figure()
    plt.hist(BLW[:,i], bins=100)
    plt.axvline(blw_cut, ymin=0, ymax=1, color='k')
    plt.title('BLW {}'.format(pmts[i]))


for i, pmt in enumerate(pmts):
    fig=plt.figure()
    fig.suptitle(pmt)
    ax=fig.add_subplot(211)
    ax.plot(np.mean(spectrum[i], axis=1), 'k.-')

    ax=fig.add_subplot(212)
    ax.plot(mean_WF[i], 'k.-')
    ax.plot(Recon_wf[i], 'r.-')

plt.show()
