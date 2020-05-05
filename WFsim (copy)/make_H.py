import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt

chi2_cut=1e6
left=38*5
right=175*5
up=60
dn=40
pmt=4
path='/home/gerak/Desktop/DireXeno/WFsim/PMT{}/'.format(pmt)
Data=np.load(path+'spectra.npz'.format(pmt))
spec=Data['spectrum']
chi2=Data['Chi2']
H=np.zeros((int(np.amax(spec)),1000))
rng=np.nonzero(np.logical_and(chi2<chi2_cut, np.logical_and(np.sum(spec, axis=1)<up, np.sum(spec, axis=1)<dn)))[0]
for i in range(1000):
    H[:,i]=np.histogram(spec[rng,i], bins=np.arange(np.amax(spec)+1)-0.5)[0]

spec_y, bins=np.histogram(np.sum(spec[rng,left:right], axis=1), bins=np.arange(100)-0.5)
spec_x=0.5*(bins[1:]+bins[:-1])
plt.plot(spec_x, spec_y, 'ro')
plt.show()
np.savez(path+'H', H=H, spec_x=spec_x, spec_y=spec_y)
