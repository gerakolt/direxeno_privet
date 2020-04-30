import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt

chi2_cut=3e5
left=38*5
right=175*5
up=60
dn=40
pmt=1
path='/home/gerak/Desktop/DireXeno/WFsim/PMT{}/'.format(pmt)
Data=np.load(path+'spectra.npz')
spec=Data['spectrum']
chi2=Data['Chi2']
H=np.zeros((int(np.amax(spec)),1000))
rng=np.nonzero(np.logical_and(chi2<chi2_cut, np.logical_and(np.sum(spec, axis=1)<up, np.sum(spec, axis=1)>dn)))[0]

# for i, s in enumerate(spec):
#     spec[i]=np.roll(s, -np.amin(np.nonzero(s>0)[0]))

for i in range(1000):
    H[:,i]=np.histogram(spec[rng,i], bins=np.arange(np.amax(spec)+1)-0.5)[0]

h_spec, bins_spec=np.histogram(np.sum(spec[rng,left:right], axis=1), bins=np.arange(100)-0.5)
spec_x=0.5*(bins_spec[1:]+bins_spec[:-1])
n_events=np.sum(h_spec)
plt.figure()
plt.step(spec_x, h_spec, 'k')

plt.figure()
plt.plot(np.average(H, axis=0, weights=np.arange(len(H[:,0])))/n_events)
np.savez(path+'H', H=H, left=left, right=right, chi2_cut=chi2_cut, bins_spec=bins_spec, h_spec=h_spec, n_events=n_events)
plt.show()
