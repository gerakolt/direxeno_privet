import numpy as np
import matplotlib.pyplot as plt
import time
import os

pmts=[0,7,8]
path='/home/gerak/Desktop/DireXeno/190803/Co57/EventRecon/'
rec=np.load(path+'recon.npz')['rec']
blw_cut=4.7
init_cut=20
chi2_cut=500

rec=rec[np.all(rec['init_wf']>init_cut, axis=1)]
rec=rec[np.all(rec['blw']<blw_cut, axis=1)]
rec=rec[np.all(rec['chi2']<chi2_cut, axis=1)]

H_specs=[]
for i, pmt in enumerate(pmts):
    H_specs.append(np.histogram(np.sum(rec['h'][:,:,i], axis=1), bins=np.arange(150)-0.5)[0])

ns=np.arange(25,75)
H_areas=[]
areas=[]
rng_areas=[]
N_height_cuts=[]
for i, pmt in enumerate(pmts):
    path='/home/gerak/Desktop/DireXeno/190803/pulser/NEWPMT{}/'.format(pmt)
    data=np.load(path+'areas.npz')
    H_areas.append(data['H_areas'])
    areas.append(data['areas'])
    rng_areas.append(data['rng_area'])
    N_height_cuts.append(data['N_height_cut'])

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6))=plt.subplots(3,2)

ax1.plot(H_specs[0], 'k.')
ax1.plot(ns, H_specs[0][ns], 'r.')
ax2.plot(areas[0], H_areas[0], 'k.')
ax2.set_yscale('log')

ax3.plot(H_specs[1], 'k.')
ax3.plot(ns, H_specs[1][ns], 'r.')
ax4.plot(areas[1], H_areas[1], 'k.')
ax4.set_yscale('log')

ax5.plot(H_specs[2], 'k.')
ax5.plot(ns, H_specs[2][ns], 'r.')
ax6.plot(areas[2], H_areas[2], 'k.')
ax6.set_yscale('log')

plt.show()
