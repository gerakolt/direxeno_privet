import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

pmt=1
rec=np.load('PMT{}/simWFs.npz'.format(pmt))['hits']
blw=rec['blw'][np.unique(rec['id'], return_index=True)[1]]
blw_cut=60
height_cut=150

init=[]
h=[]
for id in np.unique(rec['id']):
    sub=rec[rec['id']==id]
    if np.any((sub['blw']<blw_cut) & (sub['height']>height_cut)):
        init.append(np.amin(sub[(sub['blw']<blw_cut) & (sub['height']>height_cut)]['init']))
        h.append(sub[(sub['blw']<blw_cut) & (sub['height']>height_cut)]['height'][np.argmin(sub[(sub['blw']<blw_cut) & (sub['height']>height_cut)]['init'])])
fig=plt.figure()
ax=fig.add_subplot(221)
ax.hist(blw, bins=100, range=[0,100], label='blw')
ax.axvline(x=blw_cut, ymin=0, ymax=1, color='k')
ax.set_yscale('log')
ax.legend()

ax=fig.add_subplot(222)
ax.hist(rec['init'], bins=100, label='init', histtype='step')
ax.hist(init, bins=100, label='init over height cut', histtype='step')
ax.legend()

ax=fig.add_subplot(223)
ax.hist(rec['height'], bins=100, range=[0,400], label='height', histtype='step')
ax.hist(h, bins=100, range=[0,400], label='the height', histtype='step')
ax.axvline(x=height_cut, ymin=0, ymax=1, color='k')
ax.legend()

ax=fig.add_subplot(224)
ax.hist2d(rec['init'], rec['height'], bins=[100,100], range=[[0,1000], [0,600]], norm=LogNorm())
ax.axhline(y=height_cut, xmin=0, xmax=1, color='k')
ax.legend()

plt.show()
