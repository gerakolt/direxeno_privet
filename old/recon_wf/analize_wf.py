import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from fun import find_bl, import_spe, Recon_WF, find_hits, find_init10
from classes import DataSet, Event, WaveForm
from matplotlib.colors import LogNorm


PMT_num=20
time_samples=1024
pmts=np.array([0,1,2,3,4,5,6,7,8,9,10,11,17,14,15,16,18,19])
pmt=0
source='Co57'
type=''
path='/home/gerak/Desktop/DireXeno/'+source+'_190803'+type+'/PMT{}/'.format(pmt)
rec=np.load(path+'AllWFs.npz')['rec']
blw=rec['blw'][np.unique(rec['id'], return_index=True)[1]]

blw_cut=15
height_cut=30

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
ax.hist(init, bins=100, label='the init', histtype='step')
ax.legend()

ax=fig.add_subplot(223)
ax.hist(rec['height'], bins=100, range=[0,1500], label='height', histtype='step')
ax.hist(h, bins=100, range=[0,1500], label='the height', histtype='step')
ax.axvline(x=height_cut, ymin=0, ymax=1, color='k')
ax.legend()

ax=fig.add_subplot(224)
ax.hist2d(rec['init'], rec['height'], bins=[100,100], range=[[0,1000], [0,1500]], norm=LogNorm())
ax.axhline(y=height_cut, xmin=0, xmax=1, color='k')
ax.legend()

plt.show()
