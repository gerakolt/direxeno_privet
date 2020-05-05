import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

blw_cut=20
height_cut=60

pmt=0
source='Co57'
type=''
path='/home/gerak/Desktop/DireXeno/190803/'+source+type+'/'
Rec=np.load(path+'Allhits.npz')['rec']
rec0=Rec[Rec['pmt']==pmt]
# rec0=Rec
rec=rec0[rec0['blw']<blw_cut]

fig=plt.figure()
fig.suptitle('PMT{}, {} events'.format(pmt, len(rec)))
ax=fig.add_subplot(311)
ax.hist(rec0['blw'], bins=100, range=[0,200], label='blw')
ax.axvline(x=blw_cut, ymin=0, ymax=1, color='k')
ax.set_yscale('log')
ax.legend()

ax=fig.add_subplot(312)
ax.hist(rec['init_first_hit'], bins=100, label='init first hit', histtype='step')
ax.hist(rec[rec['height_first_hit']>height_cut]['init_first_hit'], bins=100, label='init first hit h cut', histtype='step')
ax.legend()

ax=fig.add_subplot(313)
ax.hist(rec['height_first_hit'], bins=100, label='height', histtype='step')
ax.axvline(x=height_cut, ymin=0, ymax=1, color='k')
ax.legend()

# ax=fig.add_subplot(224)
# ax.hist2d(rec['init'], rec['height'], bins=[100,100], range=[[0,1000], [0,400]], norm=LogNorm())
# ax.axhline(y=height_cut, xmin=0, xmax=1, color='k')
# ax.legend()

plt.show()
