import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

pmt=8
path='/home/gerak/Desktop/DireXeno/190803/Co57/PMT{}/'.format(pmt)
data=np.load(path+'raw_wf.npz')
WF=data['WF']
BL=data['BL']
init=data['init']
blw_cut=data['blw_cut']
rec=data['rec']
blw=rec['blw']
hit_init=rec['hit_init']
hit_area=rec['hit_area']
init_area_cut=75000

fig=plt.figure()
ax=fig.add_subplot(221)
ax.hist(blw, bins=100, label='blw', range=[0,400])
ax.axvline(blw_cut, ymin=0, ymax=1, color='k')
ax.set_yscale('log')
ax.legend()

ax=fig.add_subplot(222)
x=np.arange(1000)
ax.plot(x, WF, 'r.-', label='WF')
ax.plot(x, BL, 'k.-', label='BL')
ax.fill_between(x[:init], y1=np.amin(WF), y2=0, color='y', alpha=0.5)
ax.legend()

ax=fig.add_subplot(223)
ax.hist(hit_init, bins=100, label='hit_init')
ax.legend()
ax.set_yscale('log')


ax=fig.add_subplot(224)
ax.hist(hit_area, bins=100, label='hit_area')
ax.axvline(init_area_cut, ymin=0, ymax=1, color='k')
ax.legend()
ax.set_yscale('log')


plt.show()
