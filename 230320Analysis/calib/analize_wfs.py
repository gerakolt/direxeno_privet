import numpy as np
import matplotlib.pyplot as plt

pmt=19
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
data=np.load(path+'raw_wf.npz')
rec=data['rec']
left=data['left']
right=data['right']
init=data['init']
blw_cut=50
height_cut=100

x=np.arange(1000)
fig=plt.figure()
fig.suptitle('PMT{}'.format(pmt))
ax=fig.add_subplot(211)
ax.hist(rec['blw'], bins=100, range=[0,60], label='blw')
ax.axvline(blw_cut, ymin=0, ymax=1, color='k',linestyle='--')
ax.legend()
ax.set_yscale('log')

ax=fig.add_subplot(212)
ax.hist(rec['height'], bins=100, range=[0,550], label='raw height', histtype='step')
ax.axvline(height_cut, ymin=0, ymax=1, color='k', linestyle='--')
ax.set_yscale('log')
ax.legend()
np.savez(path+'cuts', blw_cut=blw_cut, height_cut=height_cut)
plt.show()
