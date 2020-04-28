import numpy as np
import matplotlib.pyplot as plt

pmt=19
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
data=np.load(path+'raw_wf.npz')
rec=data['rec']
blw_cut=50
height_cut=35
left=100
right=360

fig, ((ax1,ax2),(ax3, ax4))=plt.subplots(2,2)
fig.suptitle('PMT{}'.format(pmt))
ax1.hist(rec['height'], bins=100, label='height', range=[0,250])
ax1.axvline(height_cut, ymin=0, ymax=1)
ax1.set_yscale('log')
ax1.legend()

ax2.hist(rec['maxi'], bins=100, label='maxi')
ax2.axvline(left, ymin=0, ymax=1)
ax2.axvline(right, ymin=0, ymax=1)
ax2.legend()

ax3.hist(rec['blw'], bins=100, label='blw', range=[0,60])
ax3.axvline(blw_cut, ymin=0, ymax=1)
ax3.legend()

ax4.hist(rec['bl'], bins=100, label='bl')
ax4.legend()

np.savez(path+'cuts', blw_cut=blw_cut, height_cut=height_cut, left=left, right=right)

plt.show()
