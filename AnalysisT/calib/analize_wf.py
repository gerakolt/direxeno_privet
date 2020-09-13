import numpy as np
import matplotlib.pyplot as plt

pmts=[0,1,4,7,8,14]
chns=[2,3,6,9,10,15]

pmt=14
path='/home/gerak/Desktop/DireXeno/190803/pulser/'
data=np.load(path+'raw_wf.npz')
rec=data['rec']


blw_cut=15
height_cut=34
left=185
right=240

fig, ((ax1,ax2),(ax3, ax4))=plt.subplots(2,2)
fig.suptitle('PMT{}'.format(pmt))
ax1.hist(rec['height'][:, pmt==np.array(pmts)], bins=100, label='height', range=[0,250])
ax1.axvline(height_cut, ymin=0, ymax=1, color='k')
ax1.set_yscale('log')
ax1.legend()

ax2.hist(rec['maxi'][:, pmt==np.array(pmts)], bins=100, label='maxi')
ax2.axvline(left, ymin=0, ymax=1, color='k')
ax2.axvline(right, ymin=0, ymax=1, color='k')
ax2.legend()

ax3.hist(rec['blw'][:, pmt==np.array(pmts)], bins=100, label='blw', range=[0,60])
ax3.axvline(blw_cut, ymin=0, ymax=1, color='k')
ax3.legend()

ax4.hist(rec['bl'][:, pmt==np.array(pmts)], bins=100, label='bl')
ax4.legend()

np.savez(path+'PMT{}/cuts'.format(pmt), blw_cut=blw_cut, height_cut=height_cut, left=left, right=right)

plt.show()
