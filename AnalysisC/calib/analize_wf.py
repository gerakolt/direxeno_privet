import numpy as np
import matplotlib.pyplot as plt

pmts=[0,5,12,13,14,15,16,18,19,2,3,4,10]
chns=[0,1,2,3,4,5,6,7,8,9,10,11,13]

pmt=10
path='/home/gerak/Desktop/DireXeno/050520/pulser/'
data=np.load(path+'raw_wf.npz')
rec=data['rec']


blw_cut=15
height_cut=70
left=100
right=360

fig, ((ax1,ax2),(ax3, ax4))=plt.subplots(2,2)
fig.suptitle('PMT{}'.format(pmt))
ax1.hist(rec['height'][:, pmt==np.array(pmts)], bins=100, label='height', range=[0,250])
ax1.axvline(height_cut, ymin=0, ymax=1)
ax1.set_yscale('log')
ax1.legend()

ax2.hist(rec['maxi'][:, pmt==np.array(pmts)], bins=100, label='maxi')
ax2.axvline(left, ymin=0, ymax=1)
ax2.axvline(right, ymin=0, ymax=1)
ax2.legend()

ax3.hist(rec['blw'][:, pmt==np.array(pmts)], bins=100, label='blw', range=[0,60])
ax3.axvline(blw_cut, ymin=0, ymax=1)
ax3.legend()

ax4.hist(rec['bl'][:, pmt==np.array(pmts)], bins=100, label='bl')
ax4.legend()

np.savez(path+'PMT{}/cuts'.format(pmt), blw_cut=blw_cut, height_cut=height_cut, left=left, right=right)

plt.show()
