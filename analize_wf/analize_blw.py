import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt

pmts=np.array([0,1,2,3,4,5,6,7,8,9,10,11,17,14,15,16,18,19])
pmt=0
source='Co57'
type=''
path='/home/gerak/Desktop/DireXeno/190803/'+source+type+'/'
BLW=np.load(path+'BLW_table.npz')['BLW']
blw=BLW[:, pmt==pmts]

blw_cut=20

plt.axvline(x=blw_cut, ymin=0, ymax=1, color='k', label=len(blw[blw<blw_cut])/len(blw))
plt.hist(blw, bins=np.arange(200)-0.5)
plt.yscale('log')
plt.title('PMT{}'.format(pmt))
plt.legend()
plt.show()
