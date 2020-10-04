import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import os
import sys

pmts=np.array([0,1,4,7,8,14])
blw_cut=20
chi2_cut=2000
left=240
right=380

path='/home/gerak/Desktop/DireXeno/190803/pulser/EventRecon/'
data=np.load(path+'recon1ns.npz')
rec=data['rec']
rec=rec[np.sqrt(np.sum(rec['blw']**2, axis=1))<blw_cut]
rec=rec[np.sqrt(np.sum(rec['chi2']**2, axis=1))<chi2_cut]
h=rec['h'][:,left:right,:]
dc=rec['h'][:,:,:]

names=[]
H=np.zeros((15,21))
DC=np.zeros((15,21))
q=0
for i in range(len(pmts)-1):
    for j in range(i+1, len(pmts)):
        names.append('Delay_{}_{}'.format(pmts[i], pmts[j]))
        d=[]
        ddc=[]
        print(i, j)
        for k in range(len(h[:,0,0])):
            h1=h[k,:,i]
            h2=h[k,:,j]
            if np.any(h1>0) and np.any(h2>0):
                for l in np.nonzero(h1>0)[0]:
                    for r in np.nonzero(h2>0)[0]:
                        for p in range(h1[l]):
                            for u in range(h2[r]):
                                d.append((r-l)/5)

        for k in range(len(dc[:,0,0])):
            h1=dc[k,:,i]
            h2=dc[k,:,j]
            if np.any(h1>0) and np.any(h2>0):
                for l in np.nonzero(h1>0)[0]:
                    for r in np.nonzero(h2>0)[0]:
                        if l<left and r>=right:
                            for p in range(h1[l]):
                                for u in range(h2[r]):
                                    ddc.append((r-l)/5)

        z, bins=np.histogram(d, bins=np.arange(-10, 12)-0.5)
        w, bins=np.histogram(ddc, bins=np.arange(-10, 12)-0.5)
        H[q]=z
        DC[q]=w
        q+=1

fig, ax=plt.subplots(3,5)
x=0.5*(bins[1:]+bins[:-1])
for i in range(15):
    np.ravel(ax)[i].step(x, H[i], where='mid', label='Ph')
    np.ravel(ax)[i].step(x, DC[i], where='mid', label='DC')
    np.ravel(ax)[i].legend()
np.savez(path+'delays', H=H, names=names, BinsDelay=bins, DC=DC)
plt.show()
