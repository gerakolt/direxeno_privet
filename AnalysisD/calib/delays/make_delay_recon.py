import numpy as np
import matplotlib.pyplot as plt
from itertools import product as prdcut
import time
import os
import sys

path='/home/gerak/Desktop/DireXeno/050520/pulser/DelayRecon/'

pmts=np.array([0,5,12,13,14,15,16,18,19,2,3,4,10])
data=np.load(path+'recon49995.npz')
rec=data['rec']
# pmts=data['pmts']

dT=[]
dT10=[]

names=[]
for n in range(len(pmts)-1):
    hn=rec['h'][:,:,n]
    init10n=rec['init10'][:,n]
    maxin=np.argmax(np.mean(hn, axis=0))
    hn=hn[:,maxin-50:maxin+50]
    for m in range(n+1, len(pmts)):
        dt=[]
        dt10=[]
        hm=rec['h'][:,:,m]
        init10m=rec['init10'][:,m]
        maxim=np.argmax(np.mean(hm, axis=0))
        hm=hm[:,maxim-50:maxim+50]
        for i in np.nonzero(np.logical_and(np.any(hn>0, axis=1), np.any(hm>0, axis=1)))[0]:
            dt10.append((init10m[i]-init10n[i])/5)
            print(n,m,i)
            for j in np.nonzero(hn[i,:]>0)[0]:
                for k in np.nonzero(hm[i,:]>0)[0]:
                    for jj in range(hn[i,j]):
                        for kk in range(hm[i,k]):
                            dt.append((k-j)/5)

        dT.append(dt)
        dT10.append(dt10)
        names.append('{}-{}'.format(pmts[n], pmts[m]))
np.savez(path+'delays_lists', dT=dT, dT10=dT10, names=names)
