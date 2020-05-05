import numpy as np
import matplotlib.pyplot as plt
from itertools import product as prdcut
import time
import os
import sys

path='/home/gerak/Desktop/DireXeno/190803/pulser/DelayRecon/'

pmts=[0,1,2,4,7,8,14]
data=np.load(path+'recon.npz')
rec=data['rec']
# pmts=data['pmts']

dT=[]
dT10=[]

names=[]
for n in range(len(pmts)-1):
    hn=rec['h'][:,:,n]
    init10n=rec['init10'][:,n]
    for m in range(n+1, len(pmts)):
        dt=[]
        dt10=[]
        hm=rec['h'][:,:,m]
        init10m=rec['init10'][:,m]
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
