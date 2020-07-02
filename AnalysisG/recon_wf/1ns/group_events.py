import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

pmts=np.array([0,1,4,7,8,14])

path='/home/gerak/Desktop/DireXeno/190803/Co57/EventRecon/'
blw_cut=25
init_cut=20
chi2_cut=10000
left=170
right=250
data=np.load(path+'recon1ns81785.npz')
rec=data['rec']

rec=rec[np.all(rec['init_wf']>20, axis=1)]
rec=rec[np.sqrt(np.sum(rec['blw']**2, axis=1))<blw_cut]
rec=rec[np.sqrt(np.sum(rec['chi2']**2, axis=1))<chi2_cut]

init=np.sum(np.sum(rec['h'][:,:10,:], axis=2), axis=1)
full=np.sum(np.sum(rec['h'], axis=2), axis=1)
rec=rec[init/full<0.5]

h=np.zeros((np.shape(rec['h'])[0], 13, np.shape(rec['h'])[-1]))

h[:,:5,:]=rec['h'][:,:5,:]
dt=np.ones(5)
t=np.arange(5)+0.5*dt[-1]
for i in range(5):
    h[:,5+i,:]=np.sum(rec['h'][:,5+5*i:5+5*(i+1),:], axis=1)
    dt=np.append(dt, 5)
    if i==0:
        t=np.append(t, t[-1]+3)
    else:
        t=np.append(t, t[-1]+5)
for i in range(3):
    h[:,10+i,:]=np.sum(rec['h'][:,30+10*i:30+10*(i+1),:], axis=1)
    dt=np.append(dt, 10)
    if i==0:
        t=np.append(t, t[-1]+7.5)
    else:
        t=np.append(t, t[-1]+10)

fig, ax=plt.subplots(2,3)
fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('Co57', fontsize=25)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(np.mean(rec['h'][:,:,i], axis=0), 'k.', label='PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].plot(t, np.mean(h[:,:,i], axis=0)/dt, 'r+', label='PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].legend(fontsize=15)

plt.show()
np.savez(path+'h', h=h, t=t, dt=dt)
