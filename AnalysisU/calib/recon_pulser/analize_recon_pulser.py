import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import os
import sys

pmts=np.array([0,1,4,7,8,15])
blw_cut=20
chi2_cut=2000
left=240
right=380

path='/home/gerak/Desktop/DireXeno/190803/pulser/EventRecon/'
data=np.load(path+'recon1ns.npz')
rec=data['rec']
WFs=data['WFs']
Abins=data['Abins']
recon_WFs=data['recon_WFs']


fig, ax=plt.subplots(2,3)
fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('Co57', fontsize=25)
x=np.arange(1000)/5
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(x, WFs[i], 'r1', label='WF: PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].plot(x, recon_WFs[i], 'b-.', label='Recon')
    np.ravel(ax)[i].legend(fontsize=12)


fig, ax=plt.subplots(2,3)
fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('Co57', fontsize=25)
x=np.arange(1000)/5
for i in range(len(pmts)):
    np.ravel(ax)[i].hist(rec['blw'][:,i], bins=100, range=[0,30], label='PMT{} BLW'.format(pmts[i]))
    np.ravel(ax)[i].legend(fontsize=15)

plt.figure()
plt.hist(np.sqrt(np.sum(rec['blw']**2, axis=1)), bins=100, label='BLW', range=[0,30])
plt.axvline(blw_cut, ymin=0, ymax=1, color='k')
plt.legend(fontsize=15)
rec=rec[np.sqrt(np.sum(rec['blw']**2, axis=1))<blw_cut]

fig, ax=plt.subplots(3,2)
fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('Co57', fontsize=25)
for i in range(len(pmts)):
    np.ravel(ax)[i].hist(rec['chi2'][:,i], bins=100, label='PMT{} chi2'.format(pmts[i]))
    np.ravel(ax)[i].set_yscale('log')
    np.ravel(ax)[i].legend(fontsize=15)


plt.figure()
plt.hist(np.sqrt(np.sum(rec['chi2']**2, axis=1)), bins=100, label='chi2')
plt.axvline(chi2_cut, ymin=0, ymax=1, color='k')
plt.legend(fontsize=15)
plt.yscale('log')

rec=rec[np.sqrt(np.sum(rec['chi2']**2, axis=1))<chi2_cut]

fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].hist(np.sum(rec['h'][:,:,i], axis=1),  bins=np.arange(20), histtype='step', label='PMT{}'.format(i), linewidth=3)
    np.ravel(ax)[i].legend(fontsize=15)
    np.ravel(ax)[i].set_yscale('log')

fig, ax=plt.subplots(3,2)
# fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('Co57', fontsize=25)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(np.mean(rec['h'][:,:,i], axis=0), 'k-.', label='PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].axvline(left, 0 , 0.25, linewidth=5, color='k')
    np.ravel(ax)[i].axvline(right, 0 , 0.25, linewidth=5, color='k')
    np.ravel(ax)[i].legend()

ind=np.argmin(np.abs(0.5*(Abins[1:]+Abins[:-1])-0.74))
bool=rec['Areas'][:,3,ind]>0
# print(rec[bool]['id'])
fig, ax=plt.subplots(3,2)
fig.suptitle('Areas', fontsize=25)
for i in range(len(pmts)):
    np.ravel(ax)[i].bar(0.5*(Abins[1:]+Abins[:-1]), np.sum(rec['Areas'][:,i,:], axis=0), width=(Abins[1:]-Abins[:-1]), label='PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].legend()

plt.show()
