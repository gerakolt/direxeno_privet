import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import os
import sys

pmts=np.array([0,1,4,7,8,15])

# BGpath='/home/gerak/Desktop/DireXeno/190803/BG/EventRecon/'
path='/home/gerak/Desktop/DireXeno/190803/Co57/EventRecon/'
blw_cut=15
init_cut=20
chi2_cut=5000
left=635
right=1000
#T=1564823506349-1564820274767 #Cs
T=1564916315672-1564886605156 #Co
data=np.load(path+'recon1ns.npz')
BG=data['rec']

data=np.load(path+'recon1ns.npz')
rec=data['rec']

WFs=data['WFs']
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
fig.suptitle('Co57', fontsize=25)
x=np.arange(1000)/5
for i in range(len(pmts)):
    np.ravel(ax)[i].hist(rec['init_wf'][:,i], bins=100, range=[0,400], label='PMT{} init_wf'.format(pmts[i]))
    np.ravel(ax)[i].legend(fontsize=15)
rec=rec[np.all(rec['init_wf']>init_cut, axis=1)]
BG=BG[np.all(BG['init_wf']>init_cut, axis=1)]


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
BG=BG[np.sqrt(np.sum(BG['blw']**2, axis=1))<blw_cut]

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
rec=rec[np.sum(np.sum(rec['h'][:,:100,:], axis=2), axis=1)>0]
BG=BG[np.sqrt(np.sum(BG['chi2']**2, axis=1))<chi2_cut]
BG=BG[np.sum(np.sum(BG['h'][:,:100,:], axis=2), axis=1)>0]

init=np.sum(np.sum(rec['h'][:,:10,:], axis=2), axis=1)
full=np.sum(np.sum(rec['h'][:,:100,:], axis=2), axis=1)
BGinit=np.sum(np.sum(BG['h'][:,:10,:], axis=2), axis=1)
BGfull=np.sum(np.sum(BG['h'][:,:100,:], axis=2), axis=1)

plt.figure(figsize=(20,10))
flash_cut=0.5
plt.hist(init/full, bins=100, range=[0,1], label=r'$^{57}$'+'Co data')
plt.axvline(flash_cut, 0, 0.25, linewidth=5, label=r'$\omega<0.5$'+'cut', color='k')
plt.legend(fontsize=35)
plt.xlabel(r'$\omega$', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20)
print('flash rate:', len(rec[init/full>flash_cut]['id'])/(T*1e-3))
rec=rec[init/full<flash_cut]
BG=BG[BGinit/BGfull<flash_cut]

plt.figure(figsize=(20,10))
up=np.sum(rec['h'][:,:100,0], axis=1)+np.sum(rec['h'][:,:100,1], axis=1)
dn=np.sum(rec['h'][:,:100,-1], axis=1)+np.sum(rec['h'][:,:100,-2], axis=1)+np.sum(rec['h'][:,:100,-3], axis=1)
plt.plot(np.arange(450), np.arange(450)*3+18, 'k--', linewidth=5, label=r'$N_{bot}<3N_{top}+18$'+'\ncut')
plt.hist2d(up, dn, bins=[100, 100], range=[[0,350], [0,700]], norm=mcolors.PowerNorm(0.3))
plt.xlabel('Sum of PEs in the top floor PMTs', fontsize=25)
plt.ylabel('Sum of PEs in the bottom floor PMTs', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20)
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=25)
rec0=rec
rec=rec[dn<3*up+18]
plt.legend(fontsize=35, loc='upper right')

TB=1564926608911-1564916365644
TA=1564916315672-1564886605156
TBG=1564874707904-1564826183355
TCs=1564823506349-1564820274767

hist, bins=np.histogram(np.sum(np.sum(BG['h'][:,:100,:], axis=2), axis=1),  bins=np.arange(250)*5)
plt.figure()
plt.hist(np.sum(np.sum(rec['h'][:,:100,:], axis=2), axis=1),  bins=np.arange(250)*5, histtype='step', linewidth=5, label='All events')
plt.bar(0.5*(bins[1:]+bins[:-1]) ,TB/TBG*hist, label='BG', width=bins[1:]-bins[:-1], color='orange', alpha=0.5)
plt.axvline(left, 0 ,1, color='k')
plt.axvline(right, 0 ,1, color='k')
plt.legend(fontsize=15)

rec=rec[np.sum(np.sum(rec['h'][:,:100,:], axis=1), axis=1)>=left]
rec=rec[np.sum(np.sum(rec['h'][:,:100,:], axis=1), axis=1)<=right]
rec0=rec0[np.sum(np.sum(rec0['h'][:,:100,:], axis=1), axis=1)>=left]
rec0=rec0[np.sum(np.sum(rec0['h'][:,:100,:], axis=1), axis=1)<=right]
fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].hist(np.sum(rec['h'][:,:100,i], axis=1),  bins=np.arange(100), histtype='step', label='After\n up-dn cut\n PMT{}'.format(i), linewidth=3)
    np.ravel(ax)[i].hist(np.sum(rec0['h'][:,:100,i], axis=1),  bins=np.arange(100), histtype='step', label='Before\n up-dn cut', linewidth=3)
    np.ravel(ax)[i].legend(fontsize=15)

fig, ax=plt.subplots(3,2)
fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('Co57', fontsize=25)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(np.mean(rec['h'][:,:,i], axis=0), 'k-.', label='PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].legend()


plt.show()
