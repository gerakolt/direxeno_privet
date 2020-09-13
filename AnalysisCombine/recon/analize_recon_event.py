import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import os
import sys
# from PMTgiom import make_pmts

# pmts=np.array([0,1,4,7,8,15])
# pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn, r=make_pmts(pmts)

pmts=np.array([0,1,4,7,8,14])
blw_cut=15
chi2_cut=5000
# left=720
# right=1100
left=170
right=230


path='/home/gerak/Desktop/DireXeno/190803/Co57B/EventRecon/'
data=np.load(path+'recon1ns.npz')
rec=data['rec']
WFs=data['WFs']
recon_WFs=data['recon_WFs']

path='/home/gerak/Desktop/DireXeno/190803/Co57B/EventRecon/'
data=np.load(path+'recon1ns.npz')
recB=data['rec']
WFs=data['WFs']
recon_WFs=data['recon_WFs']
#
# fig, ax=plt.subplots(4,4)
# fig.subplots_adjust(wspace=0, hspace=0)
# fig.suptitle(name, fontsize=25)
# x=np.arange(1000)/5
# for i in range(len(pmts)):
#     np.ravel(ax)[i].plot(x, WFs[i], 'r1', label='PMT{}'.format(pmts[i]))
#     np.ravel(ax)[i].plot(x, recon_WFs[i], 'b-.')
#     np.ravel(ax)[i].legend(fontsize=12)
# plt.show()

# rec0=recB[np.logical_or(np.sum(recB['h'][:,:100,0], axis=1)<100, np.sum(recB['h'][:,:100,1], axis=1)<60)]
up=np.sum(rec['h'][:,:100,0], axis=1)+np.sum(rec['h'][:,:100,0], axis=1)
dn=np.sum(rec['h'][:,:100,-1], axis=1)+np.sum(rec['h'][:,:100,-2], axis=1)+np.sum(rec['h'][:,:100,-3], axis=1)
rec0=rec[dn>3*up+18]


fig, ax=plt.subplots(3,2)
# fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('Co57', fontsize=25)
x=np.arange(1000)/5
for i in range(len(pmts)):
    np.ravel(ax)[i].hist(rec['init_wf'][:,i], bins=100, range=[0,400], label='PMT{} init_wf'.format(pmts[i]), histtype='step')
    np.ravel(ax)[i].hist(recB['init_wf'][:,i], bins=100, range=[0,400], label='PMT{} init_wf B'.format(pmts[i]), histtype='step')
    np.ravel(ax)[i].hist(rec0['init_wf'][:,i], bins=100, range=[0,400], label='PMT{} init_wf 0'.format(pmts[i]), histtype='step')
    np.ravel(ax)[i].legend(fontsize=15)

rec=rec[np.all(rec['init_wf']>20, axis=1)]
recB=recB[np.all(recB['init_wf']>20, axis=1)]
rec0=rec0[np.all(rec0['init_wf']>20, axis=1)]


fig, ax=plt.subplots(3,2)
fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('Co57', fontsize=25)
x=np.arange(1000)/5
for i in range(len(pmts)):
    np.ravel(ax)[i].hist(rec['blw'][:,i], bins=100, range=[0,30], label='PMT{} BLW'.format(pmts[i]), histtype='step')
    np.ravel(ax)[i].hist(recB['blw'][:,i], bins=100, range=[0,30], label='PMT{} BLW B'.format(pmts[i]), histtype='step')
    np.ravel(ax)[i].hist(rec0['blw'][:,i], bins=100, range=[0,30], label='PMT{} BLW 0'.format(pmts[i]), histtype='step')
    np.ravel(ax)[i].legend(fontsize=15)

plt.figure()
plt.hist(np.sqrt(np.sum(rec['blw']**2, axis=1)), bins=100, label='BLW', range=[0,30], histtype='step')
plt.hist(np.sqrt(np.sum(recB['blw']**2, axis=1)), bins=100, label='BLW B', range=[0,30], histtype='step')
plt.hist(np.sqrt(np.sum(rec0['blw']**2, axis=1)), bins=100, label='BLW B', range=[0,30], histtype='step')

plt.axvline(blw_cut, ymin=0, ymax=1, color='k')
plt.legend(fontsize=15)
rec=rec[np.sqrt(np.sum(rec['blw']**2, axis=1))<blw_cut]
recB=recB[np.sqrt(np.sum(recB['blw']**2, axis=1))<blw_cut]
rec0=rec0[np.sqrt(np.sum(rec0['blw']**2, axis=1))<blw_cut]


fig, ax=plt.subplots(3,2)
fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('Co57', fontsize=25)
for i in range(len(pmts)):
    np.ravel(ax)[i].hist(rec['chi2'][:,i], bins=100, label='PMT{} chi2'.format(pmts[i]), histtype='step')
    np.ravel(ax)[i].hist(recB['chi2'][:,i], bins=100, label='PMT{} chi2 B'.format(pmts[i]), histtype='step')
    np.ravel(ax)[i].hist(rec0['chi2'][:,i], bins=100, label='PMT{} chi2 0'.format(pmts[i]), histtype='step')
    np.ravel(ax)[i].set_yscale('log')
    np.ravel(ax)[i].legend(fontsize=15)


plt.figure()
plt.hist(np.sqrt(np.sum(rec['chi2']**2, axis=1)), bins=100, label='chi2', histtype='step')
plt.hist(np.sqrt(np.sum(recB['chi2']**2, axis=1)), bins=100, label='chi2 B', histtype='step')
plt.hist(np.sqrt(np.sum(rec0['chi2']**2, axis=1)), bins=100, label='chi2 0', histtype='step')

plt.yscale('log')
plt.axvline(chi2_cut, ymin=0, ymax=1, color='k')
plt.legend(fontsize=15)

rec=rec[np.sqrt(np.sum(rec['chi2']**2, axis=1))<chi2_cut]
recB=recB[np.sqrt(np.sum(recB['chi2']**2, axis=1))<chi2_cut]
rec0=rec0[np.sqrt(np.sum(rec0['chi2']**2, axis=1))<chi2_cut]

init=np.sum(np.sum(rec['h'][:,:10,:], axis=2), axis=1)
full=np.sum(np.sum(rec['h'][:,:100,:], axis=2), axis=1)

initB=np.sum(np.sum(recB['h'][:,:10,:], axis=2), axis=1)
fullB=np.sum(np.sum(recB['h'][:,:100,:], axis=2), axis=1)

init0=np.sum(np.sum(rec0['h'][:,:10,:], axis=2), axis=1)
full0=np.sum(np.sum(rec0['h'][:,:100,:], axis=2), axis=1)

plt.figure()
plt.hist(init/full, bins=100, range=[0,1], label='Relative number of PEs in first 10 ns', histtype='step')
plt.hist(initB/fullB, bins=100, range=[0,1], label='Relative number of PEs in first 10 ns B', histtype='step')
plt.hist(init0/full0, bins=100, range=[0,1], label='Relative number of PEs in first 10 ns 0', histtype='step')
plt.xlabel(r'$\omega$', fontsize=25)
plt.legend(fontsize=25)
plt.yscale('log')

rec=rec[init/full<0.5]
recB=recB[initB/fullB<0.5]
rec0=rec0[init0/full0<0.5]

x=np.linspace(0,300,500)
y=3*x+18
plt.figure()
plt.hist2d(up, dn, bins=[100, 100], norm=mcolors.PowerNorm(0.2), range=[[0,400], [0,400]])
plt.xlabel('Total number of PEs in PMTs 0, 1', fontsize=15)
plt.ylabel('Total number of PEs in PMTs 7, 8, 14', fontsize=15)

plt.plot(x, y, '--', linewidth=3)


plt.figure()
plt.hist(np.sum(np.sum(rec['h'][:,:100,:], axis=2), axis=1),  bins=np.arange(300)*4-2, label='All events', histtype='step', linewidth=3)
plt.hist(np.sum(np.sum(recB['h'][:,:100,:], axis=2), axis=1),  bins=np.arange(300)*4-2, label='All events B', histtype='step', linewidth=3)
plt.hist(np.sum(np.sum(rec0['h'][:,:100,:], axis=2), axis=1),  bins=np.arange(300)*4-2, label='All events 0', histtype='step', linewidth=3)
plt.legend(fontsize=15)
plt.axvline(left, ymin=0, ymax=1, color='k', label=left)
plt.axvline(right, ymin=0, ymax=1, color='k', label=right)

rec=rec[np.sum(np.sum(rec['h'][:,:100,:], axis=2), axis=1)>left]
recB=recB[np.sum(np.sum(recB['h'][:,:100,:], axis=2), axis=1)>left]
rec0=rec0[np.sum(np.sum(rec0['h'][:,:100,:], axis=2), axis=1)>left]
rec=rec[np.sum(np.sum(rec['h'][:,:100,:], axis=2), axis=1)<right]
recB=recB[np.sum(np.sum(recB['h'][:,:100,:], axis=2), axis=1)<right]
rec0=rec0[np.sum(np.sum(rec0['h'][:,:100,:], axis=2), axis=1)<right]


fig, ax=plt.subplots(3,2)
# fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('Co57', fontsize=25)
for i in range(len(pmts)):
    np.ravel(ax)[i].hist(np.sum(rec['h'][:,:100,i], axis=1), label='PMT{}'.format(pmts[i]), histtype='step', bins=np.arange(100)*2-1)
    np.ravel(ax)[i].hist(np.sum(recB['h'][:,:100,i], axis=1), label='PMT{} B'.format(pmts[i]), histtype='step', bins=np.arange(100)*2-1)
    np.ravel(ax)[i].hist(np.sum(rec0['h'][:,:100,i], axis=1), label='PMT{}'.format(pmts[i]), histtype='step', bins=np.arange(100)*2-1)
    np.ravel(ax)[i].legend(fontsize=15)
fig.text(0.5, 0.04, 'PEs in event', ha='center', fontsize=15)

fig, ax=plt.subplots(3,2)
fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('Co57', fontsize=25)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(np.mean(rec['h'][:,:,i], axis=0), 'k-.', label='PMT{} {}'.format(pmts[i], np.sum(np.mean(rec['h'][:,:100,i], axis=0))))
    np.ravel(ax)[i].plot(np.mean(recB['h'][:,:,i], axis=0), 'r-.', label='PMT{} {} B'.format(pmts[i], np.sum(np.mean(recB['h'][:,:100,i], axis=0))))
    np.ravel(ax)[i].plot(np.mean(rec0['h'][:,:,i], axis=0), 'g-.', label='PMT{} {} 0'.format(pmts[i], np.sum(np.mean(rec0['h'][:,:100,i], axis=0))))
    np.ravel(ax)[i].legend(fontsize=15)
plt.show()

# x=np.linspace(-80, 20)
# y=185/71*x+129
# up=np.sum(rec['h'][:,:100,0], axis=1)+np.sum(rec['h'][:,:100,0], axis=1)
# dn=np.sum(rec['h'][:,:100,-1], axis=1)+np.sum(rec['h'][:,:100,-2], axis=1)+np.sum(rec['h'][:,:100,-3], axis=1)
# plt.figure()
# plt.hist2d(up, dn, bins=[100, 100], norm=mcolors.PowerNorm(0.8))
# # plt.scatter(up, dn, marker='.')
# plt.plot(x, y, '--', linewidth=3)


plt.figure()
plt.hist(np.ravel(rec['h']), bins=np.arange(np.amax(np.ravel(rec['h']))+1)-0.5, label='PEs in 1 ns on each PMT', histtype='step')
plt.hist(np.ravel(np.sum(rec['h'],axis=2)), bins=np.arange(np.amax(np.ravel(np.sum(rec['h'],axis=2)))+1)-0.5, label='PEs in 1 ns globaly', histtype='step')
plt.yscale('log')
plt.legend()

rec=rec[init/full<0.5]



rec=rec[np.sum(np.sum(rec['h'][:,:100,:], axis=1), axis=1)>left]
rec=rec[np.sum(np.sum(rec['h'][:,:100,:], axis=1), axis=1)<right]
fig, ax=plt.subplots(3,2)
fig.suptitle('Co57 - Spec - slow', fontsize=25)
for i in range(len(pmts)):
    np.ravel(ax)[i].hist(np.sum(rec['h'][:,:100,i], axis=1),  bins=np.arange(100)*4-2, histtype='step', label='all')
    np.ravel(ax)[i].legend(fontsize=15)

fig, ax=plt.subplots(3,5)
k=0
for i in range(len(pmts)-1):
    hi=rec['h'][:,:,i]
    for j in range(i+1, len(pmts)):
        hj=rec['h'][:,:,j]
        np.ravel(ax)[k].hist((np.sum(hi, axis=1)-np.mean(np.sum(hi, axis=1)))*(np.sum(hj, axis=1)-np.mean(np.sum(hj, axis=1)))/(np.mean(np.sum(hj, axis=1))*np.mean(np.sum(hi, axis=1))),
                label='PMT{}-PMT{}'.format(pmts[i], pmts[j]), bins=100, range=[-1, 1])
        np.ravel(ax)[k].legend()
        k+=1
plt.show()
