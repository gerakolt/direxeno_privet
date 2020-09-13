import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

pmts=np.array([0,1,4,7,8,15])


path='/home/gerak/Desktop/DireXeno/190803/pulser/EventRecon/'
blw_cut=25
init_cut=10
chi2_cut=10000
left=170
right=230
Rec=np.recarray(100000, dtype=[
    ('area', 'i8', len(pmts)),
    ('blw', 'f8', len(pmts)),
    ('id', 'i8'),
    ('chi2', 'f8', len(pmts)),
    ('h', 'i8', (200, len(pmts))),
    ('init_event', 'i8'),
    ('init_wf', 'i8', len(pmts))
    ])
j=0
id=0
WFs=np.zeros((len(pmts), 1000))
recon_WFs=np.zeros((len(pmts), 1000))
# for filename in os.listdir(path):
#     if filename.endswith(".npz") and filename.startswith("recon1ns"):
#         print(filename)
#         data=np.load(path+filename)
#         rec=data['rec']
#         WFs+=data['WFs']
#         recon_WFs+=data['recon_WFs']
#         for r in rec:
#             Rec[j]['area']=r['area']
#             Rec[j]['blw']=r['blw']
#             Rec[j]['id']=r['id']
#             Rec[j]['chi2']=r['chi2']
#             Rec[j]['init_wf']=r['init_wf']
#             Rec[j]['h']=r['h']
#             Rec[j]['init_event']=r['init_event']
#             if r['id']>id:
#                 id=r['id']
#             j+=1
# # sys.exit()
#         os.remove(path+filename)
# np.savez(path+'recon1ns'.format(id), rec=Rec[:j-1], WFs=WFs, recon_WFs=recon_WFs)

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
# fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('Co57', fontsize=25)
x=np.arange(1000)/5
for i in range(len(pmts)):
    np.ravel(ax)[i].hist(rec['init_wf'][:,i], bins=100, range=[0,400], label='PMT{} init_wf'.format(pmts[i]))
    np.ravel(ax)[i].legend(fontsize=15)
rec=rec[np.all(rec['init_wf']>init_cut, axis=1)]



fig, ax=plt.subplots(2,3)
fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('Co57', fontsize=25)
x=np.arange(1000)/5
for i in range(len(pmts)):
    np.ravel(ax)[i].hist(rec['blw'][:,i], bins=100, range=[0,100], label='PMT{} BLW'.format(pmts[i]))
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

init=np.sum(np.sum(rec['h'][:,:10,:], axis=2), axis=1)
full=np.sum(np.sum(rec['h'][:,:100,:], axis=2), axis=1)

# plt.figure()
# plt.hist(init/full, bins=100, range=[0,1], label='Relative number of PEs in first 10 ns')
# plt.xlabel(r'$\omega$', fontsize=25)
# plt.legend(fontsize=25)
#
# s_rec=rec[init/full<0.5]
# f_rec=rec[init/full>=0.5]
#
#
fig, ax=plt.subplots(3,2)
fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('Co57', fontsize=25)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(np.mean(rec['h'][:,:,i], axis=0), 'k-.', label='PMT{}'.format(pmts[i]))
#     np.ravel(ax)[i].plot(np.mean(s_rec['h'][:,:,i], axis=0), 'r-.', label='slow, PMT{}'.format(pmts[i]))
#     np.ravel(ax)[i].plot(np.mean(f_rec['h'][:,:,i], axis=0), 'g-.', label='fast, PMT{}'.format(pmts[i]))
#     np.ravel(ax)[i].legend(fontsize=15)


# plt.figure()
# plt.plot(np.sum(np.sum(s_rec['h'][:,:,:], axis=2), axis=0)/len(rec['h']), 'r-.', label='Scintillation', linewidth=5)
# plt.plot(np.sum(np.sum(f_rec['h'][:,:,:], axis=2), axis=0)/len(rec['h']), 'g-.', label='PMT Flashing', linewidth=5)
# plt.xlabel('Time [ns]', fontsize=15)
# plt.legend(fontsize=15)


plt.figure()
plt.hist(np.sum(np.sum(rec['h'][:,:100,:], axis=2), axis=1),  bins=np.arange(10), alpha=0.3, label='All events')
# plt.hist(np.sum(np.sum(s_rec['h'][:,:100,:], axis=2), axis=1),  bins=np.arange(300)*4-2, histtype='step', label=r'$^{137}$Cs', linewidth=4)
# plt.hist(np.sum(np.sum(f_rec['h'][:,:100,:], axis=2), axis=1), bins=np.arange(300)*4-2, histtype='step', label='Flashing PMTs', linewidth=4)

# plt.axvline(left, ymin=0, ymax=1, color='k', label=left)
# plt.axvline(right, ymin=0, ymax=1, color='k', label=right)
plt.legend(fontsize=15)


# plt.figure()
# plt.hist(np.ravel(rec['h']), bins=np.arange(np.amax(np.ravel(rec['h']))+1)-0.5, label='PEs in 1 ns on each PMT', histtype='step')
# plt.hist(np.ravel(np.sum(rec['h'],axis=2)), bins=np.arange(np.amax(np.ravel(np.sum(rec['h'],axis=2)))+1)-0.5, label='PEs in 1 ns globaly', histtype='step')
# plt.yscale('log')
# plt.legend()
# rec=rec[init/full<0.5]
# rec=rec[np.sum(np.sum(rec['h'][:,:100,:], axis=1), axis=1)>left]
# rec=rec[np.sum(np.sum(rec['h'][:,:100,:], axis=1), axis=1)<right]
fig, ax=plt.subplots(2,3)
fig.suptitle('Co57 - Spec - slow', fontsize=25)
for i in range(len(pmts)):
    np.ravel(ax)[i].hist(np.sum(rec['h'][:,:,i], axis=1),  bins=np.arange(10)-0.5, histtype='step', label='all')
    np.ravel(ax)[i].legend(fontsize=15)

# fig, ax=plt.subplots(3,5)
# k=0
# for i in range(len(pmts)-1):
#     hi=rec['h'][:,:,i]
#     for j in range(i+1, len(pmts)):
#         hj=rec['h'][:,:,j]
#         np.ravel(ax)[k].hist((np.sum(hi, axis=1)-np.mean(np.sum(hi, axis=1)))*(np.sum(hj, axis=1)-np.mean(np.sum(hj, axis=1)))/(np.mean(np.sum(hj, axis=1))*np.mean(np.sum(hi, axis=1))),
#                 label='PMT{}-PMT{}'.format(pmts[i], pmts[j]), bins=100, range=[-1, 1])
#         np.ravel(ax)[k].legend()
#         k+=1

plt.show()
