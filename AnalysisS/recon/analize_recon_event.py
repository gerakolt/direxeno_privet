import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import os
import sys

pmts=np.array([0,1,4,7,8,15])

BGpath='/home/gerak/Desktop/DireXeno/190803/BG/EventRecon/'
path='/home/gerak/Desktop/DireXeno/190803/Cs137/EventRecon/'
blw_cut=15
init_cut=20
chi2_cut=5000
left=800
right=1000
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

data=np.load(BGpath+'recon1ns.npz')
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

plt.figure()
plt.hist(init/full, bins=100, range=[0,1], label='Relative number of PEs in first 10 ns')
rec=rec[init/full<0.5]
BG=BG[BGinit/BGfull<0.5]

fig, ax=plt.subplots(3,2)
fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('Co57', fontsize=25)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(np.mean(rec['h'][:,:,i], axis=0), 'k-.', label='PMT{}'.format(pmts[i]))

plt.figure()
up=np.sum(rec['h'][:,:100,0], axis=1)+np.sum(rec['h'][:,:100,1], axis=1)
dn=np.sum(rec['h'][:,:100,-1], axis=1)+np.sum(rec['h'][:,:100,-2], axis=1)+np.sum(rec['h'][:,:100,-3], axis=1)
plt.plot(np.arange(450), np.arange(450)*3+18, 'k--')
plt.hist2d(up, dn, bins=[100, 100], range=[[0,350], [0,700]], norm=mcolors.PowerNorm(0.3))
plt.xlabel('Sum of PEs in the top floor PMTs', fontsize=25)
plt.ylabel('Sum of PEs in the bottom floor PMTs', fontsize=25)
rec0=rec
rec=rec[dn<3*up+18]
plt.legend(fontsize=15)

TB=1564926608911-1564916365644
TA=1564916315672-1564886605156
TBG=1564874707904-1564826183355
TCs=1564823506349-1564820274767

hist, bins=np.histogram(np.sum(np.sum(BG['h'][:,:100,:], axis=2), axis=1),  bins=np.arange(250)*4)
plt.figure()
plt.hist(np.sum(np.sum(rec['h'][:,:100,:], axis=2), axis=1),  bins=np.arange(250)*5, histtype='step', linewidth=5, label='All events')
# plt.bar(0.5*(bins[1:]+bins[:-1]) ,TCs/TBG*hist, label='BG', width=bins[1:]-bins[:-1], color='orange', alpha=0.5)
plt.axvline(left, 0 ,1, color='k')
plt.axvline(right, 0 ,1, color='k')
plt.legend(fontsize=15)

rec=rec[np.sum(np.sum(rec['h'][:,:100,:], axis=1), axis=1)>=left]
rec=rec[np.sum(np.sum(rec['h'][:,:100,:], axis=1), axis=1)<=right]
rec0=rec0[np.sum(np.sum(rec0['h'][:,:100,:], axis=1), axis=1)>=left]
rec0=rec0[np.sum(np.sum(rec0['h'][:,:100,:], axis=1), axis=1)<=right]
fig, ax=plt.subplots(2,3)
# fig.suptitle('Co57 - Spec - slow', fontsize=25)
for i in range(len(pmts)):
    np.ravel(ax)[i].hist(np.sum(rec['h'][:,:,i], axis=1),  bins=np.arange(500), histtype='step', label='After\n up-dn cut\n PMT{}'.format(i), linewidth=3)
    np.ravel(ax)[i].hist(np.sum(rec0['h'][:,:,i], axis=1),  bins=np.arange(500), histtype='step', label='Before\n up-dn cut', linewidth=3)
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
