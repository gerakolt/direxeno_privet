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
# Rec=np.recarray(100000, dtype=[
#     ('area', 'i8', len(pmts)),
#     ('blw', 'f8', len(pmts)),
#     ('id', 'i8'),
#     ('chi2', 'f8', len(pmts)),
#     ('h', 'i8', (200, len(pmts))),
#     ('init_event', 'i8'),
#     ('init_wf', 'i8', len(pmts))
#     ])
# j=0
# id=0
# WFs=np.zeros((len(pmts), 1000))
# recon_WFs=np.zeros((len(pmts), 1000))
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
#         os.remove(path+filename)
# np.savez(path+'recon1ns{}'.format(id), rec=Rec[:j-1], WFs=WFs, recon_WFs=recon_WFs)
# rec=Rec[:j-1]
data=np.load(path+'recon1ns81785.npz')
rec=data['rec']
# rec=rec[rec['sat']==1]
WFs=data['WFs']
recon_WFs=data['recon_WFs']

fig, ax=plt.subplots(4,4)
fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('Co57', fontsize=25)
x=np.arange(1000)/5
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(x, WFs[i], 'r1', label='PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].plot(x, recon_WFs[i], 'b-.')
    np.ravel(ax)[i].legend(fontsize=12)


fig, ax=plt.subplots(4,4)
# fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('Co57', fontsize=25)
x=np.arange(1000)/5
for i in range(len(pmts)):
    np.ravel(ax)[i].hist(rec['init_wf'][:,i], bins=100, range=[0,400], label='PMT{} init_wf'.format(pmts[i]))
    np.ravel(ax)[i].legend(fontsize=15)

# print(rec['id'][rec['init_wf'][:,0]<4])
# print(rec['init_wf'][rec['init_wf'][:,0]<4][:,0])
rec=rec[np.all(rec['init_wf']>20, axis=1)]


# dt=np.zeros((len(rec['h']),len(pmts)-1))
# for i,ev in enumerate(rec['h']):
#     for j in range(len(pmts)-1):
#         try:
#             dt[i,j]=np.amin(np.nonzero(ev[:,j]>0)[0])-np.amin(np.nonzero(ev[:,-1]>0)[0])
#         except:
#             dt[i,j]=100
#
# fig, ax=plt.subplots(4,4)
# # fig.subplots_adjust(wspace=0, hspace=0)
# fig.suptitle('Co57', fontsize=25)
# for i in range(len(pmts)-1):
#     np.ravel(ax)[i].hist(dt[:,i], bins=np.arange(40)-20.5, label='First PE delay PMT{}-PMT14'.format(pmts[i]))
#     np.ravel(ax)[i].legend(fontsize=15)



fig, ax=plt.subplots(4,4)
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


fig, ax=plt.subplots(4,4)
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
full=np.sum(np.sum(rec['h'], axis=2), axis=1)
s_rec=rec[init/full<0.5]
f_rec=rec[init/full>=0.5]

fig, ax=plt.subplots(4,4)
fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('Co57', fontsize=25)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(np.mean(rec['h'][:,:,i], axis=0), 'k-.', label='PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].plot(np.mean(s_rec['h'][:,:,i], axis=0), 'r-.', label='slow, PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].plot(np.mean(f_rec['h'][:,:,i], axis=0), 'g-.', label='fast, PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].legend(fontsize=15)


plt.figure()
plt.plot(np.mean(np.mean(rec['h'][:,:,:], axis=2), axis=0), 'k-.', label='ALL PMTs')
plt.plot(np.mean(np.mean(s_rec['h'][:,:,:], axis=2), axis=0), 'r-.', label='slow ALL PMTs')
plt.plot(np.mean(np.mean(f_rec['h'][:,:,:], axis=2), axis=0), 'g-.', label='fast ALL PMTs')
plt.legend(fontsize=15)


fig, ax=plt.subplots(4,4)
# fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('Co57', fontsize=25)
for i in range(len(pmts)):
    np.ravel(ax)[i].hist(np.sum(rec['h'][:,:,i], axis=1),  bins=np.arange(100)-0.5, histtype='step', label='all')
    np.ravel(ax)[i].legend(fontsize=15)

plt.figure()
plt.hist(np.sum(np.sum(rec['h'], axis=2), axis=1),  bins=np.arange(400)-0.5, histtype='step', label='all')
plt.hist(np.sum(np.sum(s_rec['h'], axis=2), axis=1),  bins=np.arange(400)-0.5, histtype='step', label='slow')
plt.hist(np.sum(np.sum(f_rec['h'], axis=2), axis=1), bins=np.arange(400)-0.5, histtype='step', label='fast')

plt.axvline(left, ymin=0, ymax=1, color='k', label=left)
plt.axvline(right, ymin=0, ymax=1, color='k', label=right)
plt.legend()

# fig, ax=plt.subplots(4,4)
# fig.subplots_adjust(wspace=0, hspace=0)
# fig.suptitle('Covariance', fontsize=25)
# j=0
# k=1
# for i in range(len(np.ravel(ax))):
#     if j<len(pmts)-1:
#         print('PMT{}-PMT{}:'.format(pmts[j], pmts[k]))
#         print('Var: ', np.var(np.sum(rec['h'][:,:,j], axis=1)), np.var(np.sum(rec['h'][:,:,k], axis=1)))
#         print('Cov: ', np.cov(np.sum(rec['h'][:,:,j], axis=1), np.sum(rec['h'][:,:,k], axis=1)))
#         np.ravel(ax)[i].scatter(np.sum(rec['h'][:,:,j], axis=1)-np.mean(np.sum(rec['h'][:,:,j], axis=1)), np.sum(rec['h'][:,:,k], axis=1)-np.mean(np.sum(rec['h'][:,:,k], axis=1)), color='k', label='PMT{} - PMT{}'.format(pmts[j], pmts[k]))
#         np.ravel(ax)[i].legend(fontsize=15)
#         if k<len(pmts)-1:
#             k+=1
#         else:
#             j+=1
#             k=j+1
#     else:
#         continue

plt.figure()
init=np.sum(np.sum(rec['h'][:,:10,:], axis=2), axis=1)
full=np.sum(np.sum(rec['h'], axis=2), axis=1)
plt.hist(init/full, bins=100, label='PEs W in 10 ns')
plt.legend()

fig, ax=plt.subplots(3,2)
# fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('Co57', fontsize=25)
for i in range(len(pmts)):
    init=np.sum(rec['h'][:,:10,i], axis=1)
    full=np.sum(rec['h'][:,:,i], axis=1)
    np.ravel(ax)[i].hist(init/full, bins=100, label='PEs W in 10 ns', range=[0,1])
    np.ravel(ax)[i].legend(fontsize=15)



plt.figure()
plt.hist(np.ravel(rec['h']), bins=np.arange(np.amax(np.ravel(rec['h']))+1)-0.5, label='PEs in 1 ns on each PMT', histtype='step')
plt.hist(np.ravel(np.sum(rec['h'],axis=2)), bins=np.arange(np.amax(np.ravel(np.sum(rec['h'],axis=2)))+1)-0.5, label='PEs in 1 ns globaly', histtype='step')
plt.yscale('log')
plt.legend()

plt.show()
