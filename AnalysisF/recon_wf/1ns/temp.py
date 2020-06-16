import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from PMTgiom import make_pmts
from YLM import Ylm
import matplotlib.colors as mcolors


pmts=np.array([0,1,4,7,8,14])
pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn, r=make_pmts(pmts)


path='/home/gerak/Desktop/DireXeno/190803/Cs137/EventRecon/'
blw_cut=25
init_cut=20
chi2_cut=10000
left=150
right=250
data=np.load(path+'recon1ns98999.npz')

rec=data['rec']
# rec=rec[rec['sat']==1]
WFs=data['WFs']
recon_WFs=data['recon_WFs']


rec=rec[np.all(rec['init_wf']>20, axis=1)]
rec=rec[np.sqrt(np.sum(rec['blw']**2, axis=1))<blw_cut]
rec=rec[np.sqrt(np.sum(rec['chi2']**2, axis=1))<chi2_cut]

init=np.sum(np.sum(rec['h'][:,:10,:], axis=2), axis=1)
full=np.sum(np.sum(rec['h'][:,:100,:], axis=2), axis=1)
rec=rec[init/full<0.5]
rec0=rec

plt.figure()
plt.hist(np.sum(np.sum(rec['h'][:,:100,:], axis=1), axis=1), bins=100, range=[100,500])

# rec=rec[np.sum(np.sum(rec['h'][:,:100,:], axis=1), axis=1)>left]
# rec=rec[np.sum(np.sum(rec['h'][:,:100,:], axis=1), axis=1)<right]
alm=np.recarray(len(rec), dtype=[
    ('a0', 'f8',1),
    ('a1', 'f8',3)
    ])

for i in range(len(rec)):
    alm['a0'][i], alm['a1'][i]=Ylm(np.sum(rec['h'][i, :100,:], axis=0))
rec3=rec[alm['a1'][:,1]/alm['a0']<0.2068]
rec1=rec[np.sum(rec['h'][:,:100,1], axis=1)<15]
rec2=rec1[np.sum(rec1['h'][:,:100,0], axis=1)<20]

fig, ax=plt.subplots(3,2)
fig.suptitle('Co57 - Spec - slow', fontsize=25)
for i in range(len(pmts)):
    np.ravel(ax)[i].hist(np.sum(rec['h'][:,:100,i], axis=1),  bins=np.arange(150), histtype='step', label='all')
    np.ravel(ax)[i].hist(np.sum(rec1['h'][:,:100,i], axis=1),  bins=np.arange(150), histtype='step', label='1')
    np.ravel(ax)[i].hist(np.sum(rec2['h'][:,:100,i], axis=1),  bins=np.arange(150), histtype='step', label='2')
    np.ravel(ax)[i].hist(np.sum(rec3['h'][:,:100,i], axis=1),  bins=np.arange(150), histtype='step', label='3')
    np.ravel(ax)[i].legend(fontsize=15)

fig, ax=plt.subplots(3,2)
fig.suptitle('Co57 - Spec - slow', fontsize=25)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(np.mean(rec['h'][:,:,i], axis=0),  'k.', label='all')
    np.ravel(ax)[i].plot(np.mean(rec2['h'][:,:,i], axis=0), 'r.', label='2')
    np.ravel(ax)[i].legend(fontsize=15)


alm2=np.recarray(len(rec2), dtype=[
    ('a0', 'f8',1),
    ('a1', 'f8',3)
    ])


for i in range(len(rec2)):
    alm2['a0'][i], alm2['a1'][i]=Ylm(np.sum(rec2['h'][i, :100,:], axis=0))

plt.figure()
plt.hist(alm['a1'][:,1]/alm['a0'], bins=100, histtype='step', label='a0')
plt.hist(alm2['a1'][:,1]/alm2['a0'], bins=100, histtype='step', label='a0')

plt.figure()
plt.hist2d(alm['a1'][:,1]/alm['a0'], np.sum(np.sum(rec['h'][:,:100,:], axis=1), axis=1), bins=[100,100] ,range=[[0, 0.5],[0,700]], norm=mcolors.PowerNorm(0.3))

# plt.figure()
# plt.hist2d(alm2['a1'][:,1]/alm2['a0'], np.sum(np.sum(rec2['h'][:,:100,:], axis=1), axis=1), bins=[100,100] ,range=[[0.18, 0.23],[left, right]], norm=mcolors.PowerNorm(0.3))

plt.legend()
plt.show()
