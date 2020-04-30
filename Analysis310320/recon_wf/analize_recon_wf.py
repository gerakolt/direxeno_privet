import numpy as np
import matplotlib.pyplot as plt
import time
import os

pmts=[7,8]
spe_area=np.zeros(len(pmts))
for i, pmt in enumerate(pmts):
    cuts=np.load('/home/gerak/Desktop/DireXeno/190803/pulser/NEWPMT{}/cuts.npz'.format(pmt))
    left=cuts['left']
    right=cuts['right']
    spe=np.load('/home/gerak/Desktop/DireXeno/190803/pulser/NEWPMT{}/areas.npz'.format(pmt))['spe']
    spe_area[i]=-np.sum(spe[left:right])


path='/home/gerak/Desktop/DireXeno/190803/Co57/EventRecon/'
blw_cut=4.7
init_cut=20
chi2_cut=500
Rec=np.recarray(500000, dtype=[
    ('area', 'i8', len(pmts)),
    ('blw', 'f8', len(pmts)),
    ('id', 'i8'),
    ('chi2', 'f8', len(pmts)),
    ('h', 'i8', (1000, len(pmts))),
    ('init', 'i8'),
    ('init_wf', 'i8', len(pmts)),
    ])
j=0
id=0
WFs=np.zeros((len(pmts), 1000))
recon_WFs=np.zeros((len(pmts), 1000))
for filename in os.listdir(path):
    if filename.endswith(".npz") and filename.startswith("recon"):
        print(filename)
        data=np.load(path+filename)
        rec=data['rec']
        WFs+=data['WFs']
        recon_WFs+=data['recon_WFs']
        for r in rec:
            Rec[j]['area']=r['area']
            Rec[j]['blw']=r['blw']
            Rec[j]['id']=r['id']
            Rec[j]['chi2']=r['chi2']
            Rec[j]['init_wf']=r['init_wf']
            Rec[j]['h']=r['h']
            Rec[j]['init']=r['init']
            if r['id']>id:
                id=r['id']
            j+=1
        os.remove(path+filename)
np.savez(path+'recon{}'.format(id), rec=Rec[:j-1], WFs=WFs, recon_WFs=recon_WFs)
rec=Rec[:j-1]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)
fig.suptitle('init_wf')
ax1.hist(rec['init_wf'][:,0], bins=np.arange(200)-0.5)
ax2.hist(rec['init_wf'][:,1], bins=np.arange(200)-0.5)
ax1.axvline(init_cut, ymin=0, ymax=1, color='k')
ax2.axvline(init_cut, ymin=0, ymax=1, color='k')
ax3.axvline(init_cut, ymin=0, ymax=1, color='k')

rec=rec[np.all(rec['init_wf']>init_cut, axis=1)]


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)
fig.suptitle('BLW')
ax1.hist(rec['blw'][:,0], bins=100, range=[0,15])
ax1.axvline(blw_cut, ymin=0, ymax=1, color='k')
ax2.hist(rec['blw'][:,1], bins=100, range=[0,15])
ax2.axvline(blw_cut, ymin=0, ymax=1, color='k')


rec=rec[np.all(rec['blw']<blw_cut, axis=1)]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)
fig.suptitle('Chi2')
ax1.hist(rec['chi2'][:,0], bins=100, range=[0,2000])
ax2.hist(rec['chi2'][:,1], bins=100, range=[0,2000])
ax1.axvline(chi2_cut, ymin=0, ymax=1, color='k')
ax2.axvline(chi2_cut, ymin=0, ymax=1, color='k')
ax3.axvline(chi2_cut, ymin=0, ymax=1, color='k')

rec=rec[np.all(rec['chi2']<chi2_cut, axis=1)]


x=np.arange(1000)/5
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
fig.suptitle('Mean WF')
ax1.plot(x, WFs[0], 'k.')
ax1.plot(x, recon_WFs[0], 'r.')
ax2.plot(x, WFs[1], 'k.')
ax2.plot(x, recon_WFs[1], 'r.')
ax4.plot(x, WFs[0])
ax4.plot(x, WFs[1])
ax1.axhline(0, xmin=0, xmax=1)
ax2.axhline(0, xmin=0, xmax=1)
ax3.axhline(0, xmin=0, xmax=1)
ax4.axhline(0, xmin=0, xmax=1)


fig, (ax1, ax2, ax3) = plt.subplots(3,1)
fig.suptitle('Temporal')
ax1.plot(x, np.mean(rec['h'][:,:,0], axis=0), 'r.')
ax2.plot(x, np.mean(rec['h'][:,:,1], axis=0), 'r.')
ax1.legend()
ax2.legend()
ax3.legend()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)
fig.suptitle('Spectrum')
ax1.hist(np.sum(rec['h'][:,:,0], axis=1), bins=np.arange(400)-0.5, histtype='step', label='pes')
ax2.hist(np.sum(rec['h'][:,:,1], axis=1), bins=np.arange(400)-0.5, histtype='step', label='pes')
ax1.legend()
ax2.legend()
ax3.legend()
ax4.hist(np.sum(np.sum(rec['h'][:,:,:], axis=2), axis=1), bins=np.arange(700)-0.5)

plt.show()
