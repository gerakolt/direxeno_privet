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


path='/home/gerak/Desktop/DireXeno/190803/BG/EventRecon/'
blw_cut=6.5
init_cut=20
chi2_cut=1050
left=6
right=30
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
# rec=rec[np.all(rec['init_wf'], axis=1)>init_cut]

print(len(rec))

fig, ((ax1, ax2), (ax3, ax4))=plt.subplots(2,2)
fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('Background Data', fontsize=25)
x=np.arange(1000)/5
ax1.plot(x, WFs[0], 'r1', label='PMT7 WF')
ax1.plot(x, recon_WFs[0], 'b-.', label='PMT7 reconstructed WF')
ax1.plot(x, WFs[1], 'g1', label='PMT8 WF')
ax1.plot(x, recon_WFs[1], 'y-.', label='PMT8 reconstructed WF')
ax1.legend(fontsize=15)

ax2.hist(np.sqrt(rec['blw'][:,0]**2+rec['blw'][:,1]**2), bins=100, label='BLW', range=[0,30])
ax2.axvline(blw_cut, ymin=0, ymax=1, color='k')
ax2.legend(fontsize=15)

ax3.hist(np.sqrt(rec['chi2'][:,0]**2+rec['chi2'][:,1]**2), bins=100, label=r'$\chi^2$', range=[0,1e4])
ax3.axvline(chi2_cut, ymin=0, ymax=1, color='k')
ax3.set_yscale('log')
ax3.legend(fontsize=15)

rec=rec[np.logical_and(np.sqrt(rec['chi2'][:,0]**2+rec['chi2'][:,1]**2)<chi2_cut, np.sqrt(rec['blw'][:,0]**2+rec['blw'][:,1]**2)<blw_cut)]
print(rec[np.logical_and(np.sum(np.sum(rec['h'], axis=1), axis=1)<25, np.sum(np.sum(rec['h'], axis=1), axis=1)>10)]['id'])
ax4.hist(np.sum(np.sum(rec['h'], axis=1), axis=1), label='Number of PEs in event', bins=np.arange(200)-0.5)
ax4.axvline(left, ymin=0, ymax=1, color='k')
ax4.axvline(right, ymin=0, ymax=1, color='k')
ax4.legend(fontsize=15)

ax1.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
ax2.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
ax3.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
ax4.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')


plt.show()
