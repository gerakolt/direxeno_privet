import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import poisson, binom
from scipy.optimize import curve_fit



chi2=3e6
left=38*5
right=175*5
pmt=1
Data=np.load('PMT{}/simWFs.npz'.format(pmt))
H=Data['H']

Data=np.load('PMT{}/spectra.npz'.format(pmt))
spec=Data['spectrum']
WF=Data['mean_WF']
Recon_WF=Data['Recon_wf']
Chi2=Data['Chi2']
dif=Data['dif']
ID=Data['ID']
PE_by_area=Data['PE_by_area']
N=len(spec[:,0])


t=np.arange(1000)/5
fig = plt.figure(figsize=(20,10))
fig.suptitle('SIM - PMT {} - {} events'.format(pmt, N), fontsize=25)

ax=fig.add_subplot(321)
ax.plot(t, np.mean(spec, axis=0), 'k.-', label='reconstructed')
x=[]
y=[]
ax.fill_between(t[left:right], y1=0, y2=np.mean(spec, axis=0)[left:right])
ax.legend(fontsize=25)
ax.set_ylabel('PEs', fontsize=30)

ax=fig.add_subplot(322)
ax.hist(Chi2, bins=100, label='chi2')
ax.legend()


ax=fig.add_subplot(323)
ax.plot(np.arange(1000)/5, WF/N, 'k.', label='Summed signal')
ax.plot(np.arange(1000)/5, Recon_WF/N, 'r.', label='Summed reconstructed\n signal')
ax.plot(np.arange(1000)/5, WF/N-Recon_WF/N, 'g--', label='Summed reconstructed\n signal')
ax.legend(fontsize=15, loc='best')

ax=fig.add_subplot(324)
ax.plot(dif/N, 'k.')
t=[]
y=[]
for i in range(int(np.floor(len(dif)/9))):
    t.append(4+i*9)
    y.append(np.mean(dif[i*9:(i+1)*9]/N))
ax.plot(t,y, 'r.-')
ax.axhline(0, xmin=0, xmax=1, color='g')
ax.legend(fontsize=25)

ax=fig.add_subplot(325)
h,bins, pach=ax.hist(PE_by_area, bins=np.arange(30,100)-0.5, label='PE_by_area', histtype='step')
h,bins, pach=ax.hist(np.sum(spec, axis=1), bins=np.arange(30,100)-0.5, label='recon PEs', histtype='step')
h,bins, pach=ax.hist(np.sum(H, axis=1), bins=np.arange(30,100)-0.5, label='Oreginal PEs', histtype='step')
dif_pe=-np.sum(H[:len(spec)], axis=1)+np.sum(spec, axis=1)
h,bins, pach=ax.hist(dif_pe, bins=np.arange(-20,20)-0.5, label='{:3.2f}+-{:3.2f}'.format(np.mean(dif_pe), np.std(dif_pe)), histtype='step')



ax.legend()
plt.show()
