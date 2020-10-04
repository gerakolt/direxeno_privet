import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import os
import sys

pmts=np.array([0,1,4,7,8,15])

BGpath='/home/gerak/Desktop/DireXeno/190803/BG/EventRecon/'
path='/home/gerak/Desktop/DireXeno/190803/Cs137/EventRecon/'
pathB='/home/gerak/Desktop/DireXeno/190803/Cs137B/EventRecon/'

blw_cut=15
init_cut=20
chi2_cut=5000
left=635
right=1000
T=1564823506349-1564820274767 #Cs
TB=1564825612162-1564824285761
# T=1564916315672-1564886605156 #Co
# TB=1564926608911-1564916365644 #CoB
TBG=1564874707904-1564826183355
data=np.load(BGpath+'recon1ns.npz')
BG=data['rec']
data=np.load(path+'recon1ns.npz')
rec=data['rec']
data=np.load(BGpath+'recon1ns.npz')
BG=data['rec']
data=np.load(pathB+'recon1ns.npz')
recB=data['rec']


recB=recB[np.all(recB['init_wf']>init_cut, axis=1)]
rec=rec[np.all(rec['init_wf']>init_cut, axis=1)]
BG=BG[np.all(BG['init_wf']>init_cut, axis=1)]

recB=recB[np.sqrt(np.sum(recB['blw']**2, axis=1))<blw_cut]
rec=rec[np.sqrt(np.sum(rec['blw']**2, axis=1))<blw_cut]
BG=BG[np.sqrt(np.sum(BG['blw']**2, axis=1))<blw_cut]

recB=recB[np.sqrt(np.sum(recB['chi2']**2, axis=1))<chi2_cut]
recB=recB[np.sum(np.sum(recB['h'][:,:100,:], axis=2), axis=1)>0]
rec=rec[np.sqrt(np.sum(rec['chi2']**2, axis=1))<chi2_cut]
rec=rec[np.sum(np.sum(rec['h'][:,:100,:], axis=2), axis=1)>0]
BG=BG[np.sqrt(np.sum(BG['chi2']**2, axis=1))<chi2_cut]
BG=BG[np.sum(np.sum(BG['h'][:,:100,:], axis=2), axis=1)>0]

init=np.sum(np.sum(rec['h'][:,:10,:], axis=2), axis=1)
full=np.sum(np.sum(rec['h'][:,:100,:], axis=2), axis=1)
initB=np.sum(np.sum(recB['h'][:,:10,:], axis=2), axis=1)
fullB=np.sum(np.sum(recB['h'][:,:100,:], axis=2), axis=1)
BGinit=np.sum(np.sum(BG['h'][:,:10,:], axis=2), axis=1)
BGfull=np.sum(np.sum(BG['h'][:,:100,:], axis=2), axis=1)

flash_cut=0.5
recB=recB[initB/fullB<flash_cut]
rec=rec[init/full<flash_cut]
BG=BG[BGinit/BGfull<flash_cut]

up=np.sum(rec['h'][:,:100,0], axis=1)+np.sum(rec['h'][:,:100,1], axis=1)
dn=np.sum(rec['h'][:,:100,-1], axis=1)+np.sum(rec['h'][:,:100,-2], axis=1)+np.sum(rec['h'][:,:100,-3], axis=1)
rec=rec[dn<3*up+18]

up=np.sum(recB['h'][:,:100,0], axis=1)+np.sum(recB['h'][:,:100,1], axis=1)
dn=np.sum(recB['h'][:,:100,-1], axis=1)+np.sum(recB['h'][:,:100,-2], axis=1)+np.sum(recB['h'][:,:100,-3], axis=1)
recB=recB[dn<3*up+18]

up=np.sum(BG['h'][:,:100,0], axis=1)+np.sum(BG['h'][:,:100,1], axis=1)
dn=np.sum(BG['h'][:,:100,-1], axis=1)+np.sum(BG['h'][:,:100,-2], axis=1)+np.sum(BG['h'][:,:100,-3], axis=1)
BG=BG[dn<3*up+18]

BGhist, bins=np.histogram(np.sum(np.sum(BG['h'][:,:100,:], axis=2), axis=1),  bins=np.arange(250)*5)
hist, bins=np.histogram(np.sum(np.sum(rec['h'][:,:100,:], axis=2), axis=1),  bins=np.arange(250)*5)
histB, bins=np.histogram(np.sum(np.sum(recB['h'][:,:100,:], axis=2), axis=1),  bins=np.arange(250)*5)
x=0.5*(bins[1:]+bins[:-1])
plt.figure(figsize=(20,10))
plt.step(x, 1000*hist/T*1355/540, where='mid', linewidth=5, label=r'$^{137}$'+'Cs', color='b')
plt.step(x, 1000*histB/TB, where='mid', linewidth=5, label=r'$^{137}$'+'Cs - 90'+r'$^{\circ} \quad$'+'rotated', color='r')
plt.bar(x, 1000*BGhist/TBG, width=bins[1:]-bins[:-1], alpha=0.5, label='Background', color='k')
plt.axvline(left, 0, 0.25, linewidth=5, color='k', label='Range around\n full absorption peak')
plt.axvline(right, 0, 0.25, linewidth=5, color='k')
plt.legend(fontsize=35)
plt.xlabel('PEs', fontsize=25)
plt.ylabel('Rate [Hz]', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.show()
