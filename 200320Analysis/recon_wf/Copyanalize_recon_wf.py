import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import poisson, binom
from scipy.optimize import minimize
from fun import model
import sys

pmt=8
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
data=np.load(path+'spe.npz')
spe=data['Sig_trig']
spe_area=data['rec']['area']

path='/home/gerak/Desktop/DireXeno/190803/Co57/PMT{}/'.format(pmt)
data=np.load(path+'/raw_wf.npz'.format(pmt))
init=data['init']
data=np.load(path+'recon_wf.npz')
rec=data['rec']
WF=data['WF']
recon_WF=data['recon_WF']
chi2_cut=7e5
area=-np.sum(spe[init:])
left=37
right=125

rng=np.nonzero(np.logical_and(rec['chi2']<chi2_cut, rec['init']>=70))
spec=np.mean(rec[rng]['h'], axis=0)

x=np.arange(1000)/5
fig=plt.figure(figsize=(20,10))
ax=fig.add_subplot(111)
ax.plot(x, WF, 'k.-')
ax.plot(x, recon_WF, 'r.-')
ax.legend()

fig=plt.figure(figsize=(20,10))
ax=fig.add_subplot(221)
ax.plot(x, spec, 'k.', label='spec')
ax.legend()

ax=fig.add_subplot(222)
ax.hist(rec['chi2'], bins=100, label='chi2', range=[0,1e7])
ax.axvline(chi2_cut, ymin=0, ymax=1, color='k')
ax.set_yscale('log')
ax.legend()

ax=fig.add_subplot(223)
ha, ba, pa=ax.hist(spe_area, bins=100, label='spe area', range=[-2000, 4000])
xa=0.5*(ba[1:]+ba[:-1])
rnga=np.nonzero(np.logical_and(xa>-900, xa<3000))
ax.set_yscale('log')
ax.legend()

ax=fig.add_subplot(224)
h, bins, pat = ax.hist(np.sum(rec[rng]['h'], axis=1), bins=np.arange(201)-0.5, label='recon pe', histtype='step')
N=np.arange(len(h))
ax.plot(N[left:right], h[left:right], 'ro')
# ax.hist(rec[rng]['area']/area, bins=20, label='area', histtype='step', range=[0,200])
# x=0.5*(bins[1:]+bins[:-1])
# rng=np.nonzero(np.logical_and(x>left, x<right))[0]
# p0=[np.sum(h), np.sum(h*x)/np.sum(h), 5000, 1000, 150, 0, 500, 1500, 0.5]
# f, fa=model(x[rng].astype(int), h[rng], xa[rnga], ha[rnga], p0)
# ax.plot(x[rng], f, 'ro')
# ax.legend()
#
# ax=fig.add_subplot(223)
# ax.plot(xa[rnga], fa, 'ro')


rng=np.nonzero(np.logical_and(np.logical_and(rec['chi2']<chi2_cut, rec['init']>=init), np.logical_and(np.sum(rec['h'], axis=1)>left, np.sum(rec['h'], axis=1)<right)))[0]
H=np.zeros((10, 1000))
# for i in range(1000):
#     print(i)
#     H[:,i]=np.histogram(rec[rng]['h'][:,i], bins=np.arange(11)-0.5)[0]
# np.savez(path+'H', H=H, N_events=len(rng))
plt.show()
