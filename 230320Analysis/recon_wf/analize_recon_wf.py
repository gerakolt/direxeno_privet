import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import poisson, binom
from scipy.optimize import minimize
from fun import fit_spectra
from scipy.optimize import curve_fit
import sys

def func(x, a,b,c,m_bl,s_bl,m,s):
    return a*np.exp(-0.5*(x-m_bl)**2/s_bl**2)+b*np.exp(-0.5*(x-(m_bl+m))**2/(s_bl**2+s**2))+c*np.exp(-0.5*(x-(m_bl+2*m))**2/(s_bl**2+2*s**2))

pmt=0
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
data=np.load(path+'spe.npz')
spe=data['spe']
areas=data['areas']
h_area=data['h_area']
p_area=data['p_area']

path='/home/gerak/Desktop/DireXeno/190803/BG/PMT{}/'.format(pmt)
data=np.load(path+'/raw_wf.npz'.format(pmt))
data=np.load(path+'recon_wf.npz')
rec_BG=data['rec']


path='/home/gerak/Desktop/DireXeno/190803/Co57/PMT{}/'.format(pmt)
data=np.load(path+'/raw_wf.npz'.format(pmt))
init=data['init']
data=np.load(path+'recon_wf.npz')
rec=data['rec']
WF=data['WF']
recon_WF=data['recon_WF']
chi2_cut=1.7e6
area=-np.sum(spe[init:])
left=30
right=70

rng=np.nonzero(np.logical_and(rec['chi2']<chi2_cut, rec['init']>=70))
spec=np.mean(rec[rng]['h'], axis=0)
rng_BG=np.nonzero(np.logical_and(rec_BG['chi2']<chi2_cut, rec_BG['init']>=70))
spec_BG=np.mean(rec_BG[rng_BG]['h'], axis=0)


x=np.arange(1000)/5
fig=plt.figure(figsize=(20,10))
ax=fig.add_subplot(111)
ax.plot(x, WF, 'k.-')
ax.plot(x, recon_WF, 'r.-')
ax.legend()

fig=plt.figure(figsize=(20,10))
ax=fig.add_subplot(221)
ax.plot(x, spec_BG, 'k.', label='spec_BG')
ax.plot(x, spec, 'r.', label='spec')
ax.legend()

ax=fig.add_subplot(222)
ax.hist(rec['chi2'], bins=100, label='chi2', range=[0,2e7], histtype='step')
ax.hist(rec_BG['chi2'], bins=100, label='chi2_BG', range=[0,2e7], histtype='step')
ax.axvline(chi2_cut, ymin=0, ymax=1, color='k')
ax.set_yscale('log')
ax.legend()

ax=fig.add_subplot(223)
ax.plot(areas, h_area, '.', label='area')
ax.plot(areas, func(areas, *p_area), 'r.-')
ax.set_yscale('log')
ax.legend()

ax=fig.add_subplot(224)
# ax.hist(-rec[rng]['area']/area+np.sum(rec[rng]['h'], axis=1), bins=np.arange(750)-50.5, label='diff', histtype='step')
ax.hist(rec[rng]['area']/area, bins=np.arange(100)-0.5, label='pe by area', histtype='step')
ax.hist(rec_BG[rng_BG]['area']/area, bins=np.arange(100)-0.5, label='pe by area - BG', histtype='step')
h_BG, bins, pat = ax.hist(np.sum(rec_BG[rng_BG]['h'], axis=1), bins=np.arange(100)-0.5, label='BG recon pe', histtype='step')
h, bins, pat = ax.hist(np.sum(rec[rng]['h'], axis=1), bins=np.arange(100)-0.5, label='recon pe', histtype='step')
print(rec[rng][np.sum(rec[rng]['h'], axis=1)<20]['id'])
ax.set_yscale('log')
x=0.5*(bins[1:]+bins[:-1])
rngs=np.nonzero(np.logical_and(x>left, x<right))[0]
# p0=[np.sum(h)/2, 50, 0.8, 0.02]
# p0=[322366.14973332774, 35.28371786085655, 0.8028204609854822, 0.04271088455414705]
# bounds=[[0, 35, 0, 0], [1e9, 1000, 3, 1]]
# p, cov=curve_fit(fit_spectra, x[rngs], h[rngs], p0=p0, bounds=bounds)
# print(p)
# ax.plot(x, fit_spectra(x, *p), 'ro')
ax.legend()
rng=np.nonzero(np.logical_and(np.logical_and(rec['chi2']<chi2_cut, rec['init']>=init), np.logical_and(np.sum(rec['h'], axis=1)>left, np.sum(rec['h'], axis=1)<right)))[0]
H=np.zeros((10, 1000))
H_BG=np.zeros((10, 1000))
# for i in range(1000):
#     print(i)
#     H[:,i]=np.histogram(rec[rng]['h'][:,i], bins=np.arange(11)-0.5)[0]
#     H_BG[:,i]=np.histogram(rec_BG[rng]['h'][:,i], bins=np.arange(11)-0.5)[0]
# np.savez(path+'H', H=H, H_BG=H_BG, N_events=len(rng), Ns=x[rngs], h_spec=h[rngs], h_spec_BG=h_BG[rngs])
plt.show()
