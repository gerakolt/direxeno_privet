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

dt=29710
dt_BG=48524

pmt=8
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
data=np.load(path+'spe.npz')
spe=data['spe']
areas=data['areas']
h_area=data['h_area']
p_area=data['p_area']

path='/home/gerak/Desktop/DireXeno/190803/BG/PMT{}/'.format(pmt)
data=np.load(path+'/raw_wf.npz'.format(pmt))
init=data['init']
data=np.load(path+'recon_wf.npz')
rec_BG=data['rec']
chi2_cut=1.7e6
rng_BG=np.nonzero(np.logical_and(rec_BG['chi2']<chi2_cut, rec_BG['init']>=70))

path='/home/gerak/Desktop/DireXeno/190803/Co57/PMT{}/'.format(pmt)
data=np.load(path+'recon_wf.npz')
rec=data['rec']
chi2_cut=1.7e6
left=30
right=90
rng=np.nonzero(np.logical_and(rec['chi2']<chi2_cut, rec['init']>=70))


fig=plt.figure(figsize=(20,10))
ax=fig.add_subplot(111)
h_BG, bins=np.histogram(np.sum(rec_BG[rng_BG]['h'], axis=1), bins=np.arange(200)-0.5, density=True)
h, bins= np.histogram(np.sum(rec[rng]['h'], axis=1), bins=np.arange(200)-0.5, density=True)
x=0.5*(bins[1:]+bins[:-1])
plt.plot(x, h, 'r.', label='recon_pes')
plt.plot(x, h_BG*dt/dt_BG, 'k.', label='BG recon_pes')
plt.step(x, h-h_BG*dt/dt_BG)


# x=0.5*(bins[1:]+bins[:-1])
# rng=np.nonzero(np.logical_and(x>left, x<right))[0]
# p0=[np.sum(h)/2, 50, 0.8, 0.02]
# p0=[322366.14973332774, 35.28371786085655, 0.8028204609854822, 0.04271088455414705]
# bounds=[[0, 35, 0, 0], [1e9, 1000, 3, 1]]
# p, cov=curve_fit(fit_spectra, x[rng], h[rng], p0=p0, bounds=bounds)
# print(p)
# ax.plot(x, fit_spectra(x, *p), 'ro')

ax.legend()

plt.show()
