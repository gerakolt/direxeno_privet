import matplotlib.pyplot as plt
import os, sys
import numpy as np
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from fun import do_smd, do_dif, find_peaks, analize_peaks
########## This is a develop branch#########

def func(x, a,b,c):
    return a*np.exp(-0.5*(x-b)**2/c**2)

l=125
r=340
blw_cut=20
left=85
right=107
height_cut=25
d_cut=4
init_cut=100

pmt=1
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
np.savez(path+'cuts{}.npz'.format(pmt), l=l, r=r, blw_cut=blw_cut, left=left, right=right, height_cut=height_cut, d_cut=d_cut, init_cut=init_cut)
peaks=np.load(path+'Peaks{}.npz'.format(pmt))['rec']
try:
    SPE=np.load(path+'SPEs{}.npz'.format(pmt))['SPE']
    spe=np.mean(SPE, axis=0)
    [area_wind, peak]=next(find_peaks(np.array(spe), np.sqrt(np.mean(spe[:init_cut]**2)), 0, l, r))
except:
    spe=np.zeros(1000)
    SPE=[]
    print('Cant open SPEs{}'.format(pmt))

blw=peaks[np.unique(peaks['id'], return_index=True)[1]]['blw']
fig=plt.figure()
fig.suptitle('PMT{}'.format(pmt))
fig.tight_layout()

ax=fig.add_subplot(111)
ax.hist(blw, bins=100, range=[0,100], label='blw')
ax.axvline(x=blw_cut, ymin=0, ymax=1, color='k')
ax.legend()
ax.set_yscale('log')

peaks=peaks[(peaks['blw']<blw_cut)]

fig=plt.figure()
fig.suptitle('PMT{}'.format(pmt))
fig.tight_layout()
ax=fig.add_subplot(211)
ax.hist2d(peaks['t'], peaks['h'], bins=[100,100], range=[[0,800], [0,500]], norm=LogNorm())
ax.axvline(x=left, ymin=0, ymax=1, color='k')
ax.axvline(x=right, ymin=0, ymax=1, color='k')
ax.axhline(y=height_cut, xmin=0, xmax=1, color='k')
ax.legend()

ax=fig.add_subplot(212)
ax.hist2d(peaks['d'], peaks['h'], bins=[40,100], range=[[0,50], [0,500]], norm=LogNorm())
ax.axvline(x=d_cut, ymin=0, ymax=1, color='k')


peaks=peaks[(peaks['t']<right) & (peaks['t']>left) & (peaks['d']>d_cut)]

fig=plt.figure()
fig.suptitle('PMT{}'.format(pmt))
fig.tight_layout()

ax=fig.add_subplot(111)
ax.hist(peaks['init'], bins=100, label='init')
ax.axvline(x=init_cut, ymin=0, ymax=1, color='k')
ax.legend()


fig=plt.figure()
fig.suptitle('PMT{}'.format(pmt))
fig.tight_layout()

ax=fig.add_subplot(311)
h, bins,pa=ax.hist(peaks['area_peak'], bins=100, range=[0,6000], label='peak area', histtype='step')
ax.hist(peaks['area_wind'], bins=100, range=[-5000,5000], label='area wind', histtype='step')
x=0.5*(bins[1:]+bins[:-1])
area_left=320
area_right=2000
rng=np.nonzero(np.logical_and(x>area_left, x<area_right))
p0=[np.amax(h[rng]), x[rng][np.argmax(h[rng])], 0.5*x[rng][np.argmax(h[rng])]]
try:
    p, cov=curve_fit(func, x[rng], h[rng], p0=p0)
    ax.plot(x[rng], func(x[rng], *p), 'r--', label=r'$\bar{A}=$'+'{:3.2f}'.format(p[1])+r'$\pm$'+'{:3.2f}'.format(p[2]))
    np.savez(path+'area{}'.format(pmt), bins_area=bins, h_area=h, Mpe=p[1], Spe=p[2], left=area_left, right=area_right, mean_spe_area=peak.area)
    ax.legend()
except:
    np.savez(path+'area{}'.format(pmt), bins_area=bins, h_area=h, left=area_left, right=area_right)
    print('No area fit')

ax.set_yscale('log')
ax.legend()

ax=fig.add_subplot(312)
ax.hist(peaks['h'], bins=100, range=[0,200], label='peak area', histtype='step')
ax.axvline(height_cut, ymin=0, ymax=1)

ax=fig.add_subplot(313)
ax.hist2d(peaks['area_peak'], peaks['h'], bins=[100,100], range=[[0,5000], [0,500]], norm=LogNorm())

x=np.arange(1000)
fig=plt.figure()
fig.suptitle('PMT{} - SPE'.format(pmt))
fig.tight_layout()
ax=fig.add_subplot(111)
ax.plot(x, spe, 'k.')
I=np.zeros(len(spe))
for i in range(len(I)):
    I[i]=np.sum(spe[:i])
ax.plot(x, I/100, 'r-')
try:
    ax.fill_between(x[peak.init:peak.fin], y1=spe[peak.init:peak.fin], y2=0, color='y', label='area={}'.format(peak.area))
except:
    temp=1
ax.legend()
plt.show()
