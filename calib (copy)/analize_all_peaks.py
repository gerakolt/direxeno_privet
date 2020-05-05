import matplotlib.pyplot as plt
import os, sys
import numpy as np
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from fun import do_smd, do_dif, find_peaks, analize_peaks
########## This is a develop branch#########
l=150
r=325
blw_cut=15
left=70
right=115
height_cut=35
d_cut=4

pmts=np.array([0,1,2,3,4,5,6,7,8,9,10,11,17,14,15,16,18,19])
pmt=1
path='/home/gerak/Desktop/DireXeno/190803/pulser/'
Peaks=np.load(path+'AllPeaks.npz')['rec']
SPEpeaks=[]
try:
    SPE=np.load(path+'SPEs{}.npz'.format(pmt))['SPE']
    # spe=np.sum(SPE, axis=0)
    spe=np.mean(SPE, axis=0)
    # spe_cut=np.sum(SPE[np.nonzero(np.amin(SPE, axis=1)<-height_cut)[0]], axis=0)
    # spe=spe-np.median(spe[:150])
    # spe_cut=spe_cut-np.median(spe_cut[:150])
    # blw=np.sqrt(np.mean(spe_cut[:150]**2))
except:
    spe=np.zeros(1000)
    SPE=[]
    print('Cant open SPEs{}'.format(pmt))

peaks=Peaks[Peaks['pmt']==pmt]
# peaks=Peaks
peaks_tcut=peaks[(peaks['blw']<blw_cut) & (peaks['t']<right) & (peaks['t']>left)]
peaks_tdcut=peaks[(peaks['blw']<blw_cut) & (peaks['t']<right) & (peaks['t']>left) & (peaks['d']>d_cut)]
peaks_off=peaks[(peaks['blw']<blw_cut) & (peaks['t']<right+300) & (peaks['t']>left+300) & (peaks['d']>d_cut)]
blw=peaks[np.unique(peaks['id'], return_index=True)[1]]['blw']

plt.figure()
plt.plot(spe, 'k.-', label='Mean SPE after time window cut and deriv cut')
plt.legend(fontsize=25)
plt.axhline(y=0, color='r')
plt.fill_between(np.arange(1000)[l:r], y1=spe[l:r], y2=0, color='y', alpha=0.5)
# plt.xlabel('derivative of rise part', fontsize=25)
# plt.ylabel('Peak height', fontsize=25)
plt.show()

fig=plt.figure()
fig.suptitle('PMT{}'.format(pmt))
fig.tight_layout()

ax=fig.add_subplot(421)
ax.hist(blw, bins=100, range=[0,100], label='blw')
ax.axvline(x=blw_cut, ymin=0, ymax=1, color='k')
ax.legend()
ax.set_yscale('log')

ax=fig.add_subplot(422)
ax.hist(peaks['d'], bins=np.arange(50)-0.5, label='d - all', histtype='step')
ax.hist(peaks_tcut['d'], bins=np.arange(50)-0.5, label='d - t cut', histtype='step')
ax.axvline(x=d_cut, ymin=0, ymax=1, color='k')
ax.legend()

ax=fig.add_subplot(423)
ax.hist(peaks['t'], bins=100, range=[0,800], label='t -all', histtype='step')
ax.hist(peaks_tdcut['t'], bins=100, range=[left,right], label='t - in t wind t&d cut', histtype='step')
ax.axvline(x=left, ymin=0, ymax=1, color='k')
ax.axvline(x=right, ymin=0, ymax=1, color='k')
ax.axvline(x=left+300, ymin=0, ymax=1, color='r')
ax.axvline(x=right+300, ymin=0, ymax=1, color='r')
ax.legend()

ax=fig.add_subplot(424)
ax.hist2d(peaks['t'], peaks['h'], bins=[100,100], range=[[0,800], [0,500]], norm=LogNorm())
ax.axvline(x=left, ymin=0, ymax=1, color='k')
ax.axvline(x=right, ymin=0, ymax=1, color='k')
ax.axvline(x=left+300, ymin=0, ymax=1, color='r')
ax.axvline(x=right+300, ymin=0, ymax=1, color='r')
ax.axhline(y=height_cut, xmin=0, xmax=1, color='k')
ax.legend()

ax=fig.add_subplot(425)
def func(x, a,b,c):
    return a*np.exp(-0.5*(x-b)**2/c**2)
ax.hist(peaks['area_peak'], bins=100, range=[0,5000], label='area', histtype='step')
h, bins,pa=ax.hist(peaks_tdcut['area_peak'], bins=100, range=[0,5000], label='area - t&d cut', histtype='step')
ax.hist(peaks_tdcut['area_wind'], bins=100, range=[-5000,5000], label='area wind - t&d cut', histtype='step')
h_area=np.array(h)
x=0.5*(bins[1:]+bins[:-1])
x_area=np.array(x)
rng_area=np.nonzero(np.logical_and(x>320, x<1500))
p0=[np.amax(h[rng_area]), x[rng_area][np.argmax(h[rng_area])], 0.5*x[rng_area][np.argmax(h[rng_area])]]
try:
    p_area, cov=curve_fit(func, x[rng_area], h[rng_area], p0=p0)
    ax.plot(x[rng_area], func(x[rng_area], *p_area), 'r--', label=r'$\bar{A}=$'+'{:3.2f}'.format(p_area[1]))
    factor=-np.sum(spe[l:r])/p_area[1]
    ax.legend()
except:
    print('No area fit')
ax.set_yscale('log')
ax.legend()

ax=fig.add_subplot(426)
ax.hist(peaks['h'], bins=100, range=[0,200], label='height', histtype='step')
ax.hist(peaks_off['h'], bins=100, range=[0,200], label='height (non peaks) - d&t cut', histtype='step')
h,bins, pa=ax.hist(peaks_tdcut['h'], bins=100, range=[0,200], label='height - t&d cut', histtype='step')
x=0.5*(bins[1:]+bins[:-1])
rng=np.nonzero(np.logical_and(x>25, x<70))
p0=[np.amax(h[rng_area]), x[rng_area][np.argmax(h[rng_area])], 0.5*x[rng_area][np.argmax(h[rng_area])]]
try:
    p_h, cov=curve_fit(func, x[rng], h[rng], p0=p0)
    ax.plot(x[rng], func(x[rng], *p_h), 'r--', label=r'$\bar{A}=$'+'{:3.2f}'.format(p_h[1]))
    ax.legend()
except:
    print('No height fit')
ax.axvline(x=height_cut, ymin=0, ymax=1, color='k')
ax.set_yscale('log')
ax.legend()


ax=fig.add_subplot(4,1,4)

try:
    h,bins=np.histogram(l+peaks_tdcut['dt'], bins=100)
    x=0.5*(bins[1:]+bins[:-1])
    # factor=-np.sum(spe_cut[l:r])/p_area[1]
    ax.plot(spe/factor, 'g.')
    dif=1e9
    # for i in range(l,300):
    #     a=-np.sum(spe_cut[l:i]/factor)
    #     if np.abs(a-p_area[1])<dif:
    #         r=i
    #         dif=np.abs(a-p_area[1])
    # ax.plot(x, h/np.amax(h)*np.amin(spe_cut/factor), 'b.-')
    ax.fill_between(np.linspace(l, r, 100), y1=np.amin(spe/factor), y2=0, alpha=0.2, label='{} PEs'.format(factor))
except:
    factor=1
    print('Cant plot spe_cut')
ax.axhline(y=0, xmin=0, xmax=1, color='r')
ax.legend()



try:
    np.savez(path+'SPEs{}'.format(pmt), SPE=SPE, factor=factor, zeros=[999], Spe=p_area[2]/p_area[1], mean_height=p_h[1])
    print('Saved factor')
except:
    print('didnt save SPE and factor and Spe')
plt.show()



def make_P(Spe, ns):
    P=np.zeros((ns[-1]+10, ns[-1]+10))
    P[0,0]=1
    for i in range(len(P[:,0])):
        r=np.linspace(i-0.5,i+0.5,1000)
        dr=r[1]-r[0]
        P[i,1]=dr*np.sum(np.exp(-0.5*(r-1)**2/Spe**2))/(np.sqrt(2*np.pi)*Spe)
    for j in range(2, len(P[0,:])):
        for i in range(len(P[:,0])):
            P[i,j]=np.sum(P[:i+1,1]*np.flip(P[:i+1,j-1]))
    return P[ns,:]

ns=[0,1,2,3,4,5,6]
P=make_P(p_area[2]/p_area[1], ns)

fig=plt.figure()
ax=fig.add_subplot(221)
ax.plot(x_area/p_area[1], h_area, 'k.')
ax.plot(x_area[rng_area]/p_area[1], func(x_area[rng_area]/p_area[1], *[p_area[0], 1, p_area[2]/p_area[1]]), 'r--')
ax.plot(ns, P[:,1]/np.amax(P[:,1])*p_area[0], 'go', label='P1')
ax.legend()

ax=fig.add_subplot(222)
ax.plot(x_area/p_area[1], h_area, 'k.')
ax.plot(x_area[rng_area]/p_area[1], func(x_area[rng_area]/p_area[1], *[p_area[0], 1, p_area[2]/p_area[1]]), 'r--')
ax.plot(ns, P[:,2]/np.amax(P[:,2])*p_area[0], 'go', label='P2')
ax.legend()

ax=fig.add_subplot(223)
ax.plot(x_area/p_area[1], h_area, 'k.')
ax.plot(x_area[rng_area]/p_area[1], func(x_area[rng_area]/p_area[1], *[p_area[0], 1, p_area[2]/p_area[1]]), 'r--')
ax.plot(ns, P[:,3]/np.amax(P[:,3])*p_area[0], 'go', label='P3')
ax.legend()

ax=fig.add_subplot(224)
ax.plot(x_area/p_area[1], h_area, 'k.')
ax.plot(x_area[rng_area]/p_area[1], func(x_area[rng_area]/p_area[1], *[p_area[0], 1, p_area[2]/p_area[1]]), 'r--')
ax.plot(ns, P[:,4]/np.amax(P[:,4])*p_area[0], 'go', label='P4')
ax.legend()
plt.show()
