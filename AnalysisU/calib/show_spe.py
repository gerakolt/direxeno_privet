import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors


pmt=1
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
data=np.load(path+'cuts.npz')
blw_cut=data['blw_cut']
height_cut=data['height_cut']
left=data['left']
right=data['right']
# BL=np.load(path+'BL.npz')['BL']
BL=np.zeros(1000)
dh3_cut=0.5
spk_cut=64

data=np.load(path+'spe.npz')
rec0=data['rec']
rec0=rec0[rec0['area']!=0]
rec=data['rec']
rec=rec[np.logical_and(rec['maxi']>left, rec['maxi']<right)]


rec=rec[rec['blw']<blw_cut]
rec0=rec0[rec0['blw']<blw_cut]


rec=rec[rec['dh3']<dh3_cut]
rec0=rec0[rec0['dh3']<dh3_cut]

# print(rec[np.logical_and(rec['height']>30, rec['height']<80)]['id'][:20])

rec2=rec[np.logical_and(rec['height']>30, rec['height']<100)][:50]

rec1=rec[rec['height']>height_cut]

plt.figure(figsize=(20,10))
range=[-2000, 6000]
plt.hist(rec0['area'], bins=100, label='area', range=range, histtype='step')
h_area, bins, pat=plt.hist(rec0['area'], bins=100, label='area', range=range, histtype='step')
areas=0.5*(bins[1:]+bins[:-1])
rng=np.nonzero(np.logical_or(np.logical_and(areas>-500, areas<4000), False))[0]
plt.yscale('log')
def func(x, a_pad, m_pad, s_pad, a, m, s):
    return a_pad*np.exp(-0.5*(x-m_pad)**2/s_pad**2)+a*np.exp(-0.5*(x-m)**2/s**2)
p=[2000, 0, 500, 500, 2000, 1000]


bounds_up=[1e5, 1000, 1e5, 1e5, 4000, 1e5]
bounds_dn=[0, -1e3, 0, 0, 1850, 0]

p, cov=curve_fit(func, areas[rng], h_area[rng], p0=p, bounds=[bounds_dn, bounds_up])
print(np.array(p))
plt.plot(areas[rng], func(areas[rng], *p), '.')
# ax4.set_ylim(1,np.amax(h_area)+500)

plt.figure(figsize=(20,10))
x=np.arange(100,500)/5
spe=np.sum(rec['spe'], axis=0)[100:500]
maxi=np.argmin(spe)
# spe[maxi+200:]=0
area=-(np.sum(spe[maxi-100:maxi+200])+np.sum(spe[maxi-50:maxi+150])+np.sum(spe[maxi-100:maxi+150])+np.sum(spe[maxi-50:maxi+200]))/4
spe=p[-2]*spe/area
# plt.fill_between(x[maxi-100:maxi+200], y1=-100, y2=0, alpha=0.2, hatch='|')
# plt.fill_between(x[maxi-50:maxi+150], y1=-100, y2=0, alpha=0.2, hatch='/')
plt.fill_between(x[maxi-100:maxi+150], y1=-100, y2=0, alpha=0.2, hatch='|', label='Summation Window 1')
plt.fill_between(x[maxi-50:maxi+200], y1=-100, y2=0, alpha=0.2, hatch='/', label='Summation Window 2')
plt.plot(x, x*0, 'k--')

for i in np.arange(50):
    if i==0:
        plt.plot(x, rec2['spe'][i][100:500], '--', alpha=0.65, label='Example of SPE Signals')
    else:
        plt.plot(x, rec2['spe'][i][100:500], '--', alpha=1)
plt.plot(x, spe, 'r.-', label='SPE Template', linewidth=7)
plt.legend(fontsize=35, loc='lower right')
plt.xlabel('Time [ns]', fontsize=25)
plt.ylabel('Digitizer Counts', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20)

plt.figure(figsize=(20,10))
h_heights, bins, pat=plt.hist(rec0['height'], bins=50, label='height', range=[0,250], histtype='step')
plt.axvline(-np.amin(spe), 0, 1)
plt.yscale('log')
plt.legend()

plt.show()
