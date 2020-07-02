import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors

def func(x, a_pad, a, m_pad, s_pad, m, s):
    return a_pad*np.exp(-0.5*(x-m_pad)**2/s_pad**2)+a*np.exp(-0.5*(x-m)**2/s**2)

pmt=0
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
data=np.load(path+'cuts.npz')
blw_cut=data['blw_cut']
height_cut=data['height_cut']
# height_cut=15
left=data['left']
right=data['right']
BL=np.load(path+'BL.npz')['BL']
dh3_cut=0.5
spk_cut=64

data=np.load(path+'spe.npz')
rec=data['rec']


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6))=plt.subplots(3,2)

fig.suptitle('PMT{}'.format(pmt))

ax1.hist(rec['blw'], bins=100, range=[0,60], label='blw')
ax1.axvline(blw_cut, ymin=0, ymax=1)
ax1.set_yscale('log')
ax1.legend()

rec=rec[rec['blw']<blw_cut]

ax2.hist2d(rec['height'], rec['dh3'], bins=[100,100], range=[[0,200], [0,1.5]], norm=mcolors.PowerNorm(0.3))
ax2.axhline(dh3_cut, xmin=0, xmax=1, color='k')
ax2.axvline(height_cut, ymin=0, ymax=1, color='k')
ax2.axvline(spk_cut, ymin=0, ymax=1, color='k')


rec=rec[rec['dh3']<dh3_cut]

h_heights, bins, pat=ax3.hist(rec['height'], bins=100, label='height', range=[0,250], histtype='step')
heights=0.5*(bins[1:]+bins[:-1])
ax3.axvline(height_cut, ymin=0, ymax=1)
ax3.set_yscale('log')
ax3.legend()

h_area, bins, pat=ax4.hist(rec['area'], bins=100, label='area', range=[-1000, 4000], histtype='step')
areas=0.5*(bins[:-1]+bins[1:])
rng=np.nonzero(np.logical_or(np.logical_and(areas>700, areas<4000), False))[0]
ax4.set_yscale('log')
p=[4566.3385835,   401.26508103,   0,  404.40993119,  1500,
  700]


bounds_up=[1e5, 1e5, 1000, 1000, 20000, 1e5]
bounds_dn=[0,0,-1000,0,1000, 0]

p, cov=curve_fit(func, areas[rng], h_area[rng], p0=p, bounds=[bounds_dn, bounds_up])
print(np.array(p))
ax4.plot(areas[rng], func(areas[rng], *p), '.')
ax4.set_ylim(1,np.amax(h_area)+500)

x=np.arange(1000)/5
spe=np.sum(rec['spe'], axis=0)
maxi=np.argmin(spe)
area=-(np.sum(spe[maxi-100:maxi+200])+np.sum(spe[maxi-50:maxi+150])+np.sum(spe[maxi-100:maxi+150])+np.sum(spe[maxi-50:maxi+200]))/4
spe=p[-2]*spe/area
ax5.plot(x, spe, 'r.', label='mean SPE')
ax5.fill_between(x[maxi-100:maxi+200], y1=np.amin(spe), y2=0)
ax5.fill_between(x[maxi-50:maxi+150], y1=np.amin(spe), y2=0)
# ax5.plot(x, BL, 'y.')
ax5.plot(x, x*0, 'k--')
ax3.axvline(-np.amin(spe), ymin=0, ymax=1)
ax5.legend()


# np.savez(path+'areas', areas=areas/p[-2], H_areas=h_area, rng_area=rng, spe=spe)
# np.savez(path+'cuts', blw_cut=blw_cut, height_cut=height_cut, left=left, right=right, dh3_cut=dh3_cut, spk_cut=spk_cut)

plt.show()
