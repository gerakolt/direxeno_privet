import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors

pmt=19
rise_time_cut=5
area_l=1000
area_r=3000
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
data=np.load(path+'raw_wf.npz'.format(pmt))
left=data['left']
right=data['right']
init=data['init']
data=np.load(path+'cuts.npz'.format(pmt))
blw_cut=data['blw_cut']
height_cut=data['height_cut']

data=np.load(path+'spe.npz')
WF=data['WF']
rec=data['rec']
height=rec['height']
area=rec['area']
t=rec['t']
rise_time=rec['rise_time']
spe_rng=np.nonzero(np.logical_and(height>height_cut, np.logical_and(rise_time>rise_time_cut, t>init)))[0]
spe=np.mean(rec[spe_rng]['spe'], axis=0)

def func(x, a,b,c,m_bl,s_bl,m,s):
    return a*np.exp(-0.5*(x-m_bl)**2/s_bl**2)+b*np.exp(-0.5*(x-(m_bl+m))**2/(s_bl**2+s**2))+c*np.exp(-0.5*(x-(m_bl+2*m))**2/(s_bl**2+2*s**2))

# plt.figure()
# plt.hist2d(area[rise_time>rise_time_cut], height[rise_time>rise_time_cut], bins=100, range=[[-1500, 10000],[0, 200]], norm=mcolors.PowerNorm(0.3))
# plt.axhline(height_cut, xmin=0, xmax=1, color='k')
# plt.show()

fig=plt.figure()
ax=fig.add_subplot(221)
ax.hist(area[rec['height']>height_cut], bins=50, label='area (h>height cut)', range=[-1500, 10000], histtype='step')
h,bins,pa=ax.hist(area, bins=50, label='area', range=[-1500, 10000], histtype='step')
xa=0.5*(bins[1:]+bins[:-1])
rng=np.nonzero(np.logical_and(xa>-1000, xa<10000))
p=[11000, 1200, 600, 0, 200, 2250, 1000]
bounds=[[0,0,0, -1000, 1, 1000, 1],[1e7,1e7,1e7, 1000, 1000, 4000, 4000]]
p, cov=curve_fit(func, xa[rng], h[rng], p0=p, bounds=bounds)
print(p)
ax.plot(xa[rng], func(xa[rng], *p), 'r--')
ax.set_ylim(1,np.amax(h))

p1=np.array(p)
p1[0]=0
p1[2]=0
ax.plot(xa[rng], func(xa[rng], *p1), 'g--')

p1=np.array(p)
p1[0]=0
p1[1]=0
ax.plot(xa[rng], func(xa[rng], *p1), 'g--')
ax.set_yscale('log')
ax.legend()

spe-=np.median(spe[:180])
spe=-spe*(p[3]+p[-2])/np.sum(spe[120:304])

x=np.arange(1000)
ax=fig.add_subplot(222)
ax.plot(x, spe, 'r.-', label='Sig init10')
ax.plot(x, WF, 'k.-', label='Sig trig', alpha=0.3)
ax.fill_between(x[left:right], y1=np.amin(WF), y2=0, alpha=0.3)
ax.fill_between(x[:init], y1=np.amin(WF), y2=0, alpha=0.3)

ax.axhline(0, xmin=0.02, xmax=0.98, color='k')
ax.legend()

ax=fig.add_subplot(223)
ax.hist(height, bins=100, range=[0,100], label='height', histtype='step')
ax.axvline(height_cut, ymin=0, ymax=1, color='k')
ax.axvline(-np.amin(spe), ymin=0, ymax=1, color='r')

ax.set_yscale('log')
ax.legend()

ax=fig.add_subplot(224)
ax.hist(rise_time, bins=50, label='rise_time')
ax.axvline(rise_time_cut, ymin=0, ymax=1, color='k')
ax.set_yscale('log')
ax.legend()

np.savez(path+'spe', spe=spe, WF=WF, rec=rec, rise_time_cut=rise_time_cut, height_cut=height_cut, areas=xa[rng], h_area=h[rng], p_area=p)
plt.show()
