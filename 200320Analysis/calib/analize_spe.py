import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

pmt=8
rise_time_cut=5
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
data=np.load(path+'raw_wf.npz'.format(pmt))
left=data['left']
right=data['right']
init=data['init']
data=np.load(path+'cuts.npz'.format(pmt))
blw_cut=data['blw_cut']
height_cut=data['height_cut']

data=np.load(path+'spe.npz')
Sig_trig=data['Sig_trig']
Sig_init10=data['Sig_init10']
Sig_init10=np.roll(Sig_init10, np.argmin(Sig_trig)-np.argmin(Sig_init10))
BL=data['BL']
rec=data['rec']
height=rec['height']
height_r=rec['height_r']
area=rec['area']
t=rec['t']
rise_time=rec['rise_time']


def func(x, a,b,c,m_bl,s_bl,m,s):
    return a*np.exp(-0.5*(x-m_bl)**2/s_bl**2)+b*np.exp(-0.5*(x-(m_bl+m))**2/(s_bl**2+s**2))+c*np.exp(-0.5*(x-(m_bl+2*m))**2/(s_bl**2+2*s**2))

fig=plt.figure()
ax=fig.add_subplot(221)
h,bins,pa=ax.hist(area, bins=100, label='area')
x=0.5*(bins[1:]+bins[:-1])
rng=np.nonzero(np.logical_and(x>-1000, x<4000))
p0=[11000, 1200, 600, 0, 200, 2250, 1000]
p, cov=curve_fit(func, x[rng], h[rng], p0=p0)
ax.plot(x[rng], func(x[rng], *p), 'r--')
ax.set_yscale('log')
ax.legend()

Sig_trig=-Sig_trig*(p[3]+p[-2])/np.sum(Sig_trig[left:right])
Sig_init10=-Sig_init10*(p[3]+p[-2])/np.sum(Sig_init10[left:right])

x=np.arange(1000)/5
ax=fig.add_subplot(222)
ax.plot(x, Sig_trig, 'k.-', label='Sig trigger')
ax.plot(x, Sig_init10, 'r.-', label='Sig init10')
ax.plot(x, BL, 'g.-', label='Base line')
ax.fill_between(x[left:right], y1=np.amin(Sig_init10), y2=0, color='y', alpha=0.5)
ax.fill_between(x[:init], y1=0.5*np.amin(Sig_init10), y2=0, color='b', alpha=0.5)
ax.legend()

ax=fig.add_subplot(223)
ax.hist(height, bins=100, label='height', histtype='step')
ax.hist(height_r, bins=100, label='height_r', histtype='step')
ax.axvline(height_cut, ymin=0, ymax=1, color='k')
ax.set_yscale('log')
ax.legend()

ax=fig.add_subplot(224)
ax.hist(rise_time, bins=50, label='rise_time')
ax.axvline(rise_time_cut, ymin=0, ymax=1, color='k')
ax.set_yscale('log')
ax.legend()

np.savez(path+'spe', Sig_trig=Sig_trig, Sig_init10=Sig_init10, BL=BL, rec=rec, rise_time_cut=rise_time_cut, height_cut=height_cut)
plt.show()
