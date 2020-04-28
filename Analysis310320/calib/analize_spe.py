import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# def func(x, a,b,c, m_pad, s_pad, m,s):
#     return a*np.exp(-0.5*(x-m_pad)**2/s_pad**2)+b*np.exp(-0.5*(x-(m_pad+m))**2/(s_pad**2+s**2))+c*np.exp(-0.5*(x-(m_pad+2*m))**2/(s_pad**2+2*s**2))

def func(X, a,b,c,d, m_pad, s_pad, m,s):
    y=np.zeros(len(X))
    dx=X[1]-X[0]
    for i, x in enumerate(X):
        r=np.linspace(x-0.5*dx,x+0.5*dx,100)
        dr=r[1]-r[0]
        y[i]=np.sum(a*np.exp(-0.5*(r-m_pad)**2/s_pad**2)+b*np.exp(-0.5*(r-(m_pad+m))**2/(s_pad**2+s**2))+c*np.exp(-0.5*(r-(m_pad+2*m))**2/(s_pad**2+2*s**2))+d*np.exp(-0.5*(r-(m_pad+3*m))**2/(s_pad**2+3*s**2)))*dr
    return y

def func0(X, a,t,s):
    y=np.zeros(len(X))
    dx=X[1]-X[0]
    for i, x in enumerate(X):
        r=np.linspace(x-0.5*dx,x+0.5*dx,100)
        dr=r[1]-r[0]
        y[i]=np.sum(a*np.exp(-0.5*(np.log(r/t))**2/s**2))*dr
    return y

pmt=5
path='/home/gerak/Desktop/DireXeno/190803/NEWpulser/NEWPMT{}/'.format(pmt)
data=np.load(path+'cuts.npz')
blw_cut=data['blw_cut']
height_cut=data['height_cut']
left=data['left']
right=data['right']
BL=np.load(path+'BL.npz')['BL']
rise_time_cut=8

data=np.load(path+'spe.npz')
noise=data['rec']
rec=data['rec']
WF=data['WF']


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6))=plt.subplots(3,2)

fig.suptitle('PMT{}'.format(pmt))

ax1.hist(rec['blw'], bins=100, range=[0,60], label='blw')
ax1.axvline(blw_cut, ymin=0, ymax=1)
ax1.set_yscale('log')
ax1.legend()

rec=rec[rec['blw']<blw_cut]
noise=noise[noise['blw']<blw_cut]

ax2.hist(rec['rise_time'], bins=np.arange(30)-0.5, label='rise time')
ax2.hist(noise['noise_rise_time'], bins=np.arange(30)-0.5, label='noise rise time')
ax2.axvline(rise_time_cut, ymin=0, ymax=1)
ax2.legend()

rec=rec[rec['rise_time']>rise_time_cut]
noise=noise[noise['noise_rise_time']>rise_time_cut]


ax3.hist(noise['noise'], bins=100, label='noise', range=[0,2500], histtype='step')
h_heights, bins, pat=ax3.hist(rec['height'], bins=100, label='height', range=[0,2500], histtype='step')
heights=0.5*(bins[1:]+bins[:-1])
ax3.axvline(height_cut, ymin=0, ymax=1)
ax3.set_yscale('log')
# rng=np.nonzero(heights<height_cut)[0]
# p=[np.amax(h_heights), np.argmax(h_heights), 0.5*np.argmax(h_heights)]
# p, cov=curve_fit(func0, heights[rng], h_heights[rng], p0=p)
# ax3.plot(heights, func0(heights, *p), 'r-', label='{}, {}, {}'.format(p[0], p[1], p[2]))
# ax3.set_ylim(1, np.amax(h_heights))
ax3.legend()

ax4.hist(rec[rec['height']>height_cut]['area'], bins=100, label='area, h>h_cut', range=[-2000, 20000], histtype='step')
h_area, bins, pat=ax4.hist(rec['area'], bins=100, label='area', range=[-3000, 20000], histtype='step')
areas=0.5*(bins[:-1]+bins[1:])
rng=np.nonzero(np.logical_and(areas>-1700, areas<17000))[0]
ax4.set_yscale('log')
p=[55, 0.7, 0, 0,
 24, 4.5e+02, 5000, 2500]

# p, cov=curve_fit(func, areas[rng], h_area[rng], p0=p, method='dogbox')

ax4.plot(areas[rng], func(areas[rng], *p), '.-')
# p1=np.array(p)
# p2=np.array(p)
# p3=np.array(p)
# p1[0]=0
# p1[2]=0
# p1[3]=0
#
# p2[0]=0
# p2[1]=0
# p2[3]=0
#
# p3[0]=0
# p3[1]=0
# p3[2]=0
#
# ax4.plot(areas[rng], func(areas[rng], *p1), '.-', label='SPE')
# ax4.plot(areas[rng], func(areas[rng], *p2), '.-', label='DPE')
# ax4.plot(areas[rng], func(areas[rng], *p3), '.-', label='TrPE')
ax4.set_ylim(1,10000)
ax4.legend()

x=np.arange(1000)
ax6.plot(x, WF, 'r.', label='mean WF')
ax6.plot(x, BL, 'y.-', alpha=0.2)
ax6.fill_between(x[left:right], y1=np.amin(WF), y2=0)
factor=-(p[-2])/np.sum(WF[left:right])
ax6.legend()

spe=factor*np.sum(rec['spe'], axis=0)
ax5.plot(x, spe, 'r.', label='mean SPE')
ax3.axvline(-np.amin(spe), ymin=0, ymax=1)
ax5.legend()

np.savez(path+'areas', areas=areas/p[-2], H_areas=h_area, rng_area=rng, NAC=len(rec[rec['height']>height_cut]), spe=spe, NH=len(noise['noise']), NHC=len(noise[noise['noise']>height_cut]['noise']))
np.savez(path+'cuts', blw_cut=blw_cut, height_cut=height_cut, left=left, right=right, rise_time_cut=rise_time_cut)

plt.show()
