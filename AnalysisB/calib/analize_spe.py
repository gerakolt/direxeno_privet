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

pmt=0
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
data=np.load(path+'cuts.npz')
blw_cut=data['blw_cut']
height_cut=data['height_cut']
left=data['left']
right=data['right']
BL=np.load(path+'BL.npz')['BL']
rise_time_cut=6

data=np.load(path+'spe.npz')
rec=data['rec']


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6))=plt.subplots(3,2)

fig.suptitle('PMT{}'.format(pmt))

ax1.hist(rec['blw'], bins=100, range=[0,60], label='blw')
ax1.axvline(blw_cut, ymin=0, ymax=1)
ax1.set_yscale('log')
ax1.legend()

rec=rec[rec['blw']<blw_cut]

ax2.hist(rec['rise_time'], bins=np.arange(30)-0.5, label='rise time')
ax2.set_yscale('log')
ax2.axvline(rise_time_cut, ymin=0, ymax=1)
ax2.legend()

rec=rec[rec['rise_time']>rise_time_cut]


h_heights, bins, pat=ax3.hist(rec['height'], bins=100, label='height', range=[0,200], histtype='step')
heights=0.5*(bins[1:]+bins[:-1])
ax3.axvline(height_cut, ymin=0, ymax=1)
ax3.set_yscale('log')
ax3.legend()

# ax4.hist(rec[rec['height']>height_cut]['area'], bins=100, label='area, h>h_cut', range=[-2000, 20000], histtype='step')
h_area, bins, pat=ax4.hist(rec['area'], bins=100, label='area', range=[-2700, 5000], histtype='step')
areas=0.5*(bins[:-1]+bins[1:])
rng=np.nonzero(np.logical_and(areas>1000, areas<5000))[0]
ax4.set_yscale('log')
p=[4.73908746e+01, 2.11479849e+00, 2.63826214e-01, 1.55582462e-19, 6.70539645e+02, 3.52426757e+02, 1600, 550]
bounds_up=[1e5, 30, 1e5, 1e5, 1e5, 1e5, 5000, 1e5]
bounds_dn=[0, 2, 0, 0, -2000, 0, 1500, 500]

p, cov=curve_fit(func, areas[rng], h_area[rng], p0=p, bounds=[bounds_dn, bounds_up])
print(p)
ax4.plot(areas[rng], func(areas[rng], *p), '.-')
p1=np.array(p)
p2=np.array(p)
p3=np.array(p)
p1[0]=0
p1[2]=0
p1[3]=0

p2[0]=0
p2[1]=0
p2[3]=0

p3[0]=0
p3[1]=0
p3[2]=0

ax4.plot(areas[rng], func(areas[rng], *p1), '.-', label='SPE')
ax4.plot(areas[rng], func(areas[rng], *p2), '.-', label='DPE')
ax4.plot(areas[rng], func(areas[rng], *p3), '.-', label='TrPE')
ax4.set_ylim(1,np.amax(h_area)+1000)
ax4.legend()


x=np.arange(1000)/5
spe=np.sum(rec['spe'], axis=0)
maxi=np.argmin(spe)
area=-(np.sum(spe[maxi-100:maxi+200])+np.sum(spe[maxi-50:maxi+150])+np.sum(spe[maxi-100:maxi+150])+np.sum(spe[maxi-50:maxi+200]))/4
spe=p[-2]*spe/area
ax5.plot(x, spe, 'r.', label='mean SPE')
ax5.fill_between(x[maxi-100:maxi+200], y1=np.amin(spe), y2=0)
ax5.fill_between(x[maxi-50:maxi+150], y1=np.amin(spe), y2=0)
ax5.plot(x, BL, 'y.')
ax3.axvline(-np.amin(spe), ymin=0, ymax=1)
ax5.legend()

# np.savez(path+'areas', areas=areas/p[-2], H_areas=h_area, rng_area=rng, spe=spe)
# np.savez(path+'cuts', blw_cut=blw_cut, height_cut=height_cut, left=left, right=right, rise_time_cut=rise_time_cut)

plt.show()
