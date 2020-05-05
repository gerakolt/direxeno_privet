import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


pmt=0
data=np.load('wf{}.npz'.format(pmt))
rec=data['rec']
left=data['left']
right=data['right']
init=data['init']

try:
    data=np.load('spe{}.npz'.format(pmt))
    blw_cut=data['blw_cut']
    height_cut=data['height_cut']
    REC=data['rec']
    spe=data['wf']
    bl=data['bl']
except:
    blw_cut=0
    height_cut=0

try:
    data=np.load('spe_correct{}.npz'.format(pmt))
    rec_c=data['rec']
except:
    print('no corrected data')

x=np.arange(1000)
fig=plt.figure()
fig.suptitle('PMT{}'.format(pmt))
ax=fig.add_subplot(221)
ax.hist(rec['blw'], bins=100, range=[0,50], label='blw')
ax.axvline(blw_cut, ymin=0, ymax=1, color='k',linestyle='--')
ax.legend()
ax.set_yscale('log')

ax=fig.add_subplot(222)
ax.hist(rec['height'], bins=100, range=[0,200], label='raw height', histtype='step')
try:
    ax.hist(REC['height'], bins=100, range=[0,200], label='height', histtype='step')
except:
    ax.hist(np.zeros(1000), bins=100, range=[0,200], label='height - No data', histtype='step')
try:
    ax.hist(rec_c['height'], bins=100, label='height corrected', histtype='step')
except:
    ax.hist(np.zeros(1000), bins=100, label='height corrected - No data', histtype='step')

ax.axvline(height_cut, ymin=0, ymax=1, color='k', linestyle='--')
ax.set_yscale('log')
ax.legend()

# def func(x ,a,b,m_p,s_p,m,s):
#     return a*np.exp(-0.5*(x-m_p)**2/s_p**2)+b*np.exp(-0.5*(x-m)**2/s**2)

def func(x ,a,b,c,m_p,s_p,m,s):
    return a*np.exp(-0.5*(x-m_p)**2/s_p**2)+b*np.exp(-0.5*(x-(m+m_p))**2/(s_p**2+s**2))+c*np.exp(-0.5*(x-(m+2*m_p))**2/(s_p**2+2*s**2))

ax=fig.add_subplot(223)
try:
    h,bins,pa=ax.hist(REC['area'], bins=100, label='area', histtype='step')
except:
    h,bins,pa=ax.hist(np.zeros(1000), bins=100, label='area - No data')
r=0.5*(bins[1:]+bins[:-1])
rng=np.nonzero(np.logical_and(r<7500, r>-900))[0]
p0=[np.amax(h), np.amax(h[r>1000]), np.amax(h[r>2000]), r[np.argmax(h)], 0.5*r[np.argmax(h)], 1000, 500]
try:
    p,cov=curve_fit(func, r[rng], h[rng], p0=p0)
    z=np.linspace(r[rng][0], r[rng][-1], 1000)
    ax.plot(z, func(z, *p), 'r--', label='m={}, s={}'.format(p[-2], p[-1]))
    np.savez('gian{}'.format(pmt), p=p)

except:
    ax.plot(np.zeros(1000), label='no fit')
try:
    ax.hist(rec_c['area'], bins=100, label='area corrected', histtype='step')
except:
    ax.hist(np.zeros(1000), bins=100, label='area corrected - No data', histtype='step')
ax.set_yscale('log')
ax.legend()

ax=fig.add_subplot(224)
try:
    x=np.arange(1000)
    ax.plot(np.zeros(1000), 'k--')
    ax.plot(x, spe, 'r.-', label='spe')
    ax.plot(x, bl, 'k--', label='bl')
    ax.plot(x, spe-bl, 'g.-', label='spe-bl')
    ax.fill_between(x[left:right], y1=np.amin(spe), y2=0)
except:
    ax.plot(np.zeros(1000), label='No SPE')
ax.legend()
plt.show()
