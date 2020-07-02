import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf
import warnings
from minimize import minimize, make_ps
from PMTgiom import make_pmts, make_pmts_try
from make_mash import mash
from scipy.signal import convolve2d, convolve

# r_mash, V_mash=mash()
mid, rt, up=make_pmts_try()
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_xlim(-2,2)
# ax.set_ylim(-2,2)
# ax.set_zlim(-2,2)
#
#
# for i in range(6):
#     ax.quiver(mid[i,0], mid[i,1], mid[i,2], rt[i,0], rt[i,1], rt[i,2])
#     ax.quiver(mid[i,0], mid[i,1], mid[i,2], -rt[i,0], -rt[i,1], -rt[i,2])
#     ax.quiver(mid[i,0], mid[i,1], mid[i,2], -up[i,0], -up[i,1], -up[i,2])
#     ax.quiver(mid[i,0], mid[i,1], mid[i,2], up[i,0], up[i,1], up[i,2])
# plt.show()
#
#
# def whichPMT(v, us, mid, rt, up):
#     t=np.sqrt(1-np.sum(np.cross(us.T, v)**2, axis=1))-np.matmul(us.T,v)
#     x=v*np.ones_like(us.T)+(us*t).T
#     n=np.argmax(np.matmul(x,mid.T), axis=1)
#     y=x-(np.sum(mid[n]*x, axis=1)*mid[n].T).T
#     n[np.logical_or(np.abs(np.sum(y*rt[n], axis=1))>np.sum(rt[n]**2, axis=1), np.abs(np.sum(y*up[n], axis=1))>np.sum(up[n]**2, axis=1))]=-1
#     return n
#
# def whichPMT(v, us, mid, rt, up):
#     costheta=np.matmul(mid, us)
#     n=np.argmax(costheta, axis=0)
#     a=(1-np.sum(mid[n]*v, axis=1))/np.sum(us.T*mid[n], axis=1)
#     r=v+(a*us).T-mid[n]
#     n[np.logical_or(np.sum(r*rt[n], axis=1)>np.sum(rt[n]**2, axis=1), np.sum(r*up[n], axis=1)>np.sum(up[n]**2, axis=1))]=-1
#     return n

def whichPMT(v, us, mid, rt, up):
    hits=np.arange(6)
    for i in range(6):
        a=(1-np.sum(mid[i]*v, axis=0))/np.sum(us.T*mid[i], axis=1)
        r=v+(a*us).T-mid[i]
        # n[np.logical_or(np.sum(r*rt[n], axis=1)>np.sum(rt[n]**2, axis=1), np.sum(r*up[n], axis=1)>np.sum(up[n]**2, axis=1))]=-1
        hits[i]=len(np.nonzero(np.logical_and(a>0, np.logical_and(np.abs(np.sum(r*rt[i], axis=1))<np.sum(rt[i]**2, axis=0), np.abs(np.sum(r*up[i], axis=1))<np.sum(up[i]**2, axis=0))))[0])
    return hits

# def make_dS(d, m, rt, up):
#     dS=np.zeros(len(m))
#     a=np.linspace(-1,1,100, endpoint=True)
#     I=np.arange(len(a)**2)
#     for i in range(len(dS)):
#         x=m[i,0]+a[I//len(a)]*rt[i,0]+a[I%len(a)]*up[i,0]-d[0]
#         y=m[i,1]+a[I//len(a)]*rt[i,1]+a[I%len(a)]*up[i,1]-d[1]
#         z=m[i,2]+a[I//len(a)]*rt[i,2]+a[I%len(a)]*up[i,2]-d[2]
#         dS[i]=np.sum((1-np.sum(d*m[i]))/(np.sqrt(x**2+y**2+z**2)**3))*((a[1]-a[0])*r/2)**2
#     return dS/(4*np.pi)
#
# dS=make_dS([0,0,0], pmt_mid, pmt_r, pmt_up)

# costheta=np.random.uniform(-1,1, N)
# phi=np.random.uniform(0,2*np.pi, N)
N=1000000
x=np.random.uniform(-1,1,N)
y=np.random.uniform(-1,1,N)
z=np.random.uniform(-1,1,N)
I=np.histogram(np.random.randint(0,6,N), bins=np.arange(7)-0.5)[0]
us=np.vstack((x, np.vstack((y,z))))
us=us/np.sqrt(np.sum(x**2+y**2+z**2))
# us=np.vstack((np.vstack((np.sin(np.arccos(costheta))*np.cos(phi), np.sin(np.arccos(costheta))*np.sin(phi))), costheta))
hit=whichPMT(np.array([0,0,0]), us, mid, rt, up)


y=np.sqrt(N)*(I/N-1/6)
print(np.mean(y), np.var(y), 5/36)

plt.plot(np.arange(6), hit, 'ko')

plt.show()
