import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf
import warnings
from scipy.signal import convolve2d, convolve
from PMTgiom import make_pmts
from scipy.optimize import curve_fit

# mid, pmt_r, pmt_l, pmt_up, pmt_dn, r=make_pmts(np.arange(20))
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_aspect("equal")
#
# R=0.36
# r=1*R
# u, v= np.mgrid[0:2*np.pi:18j, 0:np.pi:14j]
# x = r*np.cos(u)*np.sin(v)
# y = r*np.sin(u)*np.sin(v)
# z = r*np.cos(v)
# ax.plot_wireframe(x, y, z, color="r", alpha=0.2)
#
# V=[4*np.pi*r**3/3]
# X=[0]
# Y=[0]
# Z=[0]
#
# v=(1-R**3)*np.pi/6*np.cos(3*np.pi/8)
# r=0.75*(1-R**4)/(1-R**3)
# for i in range(8):
#     X.append(r*np.cos(i*np.pi/4))
#     Y.append(r*np.sin(i*np.pi/4))
#     Z.append(0)
#     V.append(v)
#
# v=(1-R**3)/3*np.pi/3*(np.cos(np.pi/8)-np.cos(3*np.pi/8))
# for i in range(6):
#     X.append(r*np.cos(i*np.pi/3+np.pi/6)*np.sin(np.pi/4))
#     Y.append(r*np.sin(i*np.pi/3+np.pi/6)*np.sin(np.pi/4))
#     Z.append(r*np.cos(np.pi/4))
#     V.append(v)
#
#     X.append(r*np.cos(i*np.pi/3+np.pi/6)*np.sin(np.pi/4))
#     Y.append(r*np.sin(i*np.pi/3+np.pi/6)*np.sin(np.pi/4))
#     Z.append(-r*np.cos(np.pi/4))
#     V.append(v)
#
# v=(1-R**3)/3*2*np.pi*(1-np.cos(np.pi/8))
# X.append(0)
# Y.append(0)
# Z.append(-r)
# V.append(v)
# X.append(0)
# Y.append(0)
# Z.append(r)
# V.append(v)
#
# ax.scatter(X, Y, Z, marker='.', color='k')
# print(np.sum(V), 4*np.pi/3)
#
# plt.figure()
# plt.bar(np.arange(len(V)), V)
# plt.show()

def make_dS(d, m, rt, up):
    r=np.sqrt(np.sum(rt[0]**2))
    dS=np.zeros(len(m))
    a=np.linspace(-1,1,1000, endpoint=True)
    I=np.arange(len(a)**2)
    for i in range(len(dS)):
        x=m[i,0]+a[I//len(a)]*rt[i,0]+a[I%len(a)]*up[i,0]-d[0]
        y=m[i,1]+a[I//len(a)]*rt[i,1]+a[I%len(a)]*up[i,1]-d[1]
        z=m[i,2]+a[I//len(a)]*rt[i,2]+a[I%len(a)]*up[i,2]-d[2]
        dS[i]=np.sum((1-np.sum(d*m[i]))/(np.sqrt(x**2+y**2+z**2)**3))*((a[1]-a[0])*r)**2
    return dS/(4*np.pi)

def mash(pmts):
    mid, rt, pmt_l, up, pmt_dn=make_pmts(pmts)
    R=0.36
    r=1*R
    V=[4*np.pi*r**3/3]
    X=[0]
    Y=[0]
    Z=[0]

    v=(1-R**3)*np.pi/6*np.cos(3*np.pi/8)
    r=0.75*(1-R**4)/(1-R**3)
    for i in range(8):
        X.append(r*np.cos(i*np.pi/4))
        Y.append(r*np.sin(i*np.pi/4))
        Z.append(0)
        V.append(v)

    v=(1-R**3)/3*np.pi/3*(np.cos(np.pi/8)-np.cos(3*np.pi/8))
    for i in range(6):
        X.append(r*np.cos(i*np.pi/3+np.pi/6)*np.sin(np.pi/4))
        Y.append(r*np.sin(i*np.pi/3+np.pi/6)*np.sin(np.pi/4))
        Z.append(r*np.cos(np.pi/4))
        V.append(v)

        X.append(r*np.cos(i*np.pi/3+np.pi/6)*np.sin(np.pi/4))
        Y.append(r*np.sin(i*np.pi/3+np.pi/6)*np.sin(np.pi/4))
        Z.append(-r*np.cos(np.pi/4))
        V.append(v)

    v=(1-R**3)/3*2*np.pi*(1-np.cos(np.pi/8))
    X.append(0)
    Y.append(0)
    Z.append(-r)
    V.append(v)
    X.append(0)
    Y.append(0)
    Z.append(r)
    V.append(v)

    r_mash=np.vstack((X, np.vstack((Y,Z)))).T/4
    dS=np.zeros((len(r_mash), len(pmts)))
    for i in range(len(r_mash)):
        dS[i]=make_dS(r_mash[i], mid, rt, up)


    return r_mash, np.array(V)/(4*np.pi/3), dS

def make_V(R):
    V=np.zeros(len(R))
    N=100000
    costheta=np.random.uniform(-1,1,N)
    phi=np.random.uniform(0,2*np.pi,N)
    r3=np.random.uniform(0,(10/40)**3,N)
    r=r3**(1/3)
    v=np.vstack((r*np.sin(np.arccos(costheta))*np.cos(phi), np.vstack((r*np.sin(np.arccos(costheta))*np.sin(phi), r*costheta))))
    D=(np.sum(v**2, axis=0)-2*np.matmul(R,v)).T+np.sum(R**2, axis=1)
    V=np.histogram(np.argmin(D, axis=1), bins=np.arange(len(R)+1)-0.5)[0]
    return V/N


# r_mash, V_mash=mash()
# V=V_mash/np.sum(V_mash)
# for i in range(10000):
#     print(i)
#     V=np.vstack((V, make_V(r_mash)))
#
# np.savez('V', V=V)
# V=np.load('V.npz')['V']
# v=[]
# for i in range(len(r_mash)):
#     h,bins=np.histogram(V[1:,i], bins=100)
#     v.append(np.median(V[1:,i]))
# plt.plot(v, 'k.')
#
# V=np.zeros(len(v))
# V[0]=v[0]
# V[1:9]=np.mean(v[1:9])
# V[9:-2]=np.mean(v[9:-2])
# V[-2:]=np.mean(v[-2:])
# for i in range(len(V)):
#     print(v[i], V[i])
# np.savez('V.npz', V=V)
# plt.show()
