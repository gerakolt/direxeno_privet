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
from minimize import minimize, make_ps
from scipy.signal import convolve2d, convolve
from PMTgiom import make_pmts


pmts=np.arange(20)

pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn, r=make_pmts(pmts)

while True:
    costheta=np.random.uniform(-1,1)
    phi=np.random.uniform(0,2*np.pi)
    r2=np.random.uniform(0,1)
    v=np.array([np.sqrt(r2)*np.sin(np.arccos(costheta))*np.cos(phi), np.sqrt(r2)*np.sin(np.arccos(costheta))*np.sin(phi), np.sqrt(r2)*costheta])

    costheta=np.random.uniform(-1,1)
    phi=np.random.uniform(0,2*np.pi)
    u=np.array([np.sin(np.arccos(costheta))*np.cos(phi), np.sin(np.arccos(costheta))*np.sin(phi), costheta])

    t=np.sqrt(1-np.sum(np.cross(v,u)**2))-np.sum(v*u)
    x=v+u*t

    n=np.argmax(np.matmul(pmt_mid, x))
    y=x-np.sum(x*pmt_mid[n])*pmt_mid[n]


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver(v[0], v[1], v[2], t*u[0], t*u[1], t*u[2], color='r')
    if np.logical_and(np.abs(np.sum(y*pmt_up[n]))<np.sum(pmt_up[n]**2), np.abs(np.sum(y*pmt_r[n]))<np.sum(pmt_r[n]**2)):
        ax.quiver(pmt_mid[n,0], pmt_mid[n,1], pmt_mid[n,2], y[0], y[1], y[2], color='orange')
    else:
        ax.quiver(pmt_mid[n,0], pmt_mid[n,1], pmt_mid[n,2], y[0], y[1], y[2], color='y')
    for i in range(20):
        # ax.quiver(0,0,0,pmt_mid[i,0], pmt_mid[i,1], pmt_mid[i,2], color='k')
        if i==n:
            ax.quiver(pmt_mid[i,0], pmt_mid[i,1], pmt_mid[i,2], pmt_r[i,0], pmt_r[i,1], pmt_r[i,2], color='g')
            ax.quiver(pmt_mid[i,0], pmt_mid[i,1], pmt_mid[i,2], -pmt_r[i,0], -pmt_r[i,1], -pmt_r[i,2], color='g')
            ax.quiver(pmt_mid[i,0], pmt_mid[i,1], pmt_mid[i,2], pmt_up[i,0], pmt_up[i,1], pmt_up[i,2], color='g')
            ax.quiver(pmt_mid[i,0], pmt_mid[i,1], pmt_mid[i,2], -pmt_up[i,0], -pmt_up[i,1], -pmt_up[i,2], color='g')
        else:
            ax.quiver(pmt_mid[i,0], pmt_mid[i,1], pmt_mid[i,2], pmt_r[i,0], pmt_r[i,1], pmt_r[i,2], color='b')
            ax.quiver(pmt_mid[i,0], pmt_mid[i,1], pmt_mid[i,2], -pmt_r[i,0], -pmt_r[i,1], -pmt_r[i,2], color='b')
            ax.quiver(pmt_mid[i,0], pmt_mid[i,1], pmt_mid[i,2], pmt_up[i,0], pmt_up[i,1], pmt_up[i,2], color='b')
            ax.quiver(pmt_mid[i,0], pmt_mid[i,1], pmt_mid[i,2], -pmt_up[i,0], -pmt_up[i,1], -pmt_up[i,2], color='b')

    plt.show()
