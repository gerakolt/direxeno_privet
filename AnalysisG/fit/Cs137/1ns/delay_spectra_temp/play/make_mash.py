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

mid, pmt_r, pmt_l, pmt_up, pmt_dn, r=make_pmts(np.arange(20))

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


def mash():
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

    return np.vstack((X, np.vstack((Y,Z)))).T/4, np.array(V)/(4*np.pi/3)
