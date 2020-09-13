import numpy as np
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time

while True:
    np.random.seed(int(time.time()%(2**32)))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 0.25 * np.outer(np.cos(u), np.sin(v))
    y = 0.25 * np.outer(np.sin(u), np.sin(v))
    z = 0.25 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.25)

    costheta=np.random.uniform(-1,1)
    phi=np.random.uniform(0,2*np.pi)
    r3=np.random.uniform(0,(10/40)**3)
    r=r3**(1/3)
    v=np.array([r*np.sin(np.arccos(costheta))*np.cos(phi), r*np.sin(np.arccos(costheta))*np.sin(phi), r*costheta])


    costheta=np.random.uniform(-1,1)
    phi=np.random.uniform(0,2*np.pi)
    u=np.array([np.sin(np.arccos(costheta))*np.cos(phi), np.sin(np.arccos(costheta))*np.sin(phi), costheta])
    a=np.sqrt(np.sum(u*v)**2+(0.25**2-np.sum(v**2)))-np.sum(v*u)
    ax.quiver(v[0], v[1], v[2], a*u[0], a*u[1], a*u[2], color='r')
    ax.scatter(v[0], v[1], v[2], color='r', marker='o')

    v=v+a*u
    ax.quiver(0,0,0, v[0], v[1], v[2], color='k')
    ax.quiver(v[0], v[1], v[2], v[0], v[1], v[2], color='k')
    ax.scatter(v[0], v[1], v[2], color='k', marker='o')

    nLXe=1.25
    nHPFS=1.6

    inn=np.arccos(np.sum(u*v)/np.sqrt(np.sum(v**2)))
    out=np.arcsin(np.sin(inn)*nLXe/nHPFS)
    rot=np.cross(u,v)
    rot=rot/np.sqrt(np.sum(rot**2))
    theta=inn-out
    u=rot*np.sum(rot*u)+np.cos(theta)*np.cross(np.cross(rot, u), rot)+np.sin(theta)*np.cross(rot, u)
    u=u/np.sqrt(np.sum(u*u))
    ax.quiver(v[0], v[1], v[2], u[0], u[1], u[2], color='r')

    print(np.sum(u*u), np.sqrt(np.sum(v**2)))
    print(np.sin(np.arccos(np.sum(u*v)/0.25))/np.sin(inn)*nHPFS/nLXe)

    plt.show()
