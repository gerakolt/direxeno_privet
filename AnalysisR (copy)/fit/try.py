import multiprocessing
import numpy as np
import sys
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf, expi
from scipy.special import comb
from scipy.signal import convolve2d
import time
from admin import make_iter
from PMTgiom import whichPMT
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def traceLXe(vs, us, nLXe):
    nHPFS=1.6
    # us and vs is an (3,N) array
    a=(np.sqrt(np.sum(vs*us, axis=0)**2+(0.25**2-np.sum(vs**2, axis=0)))-np.sum(us*vs, axis=0)) # N len array

    vmin=np.amin(np.sqrt(np.sum(vs**2, axis=0)))
    ind=np.argmin(np.sqrt(np.sum(vs**2, axis=0)))
    vs=vs+us*a
    rot=np.cross(us,vs, axis=0)
    rot=rot/np.sqrt(np.sum(rot**2, axis=0))
    inn=np.arccos(np.sum(vs*us, axis=0)/np.sqrt(np.sum(vs**2, axis=0))) # N len array
    TIR=np.nonzero(np.sin(inn)*nLXe/nHPFS>1)[0]
    Rif=np.nonzero(np.sin(inn)*nLXe/nHPFS<=1)[0]

    if len(Rif)>0:
        out=np.arcsin(np.sin(inn[Rif])*nLXe/nHPFS) # N len array
        theta=inn[Rif]-out
        us[:,Rif]=rot[:,Rif]*np.sum(rot[:,Rif]*us[:,Rif], axis=0)+np.cos(theta)*np.cross(np.cross(rot[:,Rif], us[:,Rif], axis=0), rot[:,Rif], axis=0)+np.sin(theta)*np.cross(rot[:,Rif], us[:,Rif], axis=0)

    if len(TIR)>0:
        theta=-(np.pi-2*inn[TIR])
        us[:,TIR]=rot[:, TIR]*np.sum(rot[:,TIR]*us[:,TIR], axis=0)+np.cos(theta)*np.cross(np.cross(rot[:,TIR], us[:,TIR], axis=0), rot[:,TIR], axis=0)+np.sin(theta)*np.cross(rot[:,TIR], us[:,TIR], axis=0)

    us=us/np.sqrt(np.sum(us*us, axis=0))
    return vs+1e-6*us, us

def tracetoLXe(vs, us, nLXe):
    nHPFS=1.6
    # us and vs is an (3,N) array
    d=np.sum(vs*us, axis=0)**2+(0.25**2-np.sum(vs**2, axis=0))
    toLXe=np.nonzero(d>=0)[0]
    toHPFS=np.nonzero(d>=0)[0]
    if len(toLXe)>0:
        a=(-np.sqrt(d[:, toLXe])-np.sum(us[:, toLXe]*vs[:, toLXe], axis=0)) # N len array
        vs[:, toLXe]=vs[:, toLXe]+us[:, toLXe]*a
        rot=np.cross(us[:, toLXe],-vs[:, toLXe], axis=0)
        rot=rot/np.sqrt(np.sum(rot**2, axis=0))
        inn=np.pi-np.arccos(np.sum(vs[:, toLXe]*us[:, toLXe], axis=0)/np.sqrt(np.sum(vs[:, toLXe]**2, axis=0))) # N len array
        TIR=np.nonzero(np.sin(inn)*nHPFS/nLXe>1)[0]
        Rif=np.nonzero(np.sin(inn)*nHPFS/nLXe<=1)[0]

        if len(Rif)>0:
            out=np.arcsin(np.sin(inn[Rif])*nHPFS/nLXe) # N len array
            theta=inn[Rif]-out
            us[:,toLXe[Rif]]=rot[:,Rif]*np.sum(rot[:,Rif]*us[:,toLXe[Rif]], axis=0)+np.cos(theta)*np.cross(np.cross(rot[:,Rif], us[:,toLXe[Rif]], axis=0),
                rot[:,Rif], axis=0)+np.sin(theta)*np.cross(rot[:,Rif], us[:,toLXe[Rif]], axis=0)

        if len(TIR)>0:
            theta=-(np.pi-2*inn[TIR])
            us[:,ToLXe[TIR]]=rot[:, TIR]*np.sum(rot[:,TIR]*us[:,ToLXe[TIR]], axis=0)+np.cos(theta)*np.cross(np.cross(rot[:,TIR], us[:,TIR], axis=0),
                rot[:,TIR], axis=0)+np.sin(theta)*np.cross(rot[:,TIR], us[:,ToLXe[TIR]], axis=0)
    if len(toHPFS)>0:
        vs[:,toHPFS], us[:,toHPFS]=tracetoVac(vs[:,toHPFS], us[:,toHPFS])
    us=us/np.sqrt(np.sum(us*us, axis=0))
    return vs+1e-6*us, us


def tracetoVac(vs, us):
    nHPFS=1.6
    # us and vs is an (3,N) array
    a=(np.sqrt(np.sum(vs*us, axis=0)**2+(0.75**2-np.sum(vs**2, axis=0)))-np.sum(us*vs, axis=0)) # N len array
    vs=vs+us*a
    rot=np.cross(us,vs, axis=0)
    rot=rot/np.sqrt(np.sum(rot**2, axis=0))
    inn=np.arccos(np.sum(vs*us, axis=0)/np.sqrt(np.sum(vs**2, axis=0))) # N len array
    TIR=np.nonzero(np.sin(inn)*nHPFS>1)[0]
    Rif=np.nonzero(np.sin(inn)*nHPFS<=1)[0]

    if len(Rif)>0:
        out=np.arcsin(np.sin(inn[Rif])*nHPFS) # N len array
        theta=inn[Rif]-out
        us[:,Rif]=rot[:,Rif]*np.sum(rot[:,Rif]*us[:,Rif], axis=0)+np.cos(theta)*np.cross(np.cross(rot[:,Rif], us[:,Rif], axis=0), rot[:,Rif], axis=0)+np.sin(theta)*np.cross(rot[:,Rif], us[:,Rif], axis=0)

    if len(TIR)>0:
        theta=-(np.pi-2*inn[TIR])
        us[:,TIR]=rot[:,TIR]*np.sum(rot[:,TIR]*us[:,TIR], axis=0)+np.cos(theta)*np.cross(np.cross(rot[:,TIR], us[:,TIR], axis=0), rot[:,TIR], axis=0)+np.sin(theta)*np.cross(rot[:,TIR], us[:,TIR], axis=0)
    us=us/np.sqrt(np.sum(us*us, axis=0))

    return vs+1e-6*us, us

count=0
while True:
    print(count)
    count+=1
    N=1
    nLXe=1.6879227009612343

    np.random.seed(int(time.time()%(2**32)))

    costheta=np.random.uniform(-1,1)
    phi=np.random.uniform(0,2*np.pi)
    r3=np.random.uniform(0,(10/40)**3)
    r=r3**(1/3)
    v=np.array([r*np.sin(np.arccos(costheta))*np.cos(phi), r*np.sin(np.arccos(costheta))*np.sin(phi), r*costheta])
    v=np.array([-0.01587863, -0.17735465, -0.17547945])

    costheta=np.random.uniform(-1,1, N)
    phi=np.random.uniform(0,2*np.pi, N)
    us=np.array([np.sin(np.arccos(costheta))*np.cos(phi), np.sin(np.arccos(costheta))*np.sin(phi), costheta])
    vs=np.repeat(v, N).reshape(3, N)
    us=np.array([-0.73495305 , 0.66167179 ,-0.14844011])
    us=us.reshape(3,N)
    while np.any(np.sqrt(np.sum(vs**2, axis=0))<0.75):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = 0.25 * np.outer(np.cos(u), np.sin(v))
        y = 0.25 * np.outer(np.sin(u), np.sin(v))
        z = 0.25 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.25)

        x = 0.75 * np.outer(np.cos(u), np.sin(v))
        y = 0.75 * np.outer(np.sin(u), np.sin(v))
        z = 0.75 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.25)
        v_init=np.array(vs)
        ind_LXe=np.nonzero(np.sqrt(np.sum(vs**2, axis=0))<=0.25)[0]
        ind_toLXe=np.nonzero(np.logical_and(np.sqrt(np.sum(vs**2, axis=0))>0.25, np.sum(vs*us, axis=0)<=0))[0]
        ind_toVac=np.nonzero(np.logical_and(np.sqrt(np.sum(vs**2, axis=0))>0.25, np.sum(vs*us, axis=0)>0))[0]
        if len(ind_LXe)>0:
            # print('LXe', ind_LXe)
            vs[:,ind_LXe], us[:,ind_LXe]=traceLXe(vs[:,ind_LXe], us[:,ind_LXe], nLXe)
        if len(ind_toLXe)>0:
            # print('HPFS to LXe', ind_toLXe)
            vs[:,ind_toLXe], us[:,ind_toLXe]=tracetoLXe(vs[:,ind_toLXe], us[:,ind_toLXe], nLXe)
        if len(ind_toVac)>0:
            # print('HPFS to Vax', ind_toVac)
            vs[:,ind_toVac], us[:,ind_toVac]=tracetoVac(vs[:,ind_toVac], us[:,ind_toVac])
        ax.quiver(v_init[0], v_init[1], v_init[2], (vs-v_init)[0], (vs-v_init)[1], (vs-v_init)[2], color='k')
        plt.show()
