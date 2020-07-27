import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def make_pmts(PMTs):

    r=21/40

    pmt_mid=np.array([[1/np.sqrt(2)*np.cos(120*np.pi/180), 1/np.sqrt(2)*np.sin(120*np.pi/180), 1/np.sqrt(2)],
                    [-1/np.sqrt(2), 0, 1/np.sqrt(2)],
                    [1/np.sqrt(2)*np.cos(240*np.pi/180), 1/np.sqrt(2)*np.sin(240*np.pi/180), 1/np.sqrt(2)],
                    [0,1,0],
                    [-1/np.sqrt(2), 1/np.sqrt(2), 0],
                    [-1,0,0],
                    [-1/np.sqrt(2), -1/np.sqrt(2),0],
                    [1/np.sqrt(2)*np.cos(120*np.pi/180), 1/np.sqrt(2)*np.sin(120*np.pi/180), -1/np.sqrt(2)],
                    [-1/np.sqrt(2), 0, -1/np.sqrt(2)],
                    [1/np.sqrt(2)*np.cos(240*np.pi/180), 1/np.sqrt(2)*np.sin(240*np.pi/180), -1/np.sqrt(2)],
                    [1/np.sqrt(2)*np.cos(300*np.pi/180), 1/np.sqrt(2)*np.sin(300*np.pi/180), 1/np.sqrt(2)],
                    [1/np.sqrt(2), 0,1/np.sqrt(2)],
                    [1/np.sqrt(2)*np.cos(60*np.pi/180), 1/np.sqrt(2)*np.sin(60*np.pi/180), 1/np.sqrt(2)],
                    [0, -1, 0],
                    [1/np.sqrt(2), -1/np.sqrt(2), 0],
                    [1, 0, 0],
                    [1/np.sqrt(2), 1/np.sqrt(2), 0],
                    [1/np.sqrt(2)*np.cos(300*np.pi/180), 1/np.sqrt(2)*np.sin(300*np.pi/180), -1/np.sqrt(2)],
                    [1/np.sqrt(2), 0, -1/np.sqrt(2)],
                    [1/np.sqrt(2)*np.cos(60*np.pi/180), 1/np.sqrt(2)*np.sin(60*np.pi/180), -1/np.sqrt(2)],
                    ])


    pmt_r=np.zeros((len(PMTs),3))
    pmt_up=np.zeros((len(PMTs),3))

    for i, pmt in enumerate(PMTs):
        a=pmt_mid[pmt][0]
        b=pmt_mid[pmt][1]
        pmt_r[i]=[-0.5*r*b/np.sqrt(a**2+b**2), 0.5*r*a/np.sqrt(a**2+b**2), 0]
        pmt_up[i]=np.cross(pmt_mid[pmt], pmt_r[i])
    return pmt_mid[PMTs], pmt_r, pmt_up



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


def make_mash(pmts):
    mid, rt, up=make_pmts(pmts)
    X=[0]
    Y=[0]
    Z=[0]

    r=0.75
    for i in range(8):
        X.append(r*np.cos(i*np.pi/4))
        Y.append(r*np.sin(i*np.pi/4))
        Z.append(0)
    for i in range(6):
        X.append(r*np.cos(i*np.pi/3+np.pi/6)*np.sin(np.pi/4))
        Y.append(r*np.sin(i*np.pi/3+np.pi/6)*np.sin(np.pi/4))
        Z.append(r*np.cos(np.pi/4))
    for i in range(6):
        X.append(r*np.cos(i*np.pi/3+np.pi/6)*np.sin(np.pi/4))
        Y.append(r*np.sin(i*np.pi/3+np.pi/6)*np.sin(np.pi/4))
        Z.append(-r*np.cos(np.pi/4))
    X.append(0)
    Y.append(0)
    Z.append(-r)
    X.append(0)
    Y.append(0)
    Z.append(r)
    r=np.vstack((X, np.vstack((Y,Z)))).T/4
    dS=np.zeros((len(r), len(pmts)))
    for i in range(len(r)):
        dS[i]=make_dS(r[i], mid, rt, up)

    try:
        V=np.load('V.npz')['V']
    except:
        V=np.zeros(len(r))
    return r, V/np.sum(V), dS


def make_v(N, mu, x1, x2, r_mash, V_mash):
    k=0
    V=np.zeros(3)
    while len(V)<=N:
        k+=1
        costheta=np.random.uniform(-1,1)
        phi=np.random.uniform(0,2*np.pi)
        r3=np.random.uniform(0,(10/40)**3)
        r=r3**(1/3)
        v=np.array([r*np.sin(np.arccos(costheta))*np.cos(phi), r*np.sin(np.arccos(costheta))*np.sin(phi), r*costheta])
        d=v[x1]+np.sqrt((10/40)**2-v[x2]**2-v[-1]**2)
        P=np.exp(-d/mu)
        if 1==np.random.choice([0,1], size=1,  p=[1-P, P]):
            V=np.vstack((V,v))
            print('In make v', len(V[:,0]), 'out of', N)

    return V[1:]
