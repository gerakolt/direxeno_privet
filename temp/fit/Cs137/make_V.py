import numpy as np
import matplotlib.pyplot as plt
from PMTgiom import make_mash

def make_V(R):
    dA=np.zeros(len(R))
    dB=np.zeros(len(R))
    V=np.zeros(len(R))
    N=1000000
    costheta=np.random.uniform(-1,1,N)
    phi=np.random.uniform(0,2*np.pi,N)
    r3=np.random.uniform(0,(10/40)**3,N)
    r=r3**(1/3)
    v=np.vstack((r*np.sin(np.arccos(costheta))*np.cos(phi), np.vstack((r*np.sin(np.arccos(costheta))*np.sin(phi), r*costheta))))
    D=(np.sum(v**2, axis=0)-2*np.matmul(R,v)).T+np.sum(R**2, axis=1)
    V=np.histogram(np.argmin(D, axis=1), bins=np.arange(len(R)+1)-0.5)[0]

    Ox=-np.sqrt((10/40)**2-v[1]**2-v[2]**2)
    Oy=-np.sqrt((10/40)**2-v[0]**2-v[2]**2)
    for i in range(len(R)):
        dA[i]=np.mean((v[0]-Ox)[np.argmin(D, axis=1)==i])
        # dB[i]=np.mean((v[1]-Oy)[np.argmin(D, axis=1)==i])
        dB[i]=R[i,0]+np.sqrt((1/4)**2-R[i,1]**2-R[i,2]**2)

    return V/N, dA, dB


r, V, dS=make_mash(np.arange(1))
V, dA, dB=make_V(r)
# for i in range(1000):
#     print(i)
#     V=np.vstack((V, make_V(r)))


plt.figure()
plt.plot(dA, 'ko')
plt.plot(dB, 'ro')

# v=np.median(V[1:], axis=0)
# V[0,0]=v[0]
# V[0,1:9]=np.mean(v[1:9])
# V[0,9:-2]=np.mean(v[9:-2])
# V[0,-2:]=np.mean(v[-2:])
# plt.plot(v, 'k.')
# plt.plot(V[0], 'r.')
# np.savez('V', V=V[0])
plt.show()
