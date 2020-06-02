from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from PMTgiom import make_pmts


def whichPMT(costheta, phi, mid, rt, up, r, d):
    x=np.sin(np.arccos(costheta))*np.cos(phi)
    y=np.sin(np.arccos(costheta))*np.sin(phi)
    z=costheta
    n=np.zeros(len(mid))
    for j in range(len(z)):
        hit=0
        for i in range(len(mid)):
            # print(i,j)
            b=np.sum(mid[i]*(mid[i]-d))
            R=np.array([x[j], y[j], z[j]])
            a=b/np.sum(R*mid[i])
            if a>0:
                v=mid[i]-d-a*R
                if np.abs(np.sum(v*up[i]))<(0.5*r)**2 and np.abs(np.sum(v*rt[i]))<(0.5*r)**2:
                    n[i]+=1
                    hit+=1
            if hit>1:
                print('FFFFFFFFFFFFFFFFFFuck!!!!')
                sys.exit()
    return n





pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn, r=make_pmts()

h=np.zeros(20)
K=15
d=[0,0,0]
pmts=np.arange(len(pmt_mid))
for i in range(K):
    print(i)
    N=7000
    costheta=np.random.uniform(-1,1,N)
    phi=np.random.uniform(0,2*np.pi,N)
    n=whichPMT(costheta, phi, pmt_mid, pmt_r, pmt_up, r, d)
    h+=n


plt.figure()
plt.bar(pmts, h/K)

plt.show()
