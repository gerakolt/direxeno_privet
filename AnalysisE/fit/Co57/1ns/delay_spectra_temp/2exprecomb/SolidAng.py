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




def make_dS(d, pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn):
    dS=np.zeros(len(pmt_mid))
    ETA=np.linspace(-1,1,100, endpoint=True)
    zeta=np.linspace(-1,1,100, endpoint=True)
    deta=ETA[1]-ETA[0]
    for i in range(len(dS)):
        for eta in ETA:
            dS[i]+=np.sum(pmt_r[i]*pmt_r[i])*deta**2*np.sum((1-np.sum(d*pmt_mid[i]))/((1-2*np.sum(d*pmt_mid[i])+np.sum(d*d)-2*eta*np.sum(d*pmt_r[i])-2*zeta*np.sum(d*pmt_up[i])
                            +eta**2*np.sum(pmt_r[i]*pmt_r[i])+zeta**2*np.sum(pmt_up[i]*pmt_up[i]))**(3/2)))
    return dS/(4*np.pi)



pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn, r=make_pmts(np.arange(20))

h=np.zeros(20)
K=25
d=np.array([0.1,-0.1,0.85])
dS=make_dS(d, pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn)
pmts=np.arange(len(pmt_mid))
for i in range(K):
    print(i)
    N=7000
    costheta=np.random.uniform(-1,1,N)
    phi=np.random.uniform(0,2*np.pi,N)
    n=whichPMT(costheta, phi, pmt_mid, pmt_r, pmt_up, r, d)
    h+=n

plt.figure()
plt.bar(pmts, h/K-N*dS)
# plt.bar(pmts, dS)

plt.show()
