import numpy as np
from scipy.special import sph_harm
from PMTgiom import make_pmts
import sys

Ry=np.array([[1,0,0], [0,0,-1], [0,1,0]])

pmts=np.array([0,1,4,7,8,14])
pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn, r=make_pmts(pmts)
pmt_mid=np.matmul(pmt_mid, Ry.T)

theta=np.arccos(pmt_mid[:,2])
phi=np.arctan(pmt_mid[:,1]/pmt_mid[:,0])

def Ylm(n):
    a0=np.sum(n)*np.sqrt(4*np.pi)
    a1=np.zeros(3)
    for i in range(3):
        m=i-1
        a1[i]=np.sum(n*np.real(sph_harm(m,1,theta, phi)+(-1)**m*sph_harm(-m,1,theta, phi)))
    return a0, a1
