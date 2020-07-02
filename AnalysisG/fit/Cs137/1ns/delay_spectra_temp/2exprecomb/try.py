import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf
import warnings
from minimize import minimize, make_ps
from PMTgiom import make_pmts
from make_mash import mash
from scipy.signal import convolve2d, convolve


pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn, r=make_pmts(np.arange(20))
r_mash, V_mash=mash()
def make_dS(d, m, rt, up):
    dS=np.zeros(len(m))
    a=np.linspace(-1,1,100, endpoint=True)
    I=np.arange(len(a)**2)
    for i in range(len(dS)):
        x=m[i,0]+a[I//len(a)]*rt[i,0]+a[I%len(a)]*up[i,0]-d[0]
        y=m[i,1]+a[I//len(a)]*rt[i,1]+a[I%len(a)]*up[i,1]-d[1]
        z=m[i,2]+a[I//len(a)]*rt[i,2]+a[I%len(a)]*up[i,2]-d[2]
        dS[i]=np.sum((1-np.sum(d*m[i]))/(np.sqrt(x**2+y**2+z**2)**3))*((a[1]-a[0])*r/2)**2
    return dS
make_dS([0.2,0.2,0.2], pmt_mid, pmt_r, pmt_up)
