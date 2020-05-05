import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf
import sys
from scipy.optimize import minimize
from fun import get_data, make_P, Int, make_z, model1, model2

lim_fit=[400]
pmts=[8,0]
Hs, N_events, Ns, H_specs=get_data(pmts)

def L(p):
    [NQ, F, tf, ts, St, Spe]=p

    if St<=0:
        return 1e8*(1-St)
    if Spe<=0:
        return 1e8*(1-Spe)
    if tf<St:
        return 1e8*(St-tf)
    if F<=0:
        return 1e8*(1-F)

    h=N_events*np.ravel(model1(p)[:shp[0],:shp[1]])
    l1=0
    for i in range(len(h)):
        if H[i]==0 and h[i]>0:
            l1-=h[i]
        elif h[i]==0 and H[i]>0:
            l1+=H[i]-h[i]-H[i]*np.log(H[i])-H[i]*1e100
        elif h[i]>0 and H[i]>0:
            l1+=H[i]*np.log(h[i])-H[i]*np.log(H[i])+H[i]-h[i]
    print(-l1, p)
    return -l1



NQ=[75]
F=[0.01, 0.05]
tau=[3, 27, 45]
St=[0.8]
St_code=0
Spe=[0.5]
Spe_code=0.1
pades=[5000]
a_spe=[500]
a_dpe=[100]
m_bl=[0]
s_bl=[500]
m_spe=[1500]
a_spec=[1e4]
a_time=[1000]
m_time=[0]

p0=[np.concatenate(NQ, F, tau, St, St_code, Spe, Spe_code, pades, a_spe, a_dpe, m_bl, s_bl, m_spe, a_spec, a_time, m_time)]
p1=minimize(L, p0, method='Nelder-Mead', options={'disp':True, 'maxfev':10000})
p1=minimize(L, p1.x, method='Nelder-Mead', options={'disp':True, 'maxfev':10000})
print(p1)
