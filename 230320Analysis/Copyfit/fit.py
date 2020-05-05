import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf
import sys
from scipy.optimize import minimize
from fun import get_data, make_P, Int, make_z, model2, model3, model_spec
from scipy.optimize import curve_fit


dt=29710
dt_BG=48524

lim_fit=[400]
pmts=[0]
Hs, N_events, Hs_BG, Ns, Hs_spec, Hs_spec_BG=get_data(pmts, 'Co57')

ns=np.arange(len(Hs[0]))
NQ=[58, 50]
F=0.1
Tf=5
Ts=45
St=[0.8,0.8]
Spe=[1.3,0.5]
a=[25000]

for i, pmt in enumerate(pmts):
    P=make_P(0.25, 0.25)
    p=[NQ[i], F, Tf, Ts, St[i]]
    # h=model2(p, P, len(ns))
    h_spec=model_spec(Ns[i], NQ[i], P, a[i])

    x=np.arange(1000)/5
    fig, ((ax1), (ax2)) = plt.subplots(1, 2)
    ax1.plot(x, np.mean(Hs[i].T*ns, axis=1), 'k.', label='data')
    # ax1.plot(x, dt*np.mean(Hs_BG[i].T*ns, axis=1)/dt_BG, 'b.', label='BG')
    ax1.plot(x, np.mean(Hs[i].T*ns, axis=1)-np.mean(dt*Hs_BG[i].T*ns/dt_BG, axis=1), 'g.', label='data - BG')
    ax1.plot(x, np.mean((Hs[i].T-dt*Hs_BG[i].T/dt_BG)*ns, axis=1), 'g.', label='data - BG')
    # ax1.plot(x, N_events[i]*np.mean(h.T*ns, axis=1), 'r.', label='fit')
    ax1.legend()

    ax2.plot(Ns[i], Hs_spec[i], 'r.', label='number of PEs')
    ax2.plot(Ns[i], Hs_spec_BG[i], 'k.', label='number of PEs BG')
    ax2.plot(Ns[i], h_spec, 'b.', label='model')

    ax2.legend()
    plt.show()
