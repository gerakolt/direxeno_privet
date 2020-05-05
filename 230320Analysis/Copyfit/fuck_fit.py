import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf
import sys
from scipy.optimize import minimize
from fuck_fun import get_data, make_P, Int, make_z, model2, model3


dt=29710
dt_BG=48524

lim_fit=[400]
pmts=[0, 8]
Hs, N_events, Hs_BG, Ns, Hs_spec, Hs_spec_BG=get_data(pmts, 'Co57')

ns=np.arange(len(Hs[0]))
NQ=[50]
F=0.1
Tf=5
Ts=45
St=[0.8]
Spe=[0.5]
P0=[0.01, 0.1, 0.25]

i=0
x=np.arange(1000)/5
fig, ax1 = plt.subplots(1, 1, sharex=True, sharey='row')
for p0 in P0:
    P=make_P(Spe[i], p0)
    p=[NQ[i], F, Tf, Ts, St[i]]
    h=model2(p, P, len(ns))

    ax1.plot(x, np.mean(h.T*ns, axis=1), '.', label='fit {}'.format(p0))
    ax1.legend()
plt.show()
