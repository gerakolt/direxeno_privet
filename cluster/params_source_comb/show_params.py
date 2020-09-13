from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf, expi
from scipy.special import comb
from fun import fit_ls


source='Co57'
data=np.load('{}.npz'.format(source))
ps=data['ps']
ls=data['ls']
T=data['T']
ps=ps[:len(ls)]
ps=ps[ls<1e8]
ls=ls[ls<1e8]

print(T/len(ls))
print(ps[np.argmin(ls)])
names=['Q0', 'Q1', 'Q2', 'Q3','Q4','Q5', 'T0', 'T1', 'T2', 'T3','T4','T5', 'St0', 'St1', 'St2', 'St3','St4','St5', 'mu', 'W', 'F', 'Tf', 'Ts', 'R', 'a', 'eta']

for i in range(len(ps[0])):
    plt.figure()
    plt.title(names[i])
    plt.plot(ps[:,i], 'ko')


v, mu, a=fit_ls(ls)
fig, ax=plt.subplots(1)
fig.suptitle(source+': '+'L')
ax.plot(ls, 'ko', label='A')
ax.legend()
ax.set_yscale('log')

# ax[1].plot(v, 'ko')
# ax[2].plot(mu, 'ko')
# ax[2].set_yscale('symlog')
#
# ax[3].plot(a, 'ko')
# ax[3].set_yscale('symlog')

plt.show()
