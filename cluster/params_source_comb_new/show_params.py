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

source='../../AnalysisR/fit/Co57'
# source='Cs137B2'
data=np.load('{}.npz'.format(source))
ps=data['ps']
ls=data['ls']
T=data['T']
ps=ps[:len(ls)]

ps0=ps[ls>1e8]
ls0=ls[ls>1e8]

ps=ps[ls<1e5]
ls=ls[ls<1e5]
print(source)
print(data['note'])
print(np.amin(ls))
print(ps[np.argmin(ls)])

names=['Q0', 'Q1', 'Q2', 'Q3','Q4','Q5', 'w']

for i in range(len(ps[0])):
    plt.figure()
    plt.title(names[i]+'({})'.format(i))
    plt.plot(ps[:,i], 'ko')
    plt.plot(np.argmin(ls), ps[np.argmin(ls),i], 'ro')
    # plt.plot(ps0[:,i], 'ro')



v, mu, a=fit_ls(ls)
fig, ax=plt.subplots(1)
fig.suptitle(source+': '+'L')
ax.plot(ls, 'ko', label='A')
# ax.plot(ls0, 'ro', label='A')
ax.legend()
ax.set_yscale('log')

plt.show()
