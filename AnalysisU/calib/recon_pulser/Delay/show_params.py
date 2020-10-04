from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf, expi
from scipy.special import comb


path=''

data=np.load('Q{}_{}.npz'.format(-1, 0))
ps=data['ps']
ls=data['ls']
T=data['T']
print('Time for iter:', T/len(ls), '({} iterations)'.format(len(ls)))
ps=ps[:len(ls)]
# ps=ps[ls<3e3]
# ls=ls[ls<3e3]
print(data['note'])
print(np.amin(ls))
print(ps[np.argmin(ls)])

names=['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'St0', 'St1', 'St2', 'St3', 'St4', 'St5']


for i in range(len(ps[0])):
    plt.figure()
    plt.title(names[i]+'({})')
    plt.plot(ps[:,i], 'ko')



fig, ax=plt.subplots(1)
ax.plot(ls, 'ko', label='A')

ax.legend()
ax.set_yscale('log')

plt.show()
