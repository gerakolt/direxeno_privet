from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf, expi
from scipy.special import comb

path='../../cluster/arrays/'
path=''
data=np.load('Q-1_0.npz')
ps=data['ps']
ls=data['ls']
T=data['T']
print('Time for iter:', T/len(ls))
ps=ps[:len(ls)]

ps0=ps[ls>1e8]
ls0=ls[ls>1e8]

ps=ps[ls<1e7]
ls=ls[ls<1e7]
print(data['note'])
print(np.amin(ls))
print(ps[np.argmin(ls)])

names=['Q0', 'Q1', 'Q2', 'Q3','Q4','Q5', 'T0', 'T1', 'T2', 'T3','T4','T5', 'ST0', 'ST1', 'ST2', 'ST3','ST4','ST5', 'NbgA', 'NbgB', 'W', 'std', 'nLXe', 'sigma_smr', 'mu', 'R', 'a', 'F', 'Tf', 'Ts']


for i in range(len(ps[0])):
    plt.figure()
    plt.title(names[i]+'({})'.format(i))
    plt.plot(ps[:,i], 'ko')
    plt.plot(np.argmin(ls), ps[np.argmin(ls),i], 'ro')
    # plt.plot(ps0[:,i], 'ro')



fig, ax=plt.subplots(1)
ax.plot(ls, 'ko', label='A')
# ax.plot(ls0, 'ro', label='A')
ax.legend()
ax.set_yscale('log')

plt.show()
