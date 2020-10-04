from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf, expi
from scipy.special import comb

pmt=5
path='../../cluster/arrays/'
path=''

data=np.load('Q{}_{}.npz'.format(pmt, 0))
ps=data['ps']
ls=data['ls']
T=data['T']
print('Time for iter:', T/len(ls), '({} iterations)'.format(len(ls)))
ps=ps[:len(ls)]
ps=ps[ls<3e3]
ls=ls[ls<3e3]
print(data['note'])
print(np.amin(ls))
print(ps[np.argmin(ls)])

data=np.load('Q{}_{}.npz'.format(pmt, 1))
ps1=data['ps']
ls1=data['ls']
T=data['T']
print('Time for iter:', T/len(ls), '({} iterations)'.format(len(ls)))
ps1=ps1[:len(ls1)]
# ps1=ps1[ls1<3e3]
# ls1=ls1[ls1<3e3]
print(data['note'])
print(np.amin(ls1))
print(ps1[np.argmin(ls1)])

names=['Ma', 'Sa', 'q']


for i in range(len(ps[0])):
    plt.figure()
    plt.title(names[i]+'({})')
    plt.plot(ps[:,i], 'ko')
    plt.plot(ps1[:,i], 'ro')
    #plt.plot(np.argmin(ls), ps[np.argmin(ls),i], 'ro')
        # plt.plot(ps0[:,i], 'ro')



fig, ax=plt.subplots(1)
fig.suptitle('PMT{}'.format(pmt))
ax.plot(ls, 'ko', label='A')
ax.plot(ls1, 'ro', label='B')

ax.legend()
ax.set_yscale('log')

plt.show()
