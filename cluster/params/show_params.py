from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf, expi
from scipy.special import comb

# data=np.load('Rec_Co570.npz')
# Rec=data['Rec']
# ls=data['ls']
# T=data['time']
#
# Rec=Rec[:len(ls)]
# ls=ls[Rec['N']>2000]
# Rec=Rec[Rec['N']>2000]

source='Co57'
data=np.load('Rec_{}.npz'.format(source))
Rec1=data['Rec']
ls1=data['ls']
T1=data['time']
Rec1=Rec1[:len(ls1)]

data=np.load('Rec_{}B.npz'.format(source))
Rec=data['Rec']
ls=data['ls']
T=data['time']
Rec=Rec[:len(ls)]

print(source)
print(Rec1[np.argmin(ls1)])
print('B', Rec[np.argmin(ls)])


print('Number of iterations:', len(ls1), len(ls), 'Time in sec:', T1, T, '({} min for iter)'.format((T1+T)/((len(ls1)+len(ls))*60)))
names=Rec1.dtype.names
for name in names:
    try:
        for i in range(len(Rec1[name][0])):
            plt.figure()
            plt.title(source+': '+name+'{}'.format(i))
            # plt.plot(Rec[name][:,i], 'ko')
            plt.plot(np.arange(len(ls1)),Rec1[name][:,i], 'ro', label='A')
            plt.plot(np.arange(len(ls)),Rec[name][:,i], 'go', label='B')
            plt.legend()

    except:
        if name=='N':
            plt.figure()
            plt.title(source+': '+'W')
            plt.plot(np.arange(len(ls1)), 1000*662/Rec1[name], 'ro', label='A')
            plt.plot(np.arange(len(ls)), 1000*662/Rec[name], 'go', label='B')
            plt.legend()

        else:
            plt.figure()
            plt.title(source+': '+name)
            plt.plot(np.arange(len(ls1)), Rec1[name], 'ro', label='A')
            plt.plot(np.arange(len(ls)), Rec[name], 'go', label='B')
            plt.legend()


plt.figure()
plt.title(source+': '+'L')
# plt.plot(ls, 'ko')
plt.plot(np.arange(len(ls1)), ls1, 'ro', label='A')
plt.plot(np.arange(len(ls)), ls, 'go', label='B')
plt.legend()
plt.yscale('log')

plt.show()
