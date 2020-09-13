from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf, expi
from scipy.special import comb

# source='Co57Q0_{}'
source='../../cluster/arrays/Co57Q0{}'
Q0=[]
L=[]
STD=[]
lmin=1e10
for c in ['A', 'B', 'C', 'D', 'E', 'F']:
    data=np.load('{}.npz'.format(source.format(c)))
    ps=data['ps']
    ls=data['ls']
    ps=ps[:len(ls)]

    if np.amin(ls)<lmin:
        lmin=np.amin(ls)
        p=ps[np.argmin(ls)]

    Q0.append(data['param'])
    L.append(np.mean(np.sort(ls)[0]))
    STD.append(np.std(np.sort(ls)[:20]))

    plt.figure()
    plt.title(Q0[-1])
    plt.plot(ls, 'k.')
    plt.hlines(y=L[-1], xmin=0, xmax=len(ls), color='r')
    plt.fill_between(np.arange(0, len(ls)), y1=L[-1]-STD[-1], y2=L[-1]+STD[-1], color='y', alpha=1)
    plt.yscale('log')

plt.figure()
plt.title('L')
plt.plot(Q0, np.exp(-np.array(L)/np.amin(L)), 'k.')

plt.figure()
plt.title('-log(L)')
plt.plot(Q0, L, 'k.')
print(p, lmin)
plt.show()
