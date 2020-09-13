from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf, expi
from scipy.special import comb
import os

# source='Co57Q0_{}'
path='../../cluster/arrays/'

for pmt in range(6):
    lmin=1e10
    Q=[]
    L=[]
    STD=[]
    for filename in os.listdir(path):
        if filename.startswith("Q{}".format(pmt)):
            plt.figure()
            data=np.load(path+filename)
            ps=data['ps']
            ls=data['ls']
            ps=ps[:len(ls)]
            # print(filename,':', data['T']/len(ls), 'sec per event')
            plt.plot(ls, 'ko')

            if np.amin(ls)<lmin:
                lmin=np.amin(ls)
                p=ps[np.argmin(ls)]
            Q.append(data['param'])
            L.append(np.mean(ls[-20:]))
            STD.append(np.std(ls[-20:]))
            plt.fill_between(np.arange(0, len(ls)), y1=L[-1]-STD[-1], y2=L[-1]+STD[-1], color='y', alpha=1)
            plt.hlines(y=L[-1], xmin=0, xmax=len(ls), color='k')
            plt.yscale('log')


    plt.figure()
    plt.title('L (PMT{})'.format(pmt))
    plt.plot(Q, np.exp(-np.array(L)/lmin), 'ko', label='A')
    plt.errorbar(Q, np.exp(-np.array(L)/lmin), np.exp(-np.array(L)/lmin)*(STD/lmin), fmt='k.')
    plt.legend()
    print(p, lmin)
    # plt.show()
