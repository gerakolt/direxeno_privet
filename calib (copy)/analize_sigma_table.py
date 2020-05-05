import matplotlib.pyplot as plt
import os, sys
import numpy as np
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
import itertools

path='/home/gerak/Desktop/DireXeno/190803/pulser/'
pmts=[0,1,2,3,4,5,6,7,8,9,10,11,17,14,15,16,18,19]
S=np.load(path+'Sigma_table.npz')['S']

for i in range(len(pmts)):
    s=[]
    for j in range(len(pmts)-1):
        if not j==i:
            for k in range(j+1, len(pmts)):
                if j>i:
                    s.append(np.sqrt(S[i,j]**2+S[i,k]**2-S[j,k]**2))
                elif j<i and i<k:
                    s.append(np.sqrt(S[j,i]**2+S[i,k]**2-S[j,k]**2))
                else:
                    s.append(np.sqrt(S[j,i]**2+S[k,i]**2-S[j,k]**2))
    plt.figure()
    plt.title('PMT{}'.format(pmts[i]))
    plt.hist(s, bins=10, range=[0,5], histtype='step', label='SPE', linewidth=5)
    plt.legend()
    plt.show()
