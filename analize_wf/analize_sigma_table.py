import matplotlib.pyplot as plt
import os, sys
import numpy as np
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
import itertools

path='/home/gerak/Desktop/DireXeno/190803/pulser/'
pmts=[0,1,2,3,4,5,6,7,8,9,10,11,17,14,15,16,18,19]
Sspe=np.load(path+'Sigma_table.npz')['S']

path='/home/gerak/Desktop/DireXeno/190803/Co57/'
SAll=np.load(path+'Sigma_table.npz')['S']

for i in range(len(pmts)):
    spe=[]
    sAll=[]
    for j in range(len(pmts)-1):
        if not j==i:
            for k in range(j+1, len(pmts)):
                if j>i:
                    spe.append(np.sqrt(Sspe[i,j]**2+Sspe[i,k]**2-Sspe[j,k]**2))
                    sAll.append(np.sqrt(SAll[i,j]**2+SAll[i,k]**2-SAll[j,k]**2))
                elif j<i and i<k:
                    spe.append(np.sqrt(Sspe[j,i]**2+Sspe[i,k]**2-Sspe[j,k]**2))
                    sAll.append(np.sqrt(SAll[j,i]**2+SAll[i,k]**2-SAll[j,k]**2))
                else:
                    spe.append(np.sqrt(Sspe[j,i]**2+Sspe[k,i]**2-Sspe[j,k]**2))
                    sAll.append(np.sqrt(SAll[j,i]**2+SAll[k,i]**2-SAll[j,k]**2))

    plt.figure()
    plt.title('PMT{}'.format(pmts[i]))
    plt.hist(spe, bins=10, range=[0,2], histtype='step', label='SPE', linewidth=5)
    plt.hist(sAll, bins=10, range=[0,2], histtype='step', label='All Co57 events', linewidth=5)
    plt.legend()
    plt.show()
