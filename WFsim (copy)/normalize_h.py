import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import poisson, binom
from classes import WaveForm, Hit
from fun import find_hits, Recon_WF


height_cut=35
# height_cut=0
pmt=1
events=5000
N=50
tau=45
St=0.7

Data=np.load('/home/gerak/Desktop/DireXeno/190803/pulser/SPEs{}.npz'.format(pmt))
SPEs=Data['SPE']
SPEs=SPEs[np.nonzero(np.logical_and(np.amin(SPEs, axis=1)<-height_cut, np.amin(SPEs, axis=1)>-70))[0]]
zeros=Data['zeros']
SPEs[:,zeros]=0
factor=Data['factor']
mean_height=Data['mean_height']

h=[]
for i in range(50000):
    if i%100==0:
        print(i)
    I=np.random.randint(0,len(SPEs),N)
    spe=np.mean(SPEs[I], axis=0)
    h.append(-np.amin(spe))

plt.hist(h, bins=100)
plt.axvline(x=mean_height)
plt.show()
