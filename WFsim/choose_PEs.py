import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import poisson, binom
from classes import WaveForm, Hit
from fun import find_hits


pmt=0
events=5000
N=50
tau=45
St=0.7

Data=np.load('/home/gerak/Desktop/DireXeno/190803/pulser/SPEs{}.npz'.format(pmt))
SPEs=Data['SPE']
spe=np.mean(SPEs, axis=0)
spe=(spe-np.median(spe[:150]))
h=[]
for i in range(1000):
    I=np.random.randint(0,len(SPEs),N)
    spes=np.mean(SPEs[I], axis=0)
    plt.figure()
    plt.plot(spes, 'k.-', label='spes')
    plt.plot(spe, 'r.-', label='spe')
    plt.show()
