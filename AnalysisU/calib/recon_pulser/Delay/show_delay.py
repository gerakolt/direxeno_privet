import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import os
from scipy.optimize import curve_fit
import sys
from Sim import Sim

path='/home/gerak/Desktop/DireXeno/190803/pulser/EventRecon/'
data=np.load(path+'delays.npz')
H=data['H']
DC=data['DC']
bins=data['BinsDelay']
names=data['names']

p=[0.     ,    -2.49534478, -2.54319998, -0.84819275, -1.95004707, -2.40912782,
  0.78272572,  0.91402686 , 0.90760073 , 0.70031788 , 0.83687314 , 0.88971754]
T=p[:6]
St=p[6:]

N=10
S=np.zeros((N, 15, len(bins)-1))
for i in range(N):
    print(i)
    S[i]=Sim(T, St, bins)

fig, ax=plt.subplots(3,5,figsize=(20,10))
x=0.5*(bins[1:]+bins[:-1])
for i in range(15):
    np.ravel(ax)[i].step(x, H[i], where='mid', label='Ph')
    np.ravel(ax)[i].step(x, DC[i], where='mid', label='DC')
    np.ravel(ax)[i].legend()
    np.ravel(ax)[i].errorbar(0.5*(bins[1:]+bins[:-1])[np.argmax(H[i])-3:np.argmax(H[i])+4], np.mean(S[:,i,np.argmax(H[i])-3:np.argmax(H[i])+4], axis=0)*np.amax(H[i]), np.std(S[:,i,np.argmax(H[i])-3:np.argmax(H[i])+4], axis=0)*np.amax(H[i]),
        fmt='.', color='k')

# plt.legend(fontsize=35)
plt.xlabel('Delay [ns]', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.show()
