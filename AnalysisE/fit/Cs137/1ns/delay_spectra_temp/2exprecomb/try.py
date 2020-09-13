import numpy as np
import matplotlib.pyplot as plt
import time
import os
from fun import Sim, q0_model, make_P, model_area, Sim2, make_3D
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf
import warnings
from minimize import minimize, make_ps


pmts=[0,1,4,7,8,14]
N=60*662
rec=np.recarray(1, dtype=[
    ('Q', 'f8', len(pmts)),
    ('T', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('F', 'f8', 1),
    ('Tf', 'f8', 1),
    ('Ts', 'f8', 1),
    ('R', 'f8', 1),
    ('b', 'f8', 1),
    ])


path='/home/gerak/Desktop/DireXeno/190803/pulser/DelayRecon/'
delay_hs=[]
names=[]
delays=[]
for i in range(len(pmts)-1):
    for j in range(i+1, len(pmts)):
        data=np.load(path+'delay_hist{}-{}.npz'.format(pmts[i], pmts[j]))
        delays.append(data['x']-data['m'])
        delay_hs.append(data['h'])
        names.append('{}_{}'.format(pmts[i], pmts[j]))

T_CsB=1564825612162-1564824285761
T_BG=1564874707904-1564826183355
path='/home/gerak/Desktop/DireXeno/190803/Cs137B/EventRecon/'
data=np.load(path+'H.npz')
H=data['H']
G=data['G']
spectrum=data['spectrum']
spectra=data['spectra']
left=data['left']
right=data['right']
t=np.arange(200)
dt=t[1]-t[0]


rec[0]=([0.15031393,  0.10126694,  0.0980743,   0.15169108,  0.13757894,  0.27692865],
 [42.6965627,  42.79534384, 42.98685503, 42.85373486, 42.54194199, 42.92884848],
  [0.85148873,  0.82144334,  0.75498879,  0.84165176,  1.09559689,  0.82225653],
  0.10919454,  1.65475904, 32.72410862,  0.51185353,  5.27711599)


m=make_3D(t, N, rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['b'][0], rec['Q'][0], rec['T'][0], rec['St'][0])
s, GS, GS_spectrum, Gtrp, Gsng, GRtrp, GRsng=Sim(t, N, rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['b'][0], rec['Q'][0], rec['T'][0], rec['St'][0])

x=np.arange(200)
fig, (ax1)=plt.subplots(1,1)
ax1.plot(x, np.sum(G.T*np.arange(np.shape(G)[0]), axis=1), 'ko', label='Global Data')
ax1.plot(x, np.sum(G[:,0])*np.sum(GS.T*np.arange(np.shape(GS)[0]), axis=1), 'r-.', label='Global 2 exp sim')
ax1.plot(x, np.sum(G[:,0])*np.sum(Gtrp.T*np.arange(np.shape(GS)[0]), axis=1), 'g-.', label='triplet')
ax1.plot(x, np.sum(G[:,0])*np.sum(Gsng.T*np.arange(np.shape(GS)[0]), axis=1), 'b-.', label='singlet')
ax1.plot(x, np.sum(G[:,0])*np.sum(GRtrp.T*np.arange(np.shape(GS)[0]), axis=1), 'y-.', label='Recombination triplet')
ax1.plot(x, np.sum(G[:,0])*np.sum(GRsng.T*np.arange(np.shape(GS)[0]), axis=1), 'c-.', label='Recombination singlet')
ax1.legend()

plt.show()
