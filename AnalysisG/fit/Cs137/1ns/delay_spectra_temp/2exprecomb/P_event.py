import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf
from fun import make_3D



pmts=[0,1,4,7,8,14]

N=60*662
t=np.arange(200)
rec=np.recarray(1, dtype=[
    ('Q', 'f8', len(pmts)),
    ('T', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('F', 'f8', 1),
    ('Tf', 'f8', 1),
    ('Ts', 'f8', 1),
    ('R', 'f8', 1),
    ('a', 'f8', 1),
    ('eta', 'f8', 1),
    ])

rec[0]=([0.21537128,  0.14399147,  0.13457826,  0.20935095,  0.1962309,   0.37744515],
 [41.56126284, 41.65504493, 42.20686561, 43.39193013, 41.63506245, 42.1523902],
 [ 0.74358323,  1.11303201,  1.34520329,  1.95492766,  0.94264498,  1.10002631],
  0.10859021,  1.97690138, 39.970927,    0.57290201,  0.33058596,  0.39326104)

m=make_3D(t, N, rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0], rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0])


path='/home/gerak/Desktop/DireXeno/190803/Cs137B/EventRecon/'
rec=np.load(path+'recon1ns99992.npz')['rec']
blw_cut=15
init_cut=20
chi2_cut=5000
left=700
right=1000

rec=rec[np.all(rec['init_wf']>20, axis=1)]
rec=rec[np.sqrt(np.sum(rec['blw']**2, axis=1))<blw_cut]
rec=rec[np.sqrt(np.sum(rec['chi2']**2, axis=1))<chi2_cut]
init=np.sum(np.sum(rec['h'][:,:10,:], axis=2), axis=1)
full=np.sum(np.sum(rec['h'], axis=2), axis=1)
plt.figure()
plt.hist(init/full, bins=100)
plt.show()
plt.figure()
plt.hist((init/full)[init/full<0.5], bins=100)
plt.show()
rec=rec[init/full<0.5]
spectrum=np.histogram(np.sum(np.sum(rec['h'], axis=1), axis=1), bins=np.arange(1000)-0.5)[0]
rec=rec[np.sum(np.sum(rec['h'], axis=1), axis=1)>left]
rec=rec[np.sum(np.sum(rec['h'], axis=1), axis=1)<right]

p=np.ones(len(rec['h'][:,0,0]))
for i, h in enumerate(rec['h']):
    print(i)
    if np.all(h<30):
        for j in range(100):
            for k in range(len(pmts)):
                if m[h[j,k], j, k]==0:
                    print(h[j,k], j, k)
                    plt.figure()
                    plt.bar(t, np.sum(h, axis=1))
                    plt.show()
                p[i]+=m[h[j,k], j, k]
    if p[i]<10:
        plt.figure()
        plt.bar(t, np.sum(h, axis=1))
        init=np.sum(h[:10,:])
        full=np.sum(h)
        print(init/full)
        plt.show()
np.savez('P_event', p=p)
# p=np.load('P_event.npz')['p']
plt.hist(p[p>10], bins=100)
plt.show()
