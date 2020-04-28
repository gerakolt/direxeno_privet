import numpy as np
import matplotlib.pyplot as plt
import time
import os
from fun import Model, Sim, make_P
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf
import warnings

pmts=[0,1]

rec=np.recarray(1, dtype=[
    ('NQ', 'f8', len(pmts)),
    ('T', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('R', 'f8', len(pmts)),
    ('q0', 'f8', len(pmts)),
    ('a0', 'f8', len(pmts)),
    ('Spad', 'f8', len(pmts)),
    ('Spe', 'f8', len(pmts)),
    ('F', 'f8', 1),
    ('Tf', 'f8', 1),
    ('Ts', 'f8', 1),
    ('a_delay', 'f8', 1),
    ])

# rec['NQ'][0]=[36, 36]
# rec['T'][0]=[40, 30]
# rec['St'][0]=[0.8, 0.8]
# rec['R'][0]=[0.001, 0.001]
# rec['q0'][0]=[0.05, 0.05]
# rec['a0'][0]=[0.2, 0.2]
# rec['Spad'][0]=[0.2, 0.2]
# rec['Spe'][0]=[0.5, 0.5]
# rec['F'][0]=0.2
# rec['Tf'][0]=5
# rec['Ts'][0]=45

rec[0]=([42.13834412, 43.75971207], [38.35234205, 34.49094268], [0.86903185, 1.18155874], [0.06315727, 0.05249282], [9.61043168e-05, 4.95600863e-04], [0.19166489, 0.18150454], [0.30216196, 0.21559156], [0.50952239, 0.6087814 ],
 0.33371173, 7.67107787, 36.2697553, 500.79964305)

t=Sim(rec['NQ'][0], rec['T'][0], 1, rec['R'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0], rec['q0'][0], rec['a0'][0], rec['Spad'][0], rec['Spe'][0])
m=Model(rec['NQ'][0], rec['T'][0], rec['R'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0], rec['q0'][0], make_P(rec['a0'][0], rec['Spad'][0], rec['Spe'][0]))

x=np.arange(1000)/5
fig, (ax1, ax2)=plt.subplots(2,1, sharex=True)



ax1.plot(x, np.mean(t[:,:,0].T*np.arange(np.shape(t)[0]), axis=1), '.-', label='sim', linewidth=3)
ax1.plot(x, np.mean(m[:,:,0].T*np.arange(np.shape(t)[0]), axis=1), '.-', label='model', linewidth=3)


ax2.plot(x, np.mean(t[:,:,1].T*np.arange(np.shape(t)[0]), axis=1), '.-', label='sim', linewidth=3)
ax2.plot(x, np.mean(m[:,:,1].T*np.arange(np.shape(t)[0]), axis=1), '.-', label='model', linewidth=3)


# ax1.plot(t[:,0,0], 'k.', label='sim')
# ax1.plot(m[:,0,0], 'r.', label='model')
#
# ax2.plot(t[:,0,1], 'k.', label='sim')
# ax2.plot(m[:,0,1], 'r.', label='model')

ax1.legend()
ax2.legend()
plt.subplots_adjust(hspace=0)
plt.show()
