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
    ('a_pad', 'f8', len(pmts)),
    ('a_spe', 'f8', len(pmts)),
    ('a_dpe', 'f8', len(pmts)),
    ('a_trpe', 'f8', len(pmts)),
    ('m_pad', 'f8', len(pmts)),
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

rec[0]=([35.03317743, 33.31702128], [35.60001319, 35.58852242], [0.2, 0.2], [0.01570876, 0.01053329], [6.22176383e-05, 2.99333641e-05], [0.02490614, 0.07321714], [0.26908217, 0.24259374],
 [0.3, 0.3], [74430.94826317, 73339.58169683], [6621.14942803, 6510.3196783 ], [ 345, 345.0851499 ], [2.73044512, 0.57923595], [0.00293776, 0.00964629], 0.01675983, 2.08735154, 35.30020979, 260.73705879)

t=Sim(rec['NQ'][0], rec['T'][0], 1, rec['R'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0], rec['q0'][0], rec['a0'][0], rec['Spad'][0], rec['Spe'][0], rec['m_pad'][0])
m=Model(rec['NQ'][0], rec['T'][0], rec['R'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0], rec['q0'][0], make_P(rec['a0'][0], rec['Spad'][0], rec['Spe'][0], rec['m_pad'][0]))

x=np.arange(1000)/5
fig, (ax1, ax2)=plt.subplots(2,1, sharex=True)



ax1.plot(x[:30], np.mean(t[:,:30,0].T*np.arange(np.shape(t)[0]), axis=1), '.-', label='sim', linewidth=3)
ax1.plot(x[:30], np.mean(m[:,:30,0].T*np.arange(np.shape(t)[0]), axis=1), '.-', label='model', linewidth=3)


ax2.plot(x[:30], np.mean(t[:,:30,1].T*np.arange(np.shape(t)[0]), axis=1), '.-', label='sim', linewidth=3)
ax2.plot(x[:30], np.mean(m[:,:30,1].T*np.arange(np.shape(t)[0]), axis=1), '.-', label='model', linewidth=3)


# ax1.plot(t[:,0,0], 'k.', label='sim')
# ax1.plot(m[:,0,0], 'r.', label='model')
#
# ax2.plot(t[:,0,1], 'k.', label='sim')
# ax2.plot(m[:,0,1], 'r.', label='model')

ax1.legend()
ax2.legend()
plt.subplots_adjust(hspace=0)
plt.show()
