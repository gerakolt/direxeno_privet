import numpy as np
import time
import os
import sys
from scipy.stats import poisson, binom
from scipy.special import erf as erf
from minimize import minimize
import multiprocessing

pmts=[0,1,4,7,8,14]


Rec=np.recarray(1, dtype=[
    ('Q', 'f8', len(pmts)),
    ])

Rec[0]=([0,0,0,0,0,0],)
ind=1

for i, q in np.linspace(0.2, 0.3, 5):
    p=minimize(Rec, q, ind, 'Q{}_{}'.format(ind, i))
