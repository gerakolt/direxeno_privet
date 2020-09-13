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
    ('T', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('mu', 'f8', 1),
    ('W', 'f8', 1),
    ('F', 'f8', 1),
    ('Tf', 'f8', 1),
    ('Ts', 'f8', 1),
    ('R', 'f8', 1),
    ('a', 'f8', 1),
    ('dl', 'f8', 1),
    ])


Rec[0]=([0.28609523, 0.21198892, 0.1661045 , 0.23595573, 0.2543458 , 0.46767996], [42.43727439, 42.48680044, 42.48223214, 42.61715417, 42.97131299, 42.35603571],
 [1.14722701, 0.82496347, 0.71858647, 1.61434698, 1.48554624, 1.03053529], 2.57341188, 13.7, 0.11035399, 0.94339727, 34.3602973, 0.5760872, 0.36124252, 0.05)

p=minimize(Rec)
