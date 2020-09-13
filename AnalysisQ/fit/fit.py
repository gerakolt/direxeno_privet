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
    ('w', 'f8'),
    ('mu', 'f8')
    ])
#
# Rec[0]=([0,0,0,0,0,0], 13.7)
#
# p=minimize(Rec)

minimize()
