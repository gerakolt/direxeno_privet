import numpy as np
import time
import os
import sys
from scipy.stats import poisson, binom
from scipy.special import erf as erf
from minimize import minimize
import multiprocessing

Q=[0.15, 0.17, 0.13, 0.19, 0.11, 0.21, 0.09, 0.23, 0.07, 0.25]
i=int(sys.argv[1])
pmt=i//len(Q)
ind=i%len(Q)
q=Q[ind]
minimize(pmt, q, i)
