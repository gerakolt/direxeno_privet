import multiprocessing
import numpy as np
import sys
import time

n=6
def make_glob_array(p):
    Q=p[:n]
    W=p[n]
    mu=p[n+1]
    nLXe=p[n+2]
    sigma_smr=p[n+3]
    return Q, W, mu, nLXe, sigma_smr


def make_iter(N, Q, nLXe, sigma_smr, v):
    for i in range(len(N)):
        yield [Q, nLXe, sigma_smr, N[i], v[i], i]
