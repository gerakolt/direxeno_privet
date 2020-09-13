import multiprocessing
import numpy as np
import sys
import time

n=6
def make_glob_array(p):
    Q=p[:n]
    nLXe=p[n]
    mu=p[n+1]
    W=p[n+2]
    return Q, nLXe, mu, W

def make_iter(N, Q, v, nLXe):
    for i in range(len(N)):
        yield [Q, nLXe, N[i], v[i], i]
