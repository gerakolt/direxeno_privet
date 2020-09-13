import multiprocessing
import numpy as np
import sys
import time

n=6
def make_glob_array(p):
    Q=p[:n]
    Sa=p[n:2*n]
    Nbg=p[2*n:2*n+2]
    W=p[2*n+2]
    std=p[2*n+3]
    return Q, Sa, W, std, Nbg


def make_iter(N, Q, Sa, v, Abins):
    for i in range(len(N)):
        yield [N[i], Q, Sa, v[:,i], i, Abins]
