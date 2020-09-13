import multiprocessing
import numpy as np
import sys
import time

n=6
def make_glob_array(p):
    Q=p[:n]
    Sa=p[n:2*n]
    W=p[2*n]
    g=p[2*n+1]
    return Q, Sa, W, g


def make_iter(N, Q, Sa, x1, x2):
    for i in range(len(N)):
        yield [x1, x2, Q, Sa, N[i], i]
