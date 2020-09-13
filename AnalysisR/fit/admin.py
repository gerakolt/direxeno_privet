import multiprocessing
import numpy as np
import sys
import time

n=6
def make_glob_array(p):
    Q=p[:n]
    return Q

def make_iter(N, Q, v):
    for i in range(len(N)):
        yield [Q, N[i], v[i], i]
