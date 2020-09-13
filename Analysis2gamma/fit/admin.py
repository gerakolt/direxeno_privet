import multiprocessing
import numpy as np
import sys
import time
from memory_profiler import profile

# Rec=np.recarray(5000, dtype=[
#     ('Q', 'f8', len(pmts)),
#     ('T', 'f8', len(pmts)),
#     ('St', 'f8', len(pmts)),
#     ('mu', 'f8', 1),
#     ('N', 'f8', 1),
#     ('F', 'f8', 1),
#     ('Tf', 'f8', 1),
#     ('Ts', 'f8', 1),
#     ('R', 'f8', 1),
#     ('a', 'f8', 1),
#     ('eta', 'f8', 1),
#     ])

n=6
def make_glob_array(p):
    Q=multiprocessing.Array('d', p[:n])
    T=multiprocessing.Array('d', p[n:2*n])
    St=multiprocessing.Array('d', p[2*n:3*n])
    mu=multiprocessing.Array('d', [p[3*n]])
    W=multiprocessing.Array('d', [p[3*n+1]])
    g=multiprocessing.Array('d', [p[3*n+2]])
    F=multiprocessing.Array('d', [p[3*n+3]])
    Tf=multiprocessing.Array('d', [p[3*n+4]])
    Ts=multiprocessing.Array('d', [p[3*n+5]])
    R=multiprocessing.Array('d', [p[3*n+6]])
    a=multiprocessing.Array('d', [p[3*n+7]])

    return Q, T, St, mu, W, g, F, Tf, Ts, R, a

def make_iter(N, Q, T, St, F, Tf, Ts, R, a, v):
    for i in range(len(N)):
        np.random.seed(int(i*time.time()%2**32))
        yield [Q, T, St, N[i], F, Tf, Ts, R, a, v[i]]
