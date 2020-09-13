import multiprocessing
import numpy as np
import sys
import time

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

    return Q

def make_iter(N, Q, v):
    for i in range(len(N)):
        np.random.seed(int(i*time.time()%2**32))
        yield [Q, N[i], v[i]]
