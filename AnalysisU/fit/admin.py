import multiprocessing
import numpy as np
import sys
import time

n=6
def make_glob_array(p):
    Q=p[:n]
    T=p[n:2*n]
    St=p[2*n:3*n]
    Nbg=p[3*n:3*n+2]
    W=p[3*n+2]
    std=p[3*n+3]
    nLXe=p[3*n+4]
    sigma_smr=p[3*n+5]
    mu=p[3*n+6]
    R=p[3*n+7]
    a=p[3*n+8]
    F=p[3*n+9]
    Tf=p[3*n+10]
    Ts=p[3*n+11]

    return Q, T, St, Nbg, W, std, nLXe, sigma_smr, mu, R, a, F, Tf, Ts


def make_iter(N, Q, T, St, nLXe, sigma_smr, R, a, F, Tf, Ts, v):
    for i in range(len(N)):
        yield [N[i], Q, T, St, nLXe, sigma_smr, R, a, F, Tf, Ts, v[:,i], i]


def make_ps():
    n=3*6+2+10
    ps=np.zeros((n+1, n))
    ps[:,:6]=np.random.uniform(0.1, 0.3, size=len(np.ravel(ps[:,:6]))).reshape(np.shape(ps[:,:6]))
    ps[:,6:12]=np.random.uniform(39, 45, size=len(np.ravel(ps[:,:6]))).reshape(np.shape(ps[:,:6]))
    ps[:,12:18]=np.random.uniform(0.5, 1.2, size=len(np.ravel(ps[:,:6]))).reshape(np.shape(ps[:,:6]))
    ps[:,18:20]=np.random.uniform(0.2, 0.6, size=len(np.ravel(ps[:,12:14]))).reshape(np.shape(ps[:,12:14]))
    ps[:,20]=np.random.uniform(13, 16, size=len(np.ravel(ps[:,6]))).reshape(np.shape(ps[:,6]))
    ps[:,21]=np.random.uniform(80, 100, size=len(np.ravel(ps[:,6]))).reshape(np.shape(ps[:,6]))
    ps[:,22]=np.random.uniform(1.55, 1.72, size=len(np.ravel(ps[:,6]))).reshape(np.shape(ps[:,6]))
    ps[:,23]=np.random.uniform(0.01, 1, size=len(np.ravel(ps[:,6]))).reshape(np.shape(ps[:,6]))
    ps[:,24]=np.random.uniform(0.01, 2, size=len(np.ravel(ps[:,6]))).reshape(np.shape(ps[:,6]))
    ps[:,25]=np.random.uniform(0.01, 1, size=len(np.ravel(ps[:,6]))).reshape(np.shape(ps[:,6]))
    ps[:,26]=np.random.uniform(0.0002, 0.005, size=len(np.ravel(ps[:,6]))).reshape(np.shape(ps[:,6]))
    ps[:,27]=np.random.uniform(0.001, 1, size=len(np.ravel(ps[:,6]))).reshape(np.shape(ps[:,6]))
    ps[:,28]=np.random.uniform(1, 10, size=len(np.ravel(ps[:,6]))).reshape(np.shape(ps[:,6]))
    ps[:,29]=np.random.uniform(20, 45, size=len(np.ravel(ps[:,6]))).reshape(np.shape(ps[:,6]))
    return ps
