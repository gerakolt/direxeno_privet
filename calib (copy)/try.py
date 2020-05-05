import numpy as np
import itertools


N=np.arange(5)
for n,m in itertools.product(N,N):
    print(n,m)
