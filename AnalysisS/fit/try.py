import numpy as np
import sys

N=1
mu=50
x1=0
x2=1
vs=np.zeros((3, N))
count=0
while count<N:
    d=np.random.exponential(mu, N-count)
    X1=d[np.nonzero(d<0.5)[0]]-0.25
    print(d)
    print(X1)
    r=np.sqrt(np.random.uniform(0, 0.25**2-X1**2))
    print(r)
    phi=np.random.uniform(0, 2*np.pi, len(r))
    print(phi)
    v=np.zeros((3, len(r)))
    v[x1]=X1
    v[x2]=r*np.cos(phi)
    v[2]=r*np.sin(phi)
    print(v)
    vs[:,count:count+len(r)]=v
    count+=len(r)
    sys.exit()
