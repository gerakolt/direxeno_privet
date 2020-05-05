import numpy as np

def fuck(N):
    a=10
    b=10
    while N>0:
        N-=1
        a+=1
        b-=2
        yield [a, b]
N=5
for i, [a,b] in enumerate(fuck(N)):
    print(i,a, b)
