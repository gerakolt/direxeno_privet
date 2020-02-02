import numpy as np

def func(j):
    while j>0:
        yield j
        j-=1

for j in range(5,10):
    for k in func(j):
        print(j,k)
