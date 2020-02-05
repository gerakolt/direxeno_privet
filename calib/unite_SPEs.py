import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt

path='/home/gerak/Desktop/DireXeno/pulser_190803_46211/PMT4/'
SPE=np.zeros(1000)
rec=np.recarray(0, dtype=[
    ('id', 'i8'),
    ('init10', 'i8'),
    ])
for filename in os.listdir(path):
    if filename.startswith('SPE'):
        Data=np.load(path+filename)
        SPE=np.vstack((SPE, Data['SPE']))
        rec=np.append(rec, Data['rec'])
        os.remove(path+filename)
        print(filename)
np.savez(path+'AllSPEs', SPE=SPE[1:], rec=rec)
