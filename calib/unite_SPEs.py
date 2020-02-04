import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt

path='/home/gerak/Desktop/DireXeno/030220/pulser3/PMT19/'
SPE=np.zeros(1000)
init10=np.array([])
for filename in os.listdir(path):
    if filename.startswith('SPE'):
        Data=np.load(path+filename)
        SPE=np.vstack((SPE, Data['SPE']))
        init10=np.append(init10, Data['init10'])
        os.remove(path+filename)
        print(filename)
np.savez(path+'AllSPEs', SPE=SPE[1:], init10=init10)
