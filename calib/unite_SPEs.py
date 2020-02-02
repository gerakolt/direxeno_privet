import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt

path='/home/gerak/Desktop/DireXeno/pulser_190803_46211/PMT4/'
SPE=np.zeros(1000)
for filename in os.listdir(path):
    if filename.startswith('SPE'):
        SPE=np.vstack((SPE, np.load(path+filename)['SPE']))
        os.remove(path+filename)
        print(filename)
np.savez(path+'AllSPEs', SPE=SPE[1:])
