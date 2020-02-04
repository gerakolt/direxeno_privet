import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt

path='/home/gerak/Desktop/DireXeno/030220/pulser3/Peaks/'
rec=[]
for filename in os.listdir(path):
    if filename.startswith('peaks'):
        rec.extend(np.load(path+filename)['rec'])
        os.remove(path+filename)
        print(filename)
np.savez(path+'AllPeaks', rec=rec)
