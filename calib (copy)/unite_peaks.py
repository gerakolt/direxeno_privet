import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt

# path='/home/gerak/Desktop/DireXeno/pulser_190803_46211/Peaks/'
path='../../../850V_00000/Peaks/'
rec=[]
for filename in os.listdir(path):
    if filename.startswith('peaks'):
        rec.extend(np.load(path+filename)['rec'])
        os.remove(path+filename)
        print(filename)
np.savez(path+'AllPeaks', rec=rec)
