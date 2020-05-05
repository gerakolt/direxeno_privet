import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt

<<<<<<< HEAD
path='/home/gerak/Desktop/DireXeno/030220/pulser3/Peaks/'
=======
# path='/home/gerak/Desktop/DireXeno/pulser_190803_46211/Peaks/'
path='../../../850V_00000/Peaks/'
>>>>>>> c7766bd9473ef15b7aa60790e5624a9c78c984b2
rec=[]
for filename in os.listdir(path):
    if filename.startswith('peaks'):
        rec.extend(np.load(path+filename)['rec'])
        os.remove(path+filename)
        print(filename)
np.savez(path+'AllPeaks', rec=rec)
