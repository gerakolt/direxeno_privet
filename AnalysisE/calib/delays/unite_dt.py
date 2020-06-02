import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

path='/home/gerak/Desktop/DireXeno/190803/pulser/delays_recon/'
delays=[]
for filename in os.listdir(path):
    if filename.endswith(".npz") and filename.startswith("delays_array"):
        data=np.load(path+filename)
        for dt in data['dt']:
            delays.append(dt)
np.savez(path+'delays_array.npz', delays=delays)
