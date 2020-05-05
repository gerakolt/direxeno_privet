import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt

pmts=[0,1,2,3,4,5,6,7,8,9,10,11,17,14,15,16,18,19]
path='/home/gerak/Desktop/DireXeno/190803/Co57/'
D=np.zeros((len(pmts), len(pmts), 1))
for filename in os.listdir(path):
    if filename.startswith('subDelay3DTable'):
        D=np.dstack((D, np.load(path+filename)['D']))
        os.remove(path+filename)
        print(filename)
np.savez(path+'Delay3DTable', D=D[:,:,1:])
