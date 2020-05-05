import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt

path='/home/gerak/Desktop/DireXeno/030220/pulser3/PMT19/'
SPE=np.zeros(1000)
<<<<<<< HEAD
init10=np.array([])
=======
rec=np.recarray(0, dtype=[
    ('id', 'i8'),
    ('init10', 'i8'),
    ])
>>>>>>> c7766bd9473ef15b7aa60790e5624a9c78c984b2
for filename in os.listdir(path):
    if filename.startswith('SPE'):
        Data=np.load(path+filename)
        SPE=np.vstack((SPE, Data['SPE']))
<<<<<<< HEAD
        init10=np.append(init10, Data['init10'])
        os.remove(path+filename)
        print(filename)
np.savez(path+'AllSPEs', SPE=SPE[1:], init10=init10)
=======
        rec=np.append(rec, Data['rec'])
        os.remove(path+filename)
        print(filename)
np.savez(path+'AllSPEs', SPE=SPE[1:], rec=rec)
>>>>>>> c7766bd9473ef15b7aa60790e5624a9c78c984b2
