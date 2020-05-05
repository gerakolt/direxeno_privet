import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import os

rec=np.recarray(0, dtype=[
    ('id', 'i8'),
    ('pmt', 'i8'),
    ('init_first_hit', 'i8'),
    ('init10', 'f8'),
    ('blw', 'f8'),
    ('height_first_hit', 'f8')])

source='Co57'
type=''
path='/home/gerak/Desktop/DireXeno/190803/'+source+type+'/'
for filename in os.listdir(path):
    if filename.startswith('subAllhits'):
        rec=np.append(rec, np.load(path+filename)['rec'])
        os.remove(path+filename)
        print(filename)
np.savez(path+'Allhits', rec=rec)
