import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from fun import find_bl, import_spe, Recon_WF, find_hits, find_init10
from classes import DataSet, Event, WaveForm



PMT_num=20
time_samples=1024
pmt=0
source='Co57'
type=''
path='/home/gerak/Desktop/DireXeno/'+source+'_190803'+type+'/PMT{}/'.format(pmt)
rec=[]
for filename in os.listdir(path):
    if filename.startswith('WFs'):
        rec.extend(np.load(path+filename)['rec'])
        os.remove(path+filename)
        print(filename)
np.savez(path+'AllWFs', rec=rec)
