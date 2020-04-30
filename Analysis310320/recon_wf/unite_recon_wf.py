import numpy as np
import matplotlib.pyplot as plt
import time
from classes import WaveForm
from fun import find_hits, Recon_wf, get_spes, get_delays
import sys
import os

path='/home/gerak/Desktop/DireXeno/190803/Cs137/EventRecon/'
pmts=[0,7,8]
Rec=np.recarray(500000, dtype=[
    ('area', 'i8', len(pmts)),
    ('blw', 'i8', len(pmts)),
    ('id', 'i8'),
    ('chi2', 'f8', len(pmts)),
    ('h', 'i8', (1000, len(pmts))),
    ('init', 'i8'),
    ])
j=0
id=0
WFs=np.zeros((len(pmts), 1000))
recon_WFs=np.zeros((len(pmts), 1000))
for filename in os.listdir(path):
    if filename.endswith(".npz") and filename.startswith("recon"):
        print(filename)
        data=np.load(path+filename)
        rec=data['rec']
        WFs+=data['WFs']
        recon_WFs+=data['recon_WFs']
        for r in rec:
            Rec[j]['area']=r['area']
            Rec[j]['blw']=r['blw']
            Rec[j]['id']=r['id']
            Rec[j]['chi2']=r['chi2']
            Rec[j]['h']=r['h']
            Rec[j]['init']=r['init']
            if r['id']>id:
                id=r['id']
            j+=1
        os.remove(path+filename)

np.savez(path+'recon{}'.format(id), rec=Rec[:j-1], WFs=WFs, recon_WFs=recon_WFs)
