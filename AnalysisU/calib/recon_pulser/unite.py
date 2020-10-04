import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import os
import sys

pmts=np.array([0,1,4,7,8,15])

path='/home/gerak/Desktop/DireXeno/190803/pulser/EventRecon/'
Rec=np.recarray(100000, dtype=[
    ('Areas', 'i8', (len(pmts), 60)),
    ('DCAreas', 'i8', (len(pmts), 60)),
    ('blw', 'f8', len(pmts)),
    ('id', 'i8'),
    ('chi2', 'f8', len(pmts)),
    ('h', 'i8', (1000, len(pmts))),
    ('init_event', 'i8'),
    ('init_wf', 'i8', len(pmts))
    ])
j=0
id=0
WFs=np.zeros((len(pmts), 1000))
recon_WFs=np.zeros((len(pmts), 1000))
for filename in os.listdir(path):
    if filename.endswith(".npz") and filename.startswith("recon1ns"):
        print(filename)
        data=np.load(path+filename)
        Abins=data['Abins']
        rec=data['rec']
        WFs+=data['WFs']
        recon_WFs+=data['recon_WFs']
        for r in rec:
            Rec[j]['Areas']=r['Areas']
            Rec[j]['DCAreas']=r['DCAreas']
            Rec[j]['blw']=r['blw']
            Rec[j]['id']=r['id']
            Rec[j]['chi2']=r['chi2']
            Rec[j]['init_wf']=r['init_wf']
            Rec[j]['h']=r['h']
            Rec[j]['init_event']=r['init_event']
            if r['id']>id:
                id=r['id']
            j+=1
        os.remove(path+filename)
np.savez(path+'recon1ns'.format(id), rec=Rec[:j-1], WFs=WFs, recon_WFs=recon_WFs, Abins=Abins)
