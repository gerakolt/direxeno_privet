import os, sys
import numpy as np
from scipy import signal
from itertools import product
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
import sys
from fun import find_peaks
from classes import WaveForm
import pickle

# PMT_num=21
PMT_num=20
time_samples=1024
rec=np.recarray(150000, dtype=[
    ('id', 'i8'),
    ('pmt', 'i8'),
    ('blw', 'f8'),
    ('t', 'i8'),
    ('dt', 'i8'),
    ('d', 'i8'),
    ('h', 'i8'),
    ('area_peak', 'i8'),
    ('area_wind', 'i8')
    ])

pmt=1
l=150
r=325
blw_cut=15
left=70
right=115
height_cut=35
d_cut=4
path='/home/gerak/Desktop/DireXeno/190803/pulser/'
file=open(path+'out.DXD', 'rb')
id=0
id0=id
np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*id)
start_time = time.time()
j=0
while id<1e6:
    if id%100==0:
        print('({}), Event number {} ({} files per sec).'.format(path, id, 100/(time.time()-start_time)))
        start_time = time.time()
    Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
    if len(Data)<(PMT_num+4)*(time_samples+2):
        break
    Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
    trig=np.argmin(Data[2:1002,0])
    wf=Data[2:1002,3]
    bl=np.median(wf[:150])
    wf=wf-bl
    blw=np.sqrt(np.mean(wf[:150]**2))
    wf_copy=np.array(wf)
    for [area_wind, peak] in find_peaks(wf, blw, pmt, l, r):
        rec[j]=id, peak.pmt, peak.blw, peak.init10-trig, peak.fin-peak.init, peak.maxi-peak.init10, peak.height, peak.area, area_wind
        # if peak.area>500:
        #     plt.figure()
        #     x=np.arange(1000)
        #     plt.plot(x, wf_copy, 'k.-')
        #     plt.fill_between(x[peak.init:peak.fin],y1=wf_copy[peak.init:peak.fin], y2=0, label='Peak area={}'.format(peak.area), color='r', alpha=0.3)
        #     plt.fill_between(x[peak.init10-200+l:peak.init10-200+r], y1=np.amin(wf_copy), y2=0, label='Wind area={}'.format(area_wind), color='g', alpha=0.3)
        #     print(-np.sum(wf[l-200+peak.init10:peak.init]), -np.sum(wf[peak.fin:r-200+peak.init10]))
        #     plt.legend()
        #     plt.show()
        j+=1
        if j==len(rec):
            np.savez(path+'peaks{}to{}'.format(id0, id), rec=rec)
            id0=id+1
            j=0
    id+=1
np.savez(path+'peaks{}to{}'.format(id0, id-1), rec=rec[:j-1])
rec=[]
for filename in os.listdir(path):
    if filename.startswith('peaks'):
        rec.extend(np.load(path+filename)['rec'])
        os.remove(path+filename)
        print(filename)
np.savez(path+'AllPeaks', rec=rec)
