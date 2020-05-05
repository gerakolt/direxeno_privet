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

PMT_num=20
time_samples=1024
rec=np.recarray(150000, dtype=[
    ('id', 'i8'),
    ('pmt', 'i8'),
    ('blw', 'f8'),
    ('init', 'i8'),
    ('t', 'i8'),
    ('dt', 'i8'),
    ('d', 'i8'),
    ('h', 'i8'),
    ('area_peak', 'i8'),
    ('area_wind', 'i8')    ])

pmt=11
path='/home/gerak/Desktop/DireXeno/190803/pulser/'
if os.path.isfile(path+'PMT{}/cuts{}.npz'.format(pmt,pmt)):
    Data=np.load(path+'PMT{}/cuts{}.npz'.format(pmt,pmt))
    l=Data['l']
    r=Data['r']
    init_cut=Data['init_cut']
else:
    l=125
    r=340
    init_cut=100

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
    wf=Data[2:1002,pmt+2]
    bl=np.median(wf[:init_cut])
    wf=wf-bl
    blw=np.sqrt(np.mean(wf[:init_cut]**2))
    for [area_wind, peak] in find_peaks(wf, blw, pmt, l, r):
        rec[j]=(id, peak.pmt, peak.blw, peak.init, peak.init10-trig, peak.fin-peak.init, peak.maxi-peak.init10, peak.height,
         peak.area, area_wind)
        j+=1
        if j==len(rec):
            np.savez(path+'PMT{}/peaks{}to{}'.format(pmt, id0, id), rec=rec)
            id0=id+1
            j=0
    id+=1
np.savez(path+'PMT{}/peaks{}to{}'.format(pmt, id0, id-1), rec=rec[:j-1])
rec=[]
for filename in os.listdir(path+'PMT{}'.format(pmt)):
    if filename.startswith('peaks'):
        rec.extend(np.load(path+'PMT{}/'.format(pmt)+filename)['rec'])
        os.remove(path+'PMT{}/'.format(pmt)+filename)
        print(filename)
np.savez(path+'PMT{}/Peaks{}'.format(pmt, pmt), rec=rec)
