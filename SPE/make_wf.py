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
from fun import do_smd, do_dif, find_peaks, analize_peaks
import pickle
from classes import WaveForm




pmt=1
init=80
left=50
right=250


PMT_num=20
time_samples=1024
pmts=np.array([0,1,2,3,4,5,6,7,8,9,10,11,17,14,15,16,18,19])
W=np.zeros(1000)
BL=np.zeros(1000)
path='/home/gerak/Desktop/DireXeno/190803/pulser/'
file=open(path+'out.DXD', 'rb')
id=0
id0=id
np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*id)
start_time = time.time()
REC=[]
rec=np.recarray(5000, dtype=[
    ('blw', 'f8'),
    ('height', 'i8'),
    ])
j=0
while id<1e5:
    if id%100==0:
        print('Event number {} ({} files per sec). {} file were saved.'.format(id, 100/(time.time()-start_time), j))
        start_time = time.time()
    Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
    if len(Data)<(PMT_num+4)*(time_samples+2):
        break
    Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
    trig=np.argmin(Data[2:1002,0])
    wf=Data[2:1002, np.nonzero(pmts==pmt)[0][0]+2]
    wf=np.roll(wf, -trig)
    wf=wf-np.median(wf[:init])
    blw=np.sqrt(np.mean(wf[:init]**2))
    rec[j]['blw']=blw
    h=-np.amin((wf)[left:right])
    rec[j]['height']=h
    j+=1
    id+=1
    if j==5000:
        REC.extend(rec)
        j=0

np.savez('wf{}'.format(pmt), rec=REC, init=init, left=left, right=right)
