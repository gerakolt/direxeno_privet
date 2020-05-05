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
blw_cut=13
height_cut=25
data=np.load('wf{}.npz'.format(pmt))
left=data['left']
right=data['right']
init=data['init']

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
j=0
REC=[]
rec=np.recarray(5000, dtype=[
    ('height', 'i8'),
    ('area', 'f8'),
    ('t', 'i8')
    ])
j=0
n_BL=0
n_WF=0


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
    if blw>blw_cut:
        id+=1
        continue
    h=-np.amin((wf)[left:right])
    rec[j]['height']=h
    rec[j]['area']=-np.sum((wf)[left:right])
    j+=1
    if h>height_cut:
        W+=wf
        n_WF+=1
    else:
        BL+=wf
        n_BL+=1
    id+=1
    if j==5000:
        REC.extend(rec)
        j=0

np.savez('spe{}'.format(pmt), rec=REC, wf=W/n_WF, bl=BL/n_BL, blw_cut=blw_cut,
                height_cut=height_cut)
