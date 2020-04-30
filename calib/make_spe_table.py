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




pmt=11
areas=np.load('/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/area{}.npz'.format(pmt, pmt))
area_left=areas['left']
area_right=areas['right']
Peaks=np.load('/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/cuts{}.npz'.format(pmt, pmt))
left=Peaks['left']
right=Peaks['right']
d_cut=Peaks['d_cut']
init_cut=Peaks['init_cut']
blw_cut=Peaks['blw_cut']


PMT_num=20
time_samples=1024
pmts=np.array([0,1,2,3,4,5,6,7,8,9,10,11,17,14,15,16,18,19])
SPE=np.zeros((5000, 1000))
path='/home/gerak/Desktop/DireXeno/190803/pulser/'
file=open(path+'out.DXD', 'rb')
id=0
id0=id
np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*id)
start_time = time.time()
j=0

rec=np.recarray(5000, dtype=[
    ('id', 'i8'),
    ('init10', 'i8'),
    ])
while id<1e6:
    if id%100==0:
        print('Event number {} ({} files per sec). {} file were saved.'.format(id, 100/(time.time()-start_time), j))
        start_time = time.time()
    Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
    if len(Data)<(PMT_num+4)*(time_samples+2):
        break
    Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
    trig=np.argmin(Data[2:1002,0])
    wf=Data[2:1002, np.nonzero(pmts==pmt)[0][0]+2]
    wf=wf-np.median(wf[:init_cut])
    blw=np.sqrt(np.mean(wf[:init_cut]**2))
    if blw>blw_cut:
        id+=1
        continue
    wf_copy=np.array(wf)
    for area_wind, peak in find_peaks(np.reshape(wf, (len(wf), 1)), np.array([blw]), pmts, 0, 1000):
        if peak.init10-trig>left and peak.init10-trig<right and peak.maxi-peak.init10>d_cut and peak.area>area_left and peak.area<area_right:
            # plt.figure()
            # x=np.arange(1000)
            # plt.plot(wf_copy, 'k.-')
            # plt.fill_between(x[peak.init:peak.fin], y1=wf_copy[peak.init:peak.fin], y2=0)
            # plt.show()
            SPE[j]=np.roll(wf_copy, 200-peak.init10)
            rec[j]=id, peak.init10
            j+=1
            if j==len(SPE[:,0]):
                np.savez(path+'PMT{}/SPE{}_{}to{}'.format(pmt, pmt, id0, id), SPE=SPE, rec=rec)
                j=0
                id0=id+1
            break
    id+=1
np.savez(path+'PMT{}/SPE{}_{}to{}'.format(pmt, pmt, id0, id-1), SPE=SPE[:j-1], rec=rec[:j-1])

SPE=np.zeros(1000)
rec=np.recarray(0, dtype=[
    ('id', 'i8'),
    ('init10', 'i8'),
    ])
for filename in os.listdir(path+'PMT{}'.format(pmt)):
    if filename.startswith('SPE{}_'.format(pmt)):
        Data=np.load(path+'PMT{}/'.format(pmt)+filename)
        SPE=np.vstack((SPE, Data['SPE']))
        rec=np.append(rec, Data['rec'])
        os.remove(path+'PMT{}/'.format(pmt)+filename)
        print(filename)
np.savez(path+'PMT{}/'.format(pmt)+'SPEs{}'.format(pmt), SPE=SPE[1:], rec=rec)
