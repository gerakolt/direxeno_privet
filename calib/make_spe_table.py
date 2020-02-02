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


blw_cut=60
left=80
right=110
height_cut=30
d_cut=4

PMT_num=20
time_samples=1024
pmts=np.array([0,1,2,3,4,5,6,7,8,9,10,11,17,14,15,16,18,19])
SPE=np.zeros((5000, 1000))
pmt=4
path='/home/gerak/Desktop/DireXeno/pulser_190803_46211/'
file=open(path+'out.DXD', 'rb')
id=0
id0=id
np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*id)
start_time = time.time()
j=0
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
    wf=wf-np.median(wf[:150])
    blw=np.sqrt(np.mean(wf[:150]**2))
    if blw>blw_cut:
        id+=1
        continue
    wf_copy=np.array(wf)
    for peak in find_peaks(np.reshape(wf, (len(wf), 1)), np.array([blw]), pmts):
        if peak.init10-trig>left and peak.init10-trig<right and peak.maxi-peak.init10>d_cut:
            SPE[j]=np.roll(wf_copy, 200-peak.init10)
            j+=1
            if j==len(SPE[:,0]):
                np.savez(path+'PMT{}/SPE{}to{}'.format(pmt, id0, id), SPE=SPE)
                j=0
                id0=id+1
            break
    id+=1
np.savez(path+'PMT{}/SPE{}to{}'.format(pmt, id0, id-1), SPE=SPE[:j-1])
