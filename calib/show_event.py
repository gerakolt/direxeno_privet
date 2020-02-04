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


blw_cut=12
left=135
right=165
height_cut=21
d_cut=4

PMT_num=21
time_samples=1024
pmts=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
SPE=np.zeros((5000, 1000))
init10=np.zeros(5000)
pmt=12
path='/home/gerak/Desktop/DireXeno/850V_3/'
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
    wf=Data[2:1002, np.nonzero(pmts==pmt)[0][0]+1]
    wf=wf-np.median(wf[:150])
    blw=np.sqrt(np.mean(wf[:150]**2))
    plt.plot(x, wf, 'k.')
    plt.fill_between(x, y1=-blw, y2=blw)
    plt.show()
    if blw>blw_cut:
        id+=1
        continue
    wf_copy=np.array(wf)
    for peak in find_peaks(np.reshape(wf, (len(wf), 1)), np.array([blw]), pmts):
        print(peak.init10)
