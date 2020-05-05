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
from fun import show_peaks
from classes import WaveForm
import pickle

pmt=1
PMT_num=20
time_samples=1024
rec=np.recarray(150000, dtype=[
    ('id', 'i8'),
    ('pmt', 'i8'),
    ('blw', 'f8'),
    ('t', 'i8'),
    ('d', 'i8'),
    ('h', 'i8'),
    ('area', 'i8')
    ])

pmts=[0,1,2,3,4,5,6,7,8,9,10,11,17,14,15,16,18,19]
path='/home/gerak/Desktop/DireXeno/190803/pulser/'
file=open(path+'out.DXD', 'rb')
id=0
id0=id
np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*id)
start_time = time.time()
Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
trig=np.argmin(Data[2:1002,0])
wfs=Data[2:1002,2:len(pmts)+2]
bl=np.median(wfs[:150], axis=0)
wfs=wfs-bl
blw=np.sqrt(np.mean(wfs[:150]**2, axis=0))
for peak in show_peaks(wfs, blw, pmts):
    temp=1
