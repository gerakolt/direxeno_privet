import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from fun import find_hits
from classes import WaveForm
import os



PMT_num=20
time_samples=1024
pmts=np.array([0,1,2,3,4,5,6,7,8,9,10,11,17,14,15,16,18,19])
source='Co57'
type=''
path='/home/gerak/Desktop/DireXeno/190803/'+source+type+'/'
file=open(path+'out.DXD', 'rb')

pmt=5
event=1000
np.fromfile(file, np.float32, (PMT_num+4)*(2+time_samples)*event)

def hits_from_data(data):
    blw=np.sqrt(np.mean(data[:40]**2, axis=0))
    for i, wf in enumerate(data.T):
        WF=WaveForm(pmts[i], blw[i])
        find_hits(WF, wf)
        if len(WF.hits):
            yield WF.hits[0], pmts[i], blw[i]


Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
data=np.array(Data[2:1002,2:20])
bl=np.median(data[:40], axis=0)

wf=data[:,pmts==pmt]
x=np.arange(1000)
plt.plot(x, wf, 'k.-')
plt.show()
