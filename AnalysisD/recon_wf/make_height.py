import numpy as np
import matplotlib.pyplot as plt
import time
from classes import WaveForm
from fun import find_hits, Recon_wf, get_spes, get_delays
import sys

PMT_num=20
time_samples=1024
id=0
pmts=[0,1,4,7,8,14]
chns=[2,3,6,9,10,15]
Init=20
spes, height_cuts, rise_time_cuts, BL=get_spes(pmts)
delays=get_delays(pmts)

WFs=np.zeros((len(pmts), 1000))
recon_WFs=np.zeros((len(pmts), 1000))

start_time = time.time()
rec=np.recarray(100000, dtype=[
    ('h', 'i8', len(pmts)),
    ('id', 'i8'),
    ])

path='/home/gerak/Desktop/DireXeno/190803/Co57/'
file=open(path+'out.DXD', 'rb')
Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*id)
j=0
while True:
    if id%100==0:
        print(id, (time.time()-start_time)/100, 'sec per events')
        print(path, pmts)
        start_time=time.time()
    Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
    if len(Data)<(PMT_num+4)*(time_samples+2):
        break
    Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
    trig=np.argmin(Data[2:1002,0])
    H=np.zeros(1000)
    rec[j]['id']=id
    for i, pmt in enumerate(pmts):
        wf=Data[2:1002, chns[i]]
        wf=wf-np.median(wf[:Init])
        blw=np.sqrt(np.mean(wf[:Init]**2))
        wf-=BL[i]
        rec[j]['h'][i]=np.amin(wf)/np.amin(spes[i])
    j+=1
    id+=1
np.savez('h'.format(id-1), rec=rec[:j-1], pmts=pmts)
