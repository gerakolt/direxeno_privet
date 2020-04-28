import numpy as np
import matplotlib.pyplot as plt
import time
from classes import WaveForm
from fun import find_hits, Recon_wf, get_spes, get_delays
import sys

PMT_num=20
time_samples=1024
id=0
Init=70
pmts=[7,8]
spes, height_cuts, rise_time_cuts=get_spes(pmts)

start_time = time.time()
rec=np.recarray(1000, dtype=[
    ('blw', 'f8', len(pmts)),
    ('id', 'i8'),
    ('h', 'i8', (1000, len(pmts))),
    ])

path='/home/gerak/Desktop/DireXeno/190803/pulser/'
file=open(path+'out.DXD', 'rb')
Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*id)
j=0
while True:
    if id%1000==0:
        print(id, (time.time()-start_time)/100, 'sec per events')
        print(path, pmts)
        start_time=time.time()
    Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
    if len(Data)<(PMT_num+4)*(time_samples+2):
        break
    Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
    trig=np.argmin(Data[2:1002,0])
    H=np.zeros(1000)
    for i, pmt in enumerate(pmts):
        wf=Data[2:1002, pmt+2]
        wf=wf-np.median(wf[:Init])
        blw=np.sqrt(np.mean(wf[:Init]**2))

        waveform=WaveForm(blw)
        h, recon_wf=Recon_wf(waveform, wf, height_cuts[i], rise_time_cuts[i], spes[i], 0)
        rec[j]['blw'][i]=blw
        rec[j]['id']=id
        rec[j]['h'][:,i]=h
    j+=1
    id+=1
    if j==len(rec):
        np.savez(path+'EventRecon/recon{}'.format(id-1), rec=rec, pmts=pmts)
        WFs=np.zeros((len(pmts), 1000))
        recon_WFs=np.zeros((len(pmts), 1000))
        j=0
np.savez(path+'EventRecon/recon{}'.format(id-1), rec=rec[:j-1], pmts=pmts)
