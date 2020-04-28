import numpy as np
import matplotlib.pyplot as plt
import time
from classes import WaveForm
from fun import find_hits, Recon_wf, get_spes, get_delays
import sys

PMT_num=20
time_samples=1024
id=2000
pmts=[7,8]
Init=10
spes, height_cuts, rise_time_cuts=get_spes(pmts)
delays=get_delays(pmts)

WFs=np.zeros((len(pmts), 1000))
recon_WFs=np.zeros((len(pmts), 1000))

start_time = time.time()
rec=np.recarray(1000, dtype=[
    ('area', 'i8', len(pmts)),
    ('blw', 'f8', len(pmts)),
    ('id', 'i8'),
    ('chi2', 'f8', len(pmts)),
    ('h', 'i8', (1000, len(pmts))),
    ('init', 'i8'),
    ('init_wf', 'i8', len(pmts))
    ])

path='/home/gerak/Desktop/DireXeno/190803/BG/'
file=open(path+'out.DXD', 'rb')
Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*id)
j=0
while True:
    if id%1==0:
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
        for k in range(np.argmin(wf)):
            if np.all(wf[k:k+20]<-blw):
                rec[j]['init_wf'][i]=k
                break
        wf=np.roll(wf, -int(np.round(delays[i]*5)))
        waveform=WaveForm(blw)
        h, recon_wf=Recon_wf(waveform, wf, height_cuts[i], rise_time_cuts[i], spes[i], Init)
        H+=h
        chi2=np.sqrt(np.sum((wf[Init:]-recon_wf[Init:])**2))

        if len(np.nonzero(H>0)[0])==0:
            init=-1
        else:
            init=np.amin(np.nonzero(H>0)[0])

        if blw<4.7 and init>20 and chi2<500:
            WFs[i]+=wf
            recon_WFs[i]+=recon_wf

        rec[j]['area']=-np.sum(wf[Init:])
        rec[j]['blw'][i]=blw
        rec[j]['id']=id
        rec[j]['chi2'][i]=chi2
        rec[j]['h'][:,i]=h
    for i, pmt in enumerate(pmts):
        rec[j]['h'][:,i]=np.roll(rec[j]['h'][:,i], -init)
        rec[j]['init']=init
    j+=1
    id+=1
    if j==len(rec):
        np.savez(path+'EventRecon/recon{}'.format(id-1), rec=rec, WFs=WFs, recon_WFs=recon_WFs, pmts=pmts)
        WFs=np.zeros((len(pmts), 1000))
        recon_WFs=np.zeros((len(pmts), 1000))
        j=0
np.savez(path+'EventRecon/recon{}'.format(id-1), rec=rec[:j-1], WFs=WFs, recon_WFs=recon_WFs, pmts=pmts)
