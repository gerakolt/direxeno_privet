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
spes, height_cuts, dh3_cut, BL=get_spes(pmts)
left=240
right=380
th=np.array([0.3, 0.5, 0.3, 0.3, 0.3, 0.3])
SPEcorrection=np.array([1, 0.47, 0.42, 0.81, 0.73, 0.9])


WFs=np.zeros((len(pmts), 1000))
recon_WFs=np.zeros((len(pmts), 1000))

start_time = time.time()
rec=np.recarray(1000, dtype=[
    ('Areas', 'i8', (len(pmts), 60)),
    ('DCAreas', 'i8', (len(pmts), 60)),
    ('blw', 'f8', len(pmts)),
    ('id', 'i8'),
    ('chi2', 'f8', len(pmts)),
    ('h', 'i8', (1000, len(pmts))),
    ('init_event', 'i8'),
    ('init_wf', 'i8', len(pmts))
    ])

path='/home/gerak/Desktop/DireXeno/190803/pulser/'
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
    for i, pmt in enumerate(pmts):
        wf=Data[2:1002, chns[i]]
        wf=wf-np.median(wf[:left])
        blw=np.sqrt(np.mean(wf[:left]**2))
        wf-=BL[i]
        for k in range(np.argmin(wf)):
            if np.all(wf[k:k+20]<-blw):
                rec[j]['init_wf'][i]=k
                break
        waveform=WaveForm(blw)
        h, recon_wf, Areas, DCAreas, Abins=Recon_wf(waveform, wf, height_cuts[i], dh3_cut[i], spes[i]*SPEcorrection[i], Init, left, right, th[i])
        H+=h
        chi2=np.sqrt(np.sum((wf[Init:]-recon_wf[Init:])**2))
        if blw<10 and rec[j]['init_wf'][i]>20 and chi2<500:
            WFs[i]+=wf
            recon_WFs[i]+=recon_wf

        rec[j]['Areas'][i]=Areas
        rec[j]['DCAreas'][i]=DCAreas
        rec[j]['blw'][i]=blw
        rec[j]['id']=id
        rec[j]['chi2'][i]=chi2
        rec[j]['h'][:,i]=h

    if len(np.nonzero(H>0)[0])==0:
        init=-1
    else:
        init=np.amin(np.nonzero(H>0)[0])
    rec[j]['init_event']=init
    j+=1
    id+=1
    if j==len(rec):
        np.savez(path+'EventRecon/recon1ns{}'.format(id-1), rec=rec, WFs=WFs, recon_WFs=recon_WFs, pmts=pmts, Abins=Abins)
        WFs=np.zeros((len(pmts), 1000))
        recon_WFs=np.zeros((len(pmts), 1000))
        j=0
np.savez(path+'EventRecon/recon1ns{}'.format(id-1), rec=rec[:j-1], WFs=WFs, recon_WFs=recon_WFs, pmts=pmts, Abins=Abins, left=left, right=right)
