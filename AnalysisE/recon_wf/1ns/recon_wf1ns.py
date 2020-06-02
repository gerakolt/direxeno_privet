import numpy as np
import matplotlib.pyplot as plt
import time
from classes import WaveForm
from fun import find_hits, Recon_wf, get_spes, get_delays
import sys

def find_trig(wf):
    bl=np.median(wf[:200])
    top=np.median(wf[325:475])
    mid=0.5*(bl+top)
    return np.argmin(np.abs(wf[:400]-mid))

PMT_num=20
time_samples=1024
id=0
pmts=np.array([0,1,4,7,8,14])
chns=[2,3,6,9,10,15]
Init=20
spes, BL, height_cuts, dh3_cuts, spk_cuts=get_spes(pmts)
delays=get_delays(pmts)

WFs=np.zeros((len(pmts), 1000))
recon_WFs=np.zeros((len(pmts), 1000))

blw_cut=25

start_time = time.time()
rec=np.recarray(1000, dtype=[
    ('area', 'i8', len(pmts)),
    ('blw', 'f8', len(pmts)),
    ('id', 'i8'),
    ('sat', 'i8', len(pmts)),
    ('chi2', 'f8', len(pmts)),
    ('h', 'i8', (200, len(pmts))),
    ('init_event', 'i8'),
    ('init_wf', 'i8', len(pmts))
    ])

path='/home/gerak/Desktop/DireXeno/190803/Cs137B/'
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
    # trig=find_trig(Data[2:1002,PMT_num+2])
    trig=np.argmin(Data[2:1002,0])
    H=np.zeros(200)

    for i, pmt in enumerate(pmts):
        wf=Data[2:1002, chns[np.nonzero(pmts==pmt)[0][0]]]
        wf=wf-np.median(wf[:Init])
        blw=np.sqrt(np.mean(wf[:Init]**2))
        wf-=BL[i]
        rec[j]['sat'][i]=0
        if np.amin(wf)<-3000:
            rec[j]['sat'][i]=1
            # break
        for k in range(np.argmin(wf)):
            rec[j]['init_wf'][i]=1000
            if np.all(wf[k:k+20]<-blw):
                rec[j]['init_wf'][i]=k
                if k<4:
                    plt.figure()
                    plt.plot(wf, 'k.')
                    plt.show()
                break


        wf=np.roll(wf, int(np.round(delays[i]*5)))
        waveform=WaveForm(blw)
        h, recon_wf=Recon_wf(waveform, wf, height_cuts[i], dh3_cuts[i], spk_cuts[i], spes[i], Init)
        H+=h
        chi2=np.sqrt(np.sum((wf[Init:]-recon_wf[Init:])**2))
        if blw<blw_cut and rec[j]['init_wf'][i]>20 and chi2<10000:
            WFs[i]+=wf
            recon_WFs[i]+=recon_wf

        rec[j]['area']=-np.sum(wf[Init:])
        rec[j]['blw'][i]=blw
        rec[j]['id']=id
        rec[j]['chi2'][i]=chi2
        rec[j]['h'][:,i]=h

    if len(np.nonzero(H>0)[0])==0:
        init=-1
    else:
        init=np.amin(np.nonzero(H>0)[0])
    rec[j]['init_event']=init
    for i, pmt in enumerate(pmts):
        rec[j]['h'][:,i]=np.roll(rec[j]['h'][:,i], -init)
    j+=1
    id+=1
    if j==len(rec):
        np.savez(path+'EventRecon/recon1ns{}'.format(id-1), rec=rec, WFs=WFs, recon_WFs=recon_WFs, pmts=pmts)
        WFs=np.zeros((len(pmts), 1000))
        recon_WFs=np.zeros((len(pmts), 1000))
        j=0
np.savez(path+'EventRecon/recon1ns{}'.format(id-1), rec=rec[:j-1], WFs=WFs, recon_WFs=recon_WFs, pmts=pmts)
