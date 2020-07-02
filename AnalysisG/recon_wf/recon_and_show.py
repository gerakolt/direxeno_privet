import numpy as np
import matplotlib.pyplot as plt
import time
from classes import WaveForm
from fun import find_hits, Recon_wf, get_spes, get_delays
import sys

PMT_num=20
time_samples=1024
id=77395
pmts=[0,1,4,7,8,14]
chns=[2,3,6,9,10,15]
Init=20
spes, height_cuts, rise_time_cuts, BL=get_spes(pmts)
delays=get_delays(pmts)


path='/home/gerak/Desktop/DireXeno/190803/Co57/'
file=open(path+'out.DXD', 'rb')
Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*id)
j=0
while True:
    WF=np.zeros(1000)
    H=np.zeros(1000)
    Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
    if len(Data)<(PMT_num+4)*(time_samples+2):
        break
    Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
    for i, pmt in enumerate(pmts[:-1]):
        wf=Data[2:1002, chns[i]]
        wf=wf-np.median(wf[:Init])
        blw=np.sqrt(np.mean(wf[:Init]**2))
        wf-=BL[i]
        wf=np.roll(wf, int(np.round(delays[i]*5)))
        WF+=wf
        waveform=WaveForm(blw)
        h, recon_wf=Recon_wf(waveform, wf, height_cuts[i], rise_time_cuts[i], spes[i], Init)
        H+=h
        wf14=Data[2:1002, chns[-1]]
        wf14=wf14-np.median(wf14[:Init])
        blw=np.sqrt(np.mean(wf14[:Init]**2))
        wf14-=BL[-1]
        WF+=wf14
        #wf14=np.roll(wf14, int(np.round(delays[-1]*5)))
        waveform=WaveForm(blw)
        h14, recon_wf14=Recon_wf(waveform, wf14, height_cuts[-1], rise_time_cuts[-1], spes[-1], Init)
        H+=h14
        x=np.arange(1000)
        fig, (ax1, ax2)= plt.subplots(2,1)
        ax1.plot(x, wf, 'r+')
        ax1.plot(x, wf14, 'k+')
        ax1.plot(x, recon_wf, 'b--')
        ax1.plot(x, recon_wf14, 'g--')

        ax2.plot(h, 'r')
        ax2.plot(h14, 'k')

    fig, (ax1, ax2)= plt.subplots(2,1)
    ax1.plot(x, WF, 'k-.')
    ax2.plot(x, H, 'k.-')
    init=np.amin(np.nonzero(H>0)[0])
    ax1.fill_between(x[init:init+10], y1=np.amin(WF), y2=0)
    ax2.fill_between(x[init:init+10], y1=0, y2=np.amax(H))

    plt.show()
