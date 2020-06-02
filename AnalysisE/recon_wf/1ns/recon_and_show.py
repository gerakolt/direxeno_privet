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

PMT_num=12
time_samples=1024
id=49
pmts=np.array([0,5,12,13,14,15,16,18,19,2,3,4,10])
chns=[0,1,2,3,4,5,6,7,8,9,10,11,13]
Init=20
spes, BL, height_cuts, dh3_cuts, spk_cuts=get_spes(pmts)
delays=get_delays(pmts)

WFs=np.zeros((len(pmts), 1000))
recon_WFs=np.zeros((len(pmts), 1000))


path='/home/gerak/Desktop/DireXeno/050520/DC/'
file=open(path+'out.DXD', 'rb')
Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*id)
j=0
Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))

Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
trig=find_trig(Data[2:1002,PMT_num+2])
H=np.zeros(200)
for i, pmt in enumerate(pmts):
    if pmt==19:
        wf=Data[2:1002, chns[np.nonzero(pmts==pmt)[0][0]]]
        wf=wf-np.median(wf[:Init])
        blw=np.sqrt(np.mean(wf[:Init]**2))
        fuck=np.array(wf)
        wf-=BL[i]
        for k in range(np.argmin(wf)):
            init_wf=1000
            if np.all(wf[k:k+20]<-blw):
                init_wf=k
                break

        # wf=np.roll(wf, int(np.round(delays[i]*5)))
        waveform=WaveForm(blw)
        h, recon_wf=Recon_wf(waveform, wf, height_cuts[i], dh3_cuts[i], spk_cuts[i], spes[i], Init)

        x=np.arange(1000)
        plt.figure()
        plt.title(pmt)
        plt.plot(x, wf, 'k.')
        plt.plot(x,fuck, 'y.')
        plt.plot(x, recon_wf, 'r.', label='{} PEs, (t_PE0={})'.format(np.sum(h), np.amin(np.nonzero(h>0)[0])))
        if init_wf<1000:
            plt.plot(x[init_wf], wf[init_wf], 'go')
        plt.fill_between(x, y1=-blw, y2=0)
        plt.axhline(y=-height_cuts[i], xmin=0, xmax=1, color='k')
        plt.legend()
        plt.show()
