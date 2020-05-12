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
id=2670
Init=100
pmts=np.array([0,5,12,13,14,15,16,18,19,2,3,4,10])
chns=[0,1,2,3,4,5,6,7,8,9,10,11,13]
spes, BL, height_cuts, dh3_cuts, spk_cuts=get_spes(pmts)

start_time = time.time()
rec=np.recarray(1000, dtype=[
    ('blw', 'f8', len(pmts)),
    ('init10', 'f8', len(pmts)),
    ('id', 'i8'),
    ('h', 'i8', (1000, len(pmts))),
    ])

path='/home/gerak/Desktop/DireXeno/050520/pulser/'
file=open(path+'out.DXD', 'rb')
Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*id)
j=0
while True:
    Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
    Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
    trig=find_trig(Data[2:1002,PMT_num+2])
    H=np.zeros(1000)
    for i, pmt in enumerate(pmts):
        if pmt==19:
            wf=Data[2:1002, chns[np.nonzero(pmts==pmt)[0][0]]]
            wf=np.roll(wf, 100-trig)
            wf=wf-np.median(wf[:Init])
            blw=np.sqrt(np.mean(wf[:Init]**2))
            wf+=BL[i]
            waveform=WaveForm(blw)
            if np.amin(wf)<-100:
                h, recon_wf=Recon_wf(waveform, wf, height_cuts[i], dh3_cuts[i], spk_cuts[i], spes[i], 3)

                x=np.arange(1000)
                fig, (ax1, ax2)=plt.subplots(2,1)
                fig.suptitle(pmt)
                ax1.plot(x, wf, 'k-.')
                ax1.plot(x, recon_wf, 'r.-')
                ax2.plot(x, spes[i], 'r-.')
                plt.show()
