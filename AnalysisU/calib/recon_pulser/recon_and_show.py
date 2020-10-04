import numpy as np
import matplotlib.pyplot as plt
import time
from classes import WaveForm
from fun import find_hits, Recon_wf, get_spes, get_delays
import sys

PMT_num=20
time_samples=1024
id=48928
pmts=[0,1,4,7,8,14]
chns=[2,3,6,9,10,15]
pmts=[0]
chns=[2]
left=240
right=380
Init=20
spes, height_cuts, dh3_cut, BL=get_spes(pmts)
delays=get_delays(pmts)

WFs=np.zeros((len(pmts), 1000))
recon_WFs=np.zeros((len(pmts), 1000))

start_time = time.time()

path='/home/gerak/Desktop/DireXeno/190803/pulser/'
file=open(path+'out.DXD', 'rb')
Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*id)
j=0

Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
trig=np.argmin(Data[2:1002,0])
for i, pmt in enumerate(pmts):
    wf=Data[2:1002, chns[i]]
    wf=wf-np.median(wf[:left])
    blw=np.sqrt(np.mean(wf[:left]**2))
    wf-=BL[i]
    waveform=WaveForm(blw)
    h, recon_wf, Areas, DCAreas, Abins=Recon_wf(waveform, wf, height_cuts[i], dh3_cut[i], spes[i], Init, left, right, 0.5)

t=np.arange(1000)
fig, ax=plt.subplots(1)
fig.suptitle('..', fontsize=25)
np.ravel(ax)[0].plot(t, wf, 'ko', label='PMT{}'.format(pmts[i]))
np.ravel(ax)[0].plot(t, spes[i], 'b.-', label='PMT{}'.format(pmts[i]))
np.ravel(ax)[0].plot(t, recon_wf, 'r.-')
np.ravel(ax)[0].plot(t, wf-recon_wf, 'g--')
np.ravel(ax)[0].legend(fontsize=15)
plt.show()
