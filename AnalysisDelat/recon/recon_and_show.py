import numpy as np
import matplotlib.pyplot as plt
import time
from classes import WaveForm
from fun import find_hits, Recon_wf, get_spes, get_delays
import sys
#[ 9051  9052  9073 ... 48908 48922 48982]
PMT_num=20
time_samples=1024
id=9051
pmts=[0,1,4,7,8,14]
chns=[2,3,6,9,10,15]
Init=20
spes, height_cuts, rise_time_cuts, BL=get_spes(pmts)
delays=get_delays(pmts)

WFs=np.zeros((len(pmts), 1000))
recon_WFs=np.zeros((len(pmts), 1000))

start_time = time.time()

path='/home/gerak/Desktop/DireXeno/190803/Co57/'
file=open(path+'out.DXD', 'rb')
Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*id)
j=0

Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
trig=np.argmin(Data[2:1002,0])
H=np.zeros(200)
WFs=np.zeros((len(pmts), 1000))
recon_WFs=np.zeros((len(pmts), 1000))
for i, pmt in enumerate(pmts):
    wf=Data[2:1002, chns[i]]
    wf=wf-np.median(wf[:Init])
    blw=np.sqrt(np.mean(wf[:Init]**2))
    wf-=BL[i]
    wf=np.roll(wf, int(np.round(delays[i]*5)))
    waveform=WaveForm(blw)
    h, recon_wf=Recon_wf(waveform, wf, height_cuts[i], rise_time_cuts[i], spes[i], Init)
    WFs[i]=wf
    recon_WFs[i]=recon_wf
    H+=h
init=np.amin(np.nonzero(H>0)[0])
print(np.sum(H[init:init+10])/np.sum(H), init)
t=np.arange(1000)
fig, ax=plt.subplots(3,2)
fig.suptitle('..', fontsize=25)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(t, WFs[i], 'k.')
    np.ravel(ax)[i].plot(t, recon_WFs[i], 'r.')
    np.ravel(ax)[i].fill_between(t[5*init:5*init+50], y1=np.amin(WFs[i]), y2=0)
    np.ravel(ax)[i].legend(fontsize=15)
plt.show()
