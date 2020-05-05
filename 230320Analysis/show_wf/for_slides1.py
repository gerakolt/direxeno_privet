import numpy as np
import matplotlib.pyplot as plt
import time
from classes import WaveForm
from fun import find_hits, show_wf, Recon_wf, smd

pmt=8
id=2168

path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
BL=np.load(path+'bl.npz')['BL']
data=np.load(path+'spe.npz')
spe=data['spe']
height_cut=data['height_cut']
rise_time_cut=data['rise_time_cut']

path='/home/gerak/Desktop/DireXeno/190803/Co57/PMT{}/'.format(pmt)
data=np.load(path+'raw_wf.npz')
init=data['init']
blw_cut=data['blw_cut']



x=np.arange(1000)/5

PMT_num=20
time_samples=1024
path='/home/gerak/Desktop/DireXeno/190803/Co57/'
file=open(path+'out.DXD', 'rb')

Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*id)
Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
wf=Data[2:1002, pmt+2]
wf=wf-np.median(wf[:init])-BL
blw=np.sqrt(np.mean(wf[:init]**2))
if blw>blw_cut:
    temp=1
WF=WaveForm(blw)
h, recon_wf=Recon_wf(WF, wf, init, height_cut, rise_time_cut, spe)

Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
wf1=Data[2:1002, pmt+2]
wf1=wf1-np.median(wf1[:init])-BL
blw=np.sqrt(np.mean(wf[:init]**2))
if blw>blw_cut:
    temp=1
WF=WaveForm(blw)
h1, recon_wf1=Recon_wf(WF, wf, init, height_cut, rise_time_cut, spe)

fig, ((ax1), (ax2)) = plt.subplots(1, 2)
ax1.plot(x, wf, 'k.-', label='waveforms from\ndifferent events')
ax1.plot(x, wf1, 'r.-', label='')
ax1.legend(fontsize=24)

t=10+np.random.exponential(scale=25, size=50)
t2=5+np.random.exponential(scale=25, size=50)

ax2.plot(x[h>0], np.ones(len(x[h>0])), 'ko', label='Time of resolved PEs')
ax2.plot(t, 1.5*np.ones(len(t)), 'ro', label='Possible photon emission time')
ax2.plot(t2, 0.5*np.ones(len(t)), 'ro')
ax2.set_ylim(0,2)
ax2.legend(fontsize=24)
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.9, wspace=0, hspace=0)
plt.show()
