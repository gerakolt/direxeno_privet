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
fig, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2, 3, sharex=True, sharey='row')
fig.suptitle('Co57 Events', fontsize=24)

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

ax1.plot(x, wf, 'k.-', label='Original waveform')
ax1.plot(x, recon_wf, 'r.-', label='Reconstructed waveform')
ax1.fill_between(x, y1=-WF.blw, y2=0, alpha=0.3)
ax1.legend(fontsize=24)

ax2.step(x, h, label='Distribution of\nresolved PEs\nin time')
ax2.legend(fontsize=24)


Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
wf=Data[2:1002, pmt+2]
wf=wf-np.median(wf[:init])-BL
blw=np.sqrt(np.mean(wf[:init]**2))
if blw>blw_cut:
    temp=1
WF=WaveForm(blw)
h, recon_wf=Recon_wf(WF, wf, init, height_cut, rise_time_cut, spe)

ax3.plot(x, wf, 'k.-', label='wf ({} pes)'.format(np.sum(wf[init:])/np.sum(spe[init:])))
ax3.plot(x, recon_wf, 'r.-', label='recon wf ({} pes)'.format(np.sum(h[0])))
ax3.fill_between(x, y1=-WF.blw, y2=0, alpha=0.3)

ax4.step(x, h)

Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
wf=Data[2:1002, pmt+2]
wf=wf-np.median(wf[:init])-BL
blw=np.sqrt(np.mean(wf[:init]**2))
if blw>blw_cut:
    temp=1
WF=WaveForm(blw)
h, recon_wf=Recon_wf(WF, wf, init, height_cut, rise_time_cut, spe)

ax5.plot(x, wf, 'k.-', label='wf ({} pes)'.format(np.sum(wf[init:])/np.sum(spe[init:])))
ax5.plot(x, recon_wf, 'r.-', label='recon wf ({} pes)'.format(np.sum(h[0])))
ax5.fill_between(x, y1=-WF.blw, y2=0, alpha=0.3)

ax6.step(x, h)
plt.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.9, wspace=0, hspace=0)
fig.text(0.5, 0.05,'Time [ns]', va='center', ha='center', fontsize=24)
plt.show()
