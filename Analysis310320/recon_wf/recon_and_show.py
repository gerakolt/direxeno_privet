import numpy as np
import matplotlib.pyplot as plt
import time
from classes import WaveForm
from fun import find_hits, Recon_wf, get_spes, get_delays

PMT_num=20
time_samples=1024
id=99938
pmts=[7,8]
Init=20
spes, height_cuts, rise_time_cuts=get_spes(pmts)
delays=get_delays(pmts)
WFs=np.zeros((len(pmts), 1000))
recon_WFs=np.zeros((len(pmts), 1000))

# x=np.arange(1000)/5
# plt.figure()
# plt.plot(x[25*5:75*5], spes[0][25*5:75*5], 'k-.', label='PMT7', linewidth=5)
# plt.plot(x[25*5:75*5], spes[1][25*5:75*5], 'r-.', label='PMT8', linewidth=5)
# plt.xlabel('Time [ns]', fontsize=25)
# plt.legend(fontsize=25)
# plt.show()

start_time = time.time()
rec=np.recarray(100000, dtype=[
    ('area', 'i8', len(pmts)),
    ('blw', 'i8', len(pmts)),
    ('id', 'i8'),
    ('chi2', 'f8', len(pmts)),
    ('h', 'i8', (1000, len(pmts))),
    ('init', 'i8'),
    ('init_wf', 'i8')
    ])

path='/home/gerak/Desktop/DireXeno/190803/BG/'
file=open(path+'out.DXD', 'rb')
Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*id)
Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
trig=np.argmin(Data[2:1002,0])
x=np.arange(1000)
H=np.zeros(1000)
fig, (ax1, ax2)=plt.subplots(2,1)
ax=[ax1, ax2]
for i, pmt in enumerate(pmts):
    wf=Data[2:1002, pmt+2]
    wf=wf-np.median(wf[:Init])
    blw=np.sqrt(np.mean(wf[:Init]**2))
    init_wf=0
    for j in range(np.argmin(wf)):
        if np.all(wf[j:j+20]<-blw):
            init_wf=j
            break
    wf=np.roll(wf, -int(np.round(delays[i]*5)))
    waveform=WaveForm(blw)
    h, recon_wf=Recon_wf(waveform, wf, height_cuts[i], rise_time_cuts[i], spes[i], Init)
    chi2=np.sqrt(np.sum((wf[Init:]-recon_wf[Init:])**2))
    fig.suptitle(pmt)
    ax[i].plot(x, wf, 'k-.', label='PMT{}'.format(pmts[i]), linewidth=5)
    ax[i].plot(x, recon_wf, 'r-.', label='Reconstruction', linewidth=5)
    ax[i].fill_between(x, y1=-waveform.blw, y2=0)
    ax[i].legend()

plt.show()
