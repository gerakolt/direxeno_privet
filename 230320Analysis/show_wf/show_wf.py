import numpy as np
import matplotlib.pyplot as plt
import time
from classes import WaveForm
from fun import find_hits, show_wf

pmt=0
id=81779
init=70

PMT_num=20
time_samples=1024
path='/home/gerak/Desktop/DireXeno/190803/Co57/'
file=open(path+'out.DXD', 'rb')

Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*id)
Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
for pmt in [0,8]:
    wf=Data[2:1002, pmt+2]
    wf=wf-np.median(wf[:init])
    blw=np.sqrt(np.mean(wf[:init]**2))
    # WF=WaveForm(blw)
    # find_hits(WF, wf, init, height_cut, rise_time_cut)

    x=np.arange(1000)
    plt.figure()
    plt.title('PMT{}'.format(pmt))
    plt.plot(x, wf, 'k.-', label='wf')
    # plt.fill_between(x, y1=-WF.blw, y2=0, alpha=0.3)
    plt.legend()
    plt.show()
