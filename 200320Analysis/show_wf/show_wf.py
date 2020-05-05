import numpy as np
import matplotlib.pyplot as plt
import time
from classes import WaveForm
from fun import find_hits, show_wf

pmt=0
id=9874
path='/home/gerak/Desktop/DireXeno/190803/Co57/PMT{}/'.format(pmt)
data=np.load(path+'raw_wf.npz')
init=data['init']
blw_cut=data['blw_cut']
# height_cut=data['height_cut']
# rise_time_cut=data['rise_time_cut']
height_cut=35
rise_time_cut=5

PMT_num=20
time_samples=1024
path='/home/gerak/Desktop/DireXeno/190803/Co57/'
file=open(path+'out.DXD', 'rb')

Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*id)
Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
wf=Data[2:1002, pmt+2]
wf=wf-np.median(wf[:init])
blw=np.sqrt(np.mean(wf[:init]**2))
WF=WaveForm(blw)
find_hits(WF, wf, init, height_cut, rise_time_cut)

x=np.arange(1000)
plt.figure()
plt.title('Fin')
plt.plot(x, wf, 'k.-', label='wf')
plt.fill_between(x, y1=-WF.blw, y2=0, alpha=0.3)
for hit in WF.hits:
    plt.fill_between(x[hit.init:hit.fin], y1=np.amin(wf), y2=0, alpha=0.3, label='init={}, area={}'.format(hit.init, hit.area))
plt.legend()
plt.show()
