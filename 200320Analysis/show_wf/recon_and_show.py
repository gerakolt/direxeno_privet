import numpy as np
import matplotlib.pyplot as plt
import time
from classes import WaveForm
from fun import find_hits, show_wf, Recon_wf, smd

pmt=0
id=80776

path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
data=np.load(path+'spe.npz')
spe=data['Sig_init10']
height_cut=data['height_cut']
rise_time_cut=data['rise_time_cut']

path='/home/gerak/Desktop/DireXeno/190803/Co57/PMT{}/'.format(pmt)
data=np.load(path+'raw_wf.npz')
init=data['init']
blw_cut=data['blw_cut']


# height_cut=35
# rise_time_cut=5
# dt_cut=15

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
if blw>blw_cut:
    temp=1
WF=WaveForm(blw)
h, recon_wf=Recon_wf(WF, wf, init, height_cut, rise_time_cut, spe, 6,12)


x=np.arange(1000)
plt.figure()
plt.title('Recon WF')
plt.plot(x, wf, 'k.-', label='wf ({} pes)'.format(np.sum(wf[init:])/np.sum(spe[init:])))
plt.axhline(0, xmin=0, xmax=1, color='k')
plt.plot(x, recon_wf, 'r.-', label='recon wf ({} pes)'.format(np.sum(h[0])))
plt.fill_between(x, y1=-WF.blw, y2=0, alpha=0.3)
plt.legend()
plt.show()
