import numpy as np
import time
import matplotlib.pyplot as plt


def find_trig(wf):
    bl=np.median(wf[:200])
    top=np.median(wf[325:475])
    mid=0.5*(bl+top)
    return np.argmin(np.abs(wf[:400]-mid))

pmt=7
id=0
path='/home/gerak/Desktop/DireXeno/190803/pulser/NEWPMT{}/'.format(pmt)
data=np.load(path+'cuts.npz')
blw_cut=data['blw_cut']
height_cut=data['height_cut']
left=data['left']
right=data['right']
BL=np.load(path+'BL.npz')['BL']


PMT_num=20
time_samples=1024
path='/home/gerak/Desktop/DireXeno/190803/pulser/'.format(pmt)
file=open(path+'out.DXD', 'rb')

Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*id)
Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
# trig=find_trig(Data[2:1002,0])
trig=np.argmin(Data[2:1002,0])
wf=Data[2:1002, pmt+2]
bl=np.median(wf[trig-100:trig])
wf-=bl
blw=np.sqrt(np.mean((wf[trig-100:trig])**2))
wf=np.roll(wf, 100-trig)
wf-=BL
h=-np.amin(wf[left:right])
maxi=left+np.argmin(wf[left:right])
init10=np.amax(np.nonzero(wf[:maxi]>-0.1*h)[0])

x=np.arange(1000)
plt.figure()
plt.plot(x, wf, 'k.')
plt.plot(init10, wf[init10], 'ro')
plt.show()
