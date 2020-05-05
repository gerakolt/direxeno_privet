import numpy as np
import time
import matplotlib.pyplot as plt



pmt=19
PMT_num=20
time_samples=1024
path='/home/gerak/Desktop/DireXeno/190803/pulser/'
data=np.load(path+'PMT{}/raw_wf.npz'.format(pmt))
left=data['left']
right=data['right']
init=data['init']
data=np.load(path+'PMT{}/cuts.npz'.format(pmt))
blw_cut=data['blw_cut']
height_cut=data['height_cut']

file=open(path+'out.DXD', 'rb')
BL=np.zeros(1000)
id=0
j=0
start_time = time.time()
n_BL=0
while id<1e5:
    if id%100==0:
        print('Event number {} ({} files per sec). {} file were saved.'.format(id, 100/(time.time()-start_time), j))
        start_time = time.time()
    Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
    if len(Data)<(PMT_num+4)*(time_samples+2):
        break
    Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
    trig=np.argmin(Data[2:1002,0])
    wf=Data[2:1002, pmt]
    wf=np.roll(wf, -trig)
    wf=wf-np.median(wf[:init])
    blw=np.sqrt(np.mean(wf[:init]**2))
    if blw>blw_cut:
        id+=1
        continue
    h=-np.amin(wf[left:right])
    if h<height_cut:
        BL+=wf
        n_BL+=1
    id+=1

np.savez(path+'PMT{}/bl'.format(pmt), BL=BL/n_BL)
