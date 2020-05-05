import numpy as np
import time
import matplotlib.pyplot as plt


pmt=8
init=60
left=50
right=250

PMT_num=20
time_samples=1024
path='/home/gerak/Desktop/DireXeno/190803/pulser/'
file=open(path+'out.DXD', 'rb')
REC=[]
rec=np.recarray(5000, dtype=[
    ('blw', 'f8'),
    ('height', 'i8'),
    ])
j=0
id=0
start_time = time.time()
while id<1e5:
    if id%100==0:
        print('Event number {} ({} files per sec). {} file were saved.'.format(id, 100/(time.time()-start_time), j))
        start_time = time.time()
    Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
    if len(Data)<(PMT_num+4)*(time_samples+2):
        break
    Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
    trig=np.argmin(Data[2:1002,0])
    wf=Data[2:1002, pmt+2]
    wf=np.roll(wf, -trig)
    wf=wf-np.median(wf[:init])
    blw=np.sqrt(np.mean(wf[:init]**2))
    # plt.figure()
    # x=np.arange(1000)
    # plt.plot(x, wf, 'k.')
    # plt.fill_between(x[left:right], y1=np.amin(wf), y2=0, alpha=0.3)
    # plt.show()
    rec[j]['blw']=blw
    h=-np.amin((wf)[left:right])
    rec[j]['height']=h
    j+=1
    id+=1
    if j==5000:
        REC.extend(rec)
        j=0

np.savez(path+'PMT{}/raw_wf'.format(pmt), rec=REC, init=init, left=left, right=right)
