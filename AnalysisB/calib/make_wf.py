import numpy as np
import time
import matplotlib.pyplot as plt


def find_trig(wf):
    bl=np.median(wf[:200])
    top=np.median(wf[325:475])
    mid=0.5*(bl+top)
    return np.argmin(np.abs(wf[:400]-mid))

pmt=19
PMT_num=20
time_samples=1024
path='/home/gerak/Desktop/DireXeno/190803/pulser/'
file=open(path+'out.DXD', 'rb')

rec=np.recarray(100000, dtype=[
    ('blw', 'f8'),
    ('height', 'i8'),
    ('maxi', 'i8'),
    ('bl', 'i8'),
    ])

id=0
j=0
start_time = time.time()
while id<1e5:
    if id%100==0:
        print('Event number {} ({} files per sec). {} file were saved.'.format(id, 100/(time.time()-start_time), j))
        start_time = time.time()
    Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
    if len(Data)<(PMT_num+4)*(time_samples+2):
        break
    Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
    #trig=find_trig(Data[2:1002,20])
    trig=np.argmin(Data[2:1002,0])
    wf=Data[2:1002, pmt]
    wf=np.roll(wf, 100-trig)
    bl=np.median(wf[:100])
    wf=wf-bl
    blw=np.sqrt(np.mean(wf[:100]**2))
    rec[j]['blw']=blw
    rec[j]['height']=-np.amin(wf)
    rec[j]['maxi']=np.argmin(wf)
    rec[j]['bl']=bl
    # if np.argmin(wf)>250 and np.argmin(wf)<265:
    #     plt.plot(wf, 'k.')
    #     plt.show()
    j+=1
    id+=1

np.savez(path+'PMT{}/raw_wf'.format(pmt), rec=rec[:j-1])
