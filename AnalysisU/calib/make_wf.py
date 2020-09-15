import numpy as np
import time
import matplotlib.pyplot as plt


def find_trig(wf):
    bl=np.median(wf[:200])
    top=np.median(wf[325:475])
    mid=0.5*(bl+top)
    return np.argmin(np.abs(wf[:400]-mid))

PMT_num=20
time_samples=1024
path='/home/gerak/Desktop/DireXeno/190803/pulser/'
file=open(path+'out.DXD', 'rb')

pmts=[0,1,4,7,8,14]
chns=[2,3,6,9,10,15]

rec=np.recarray(100000, dtype=[
    ('blw', 'f8', len(pmts)),
    ('height', 'i8', len(pmts)),
    ('maxi', 'i8', len(pmts)),
    ('bl', 'i8', len(pmts)),
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
    # trig=find_trig(Data[2:1002,PMT_num+2])
    trig=np.argmin(Data[2:1002, 0])
    for i, pmt in enumerate(pmts):
        wf=Data[2:1002, chns[i]]
        wf=np.roll(wf, 100-trig)
        bl=np.median(wf[:100])
        wf=wf-bl
        blw=np.sqrt(np.mean(wf[:100]**2))
        rec[j]['blw'][i]=blw
        rec[j]['height'][i]=-np.amin(wf)
        rec[j]['maxi'][i]=np.argmin(wf)
        rec[j]['bl'][i]=bl
    j+=1
    id+=1

np.savez(path+'raw_wf'.format(pmt), rec=rec[:j-1])
