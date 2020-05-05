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
BL=np.load(path+'PMT{}/bl.npz'.format(pmt))['BL']*0
WF=np.zeros(1000)

file=open(path+'out.DXD', 'rb')
id=0
j=0
start_time = time.time()
REC=[]
rec=np.recarray(5000, dtype=[
    ('height', 'i8'),
    ('area', 'f8'),
    ('t', 'i8'),
    ('id', 'i8'),
    ('rise_time', 'i8'),
    ('spe', 'f8', 1000)
    ])
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
    wf=wf-np.median(wf[:init])-BL
    blw=np.sqrt(np.mean(wf[:init]**2))
    if blw>blw_cut:
        id+=1
        continue
    h=-np.amin(wf[left:right])
    rec[j]['height']=h
    rec[j]['area']=-np.sum((wf)[left:right])
    # if h>height_cut:
    #     plt.figure()
    #     x=np.arange(1000)
    #     plt.plot(x, wf, 'k.')
    #     plt.plot(x, wf+BL, 'g.', alpha=0.5)
    #     plt.fill_between(x[left:right], y1=np.amin(wf), y2=0, alpha=0.3)
    #     plt.show()
    maxi=left+np.argmin(wf[left:right])
    init10=np.amax(np.nonzero(wf[:maxi]>-0.1*h)[0])
    rec[j]['t']=init10
    rec[j]['id']=id
    rec[j]['rise_time']=maxi-init10
    rec[j]['spe']=np.roll(wf, 200-init10)
    WF+=wf
    j+=1
    id+=1
    if j==5000:
        REC.extend(rec)
        j=0

np.savez(path+'PMT{}/spe'.format(pmt), rec=REC, WF=WF/len(REC))
