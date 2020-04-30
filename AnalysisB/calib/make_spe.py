import numpy as np
import time
import matplotlib.pyplot as plt


def find_trig(wf):
    bl=np.median(wf[:200])
    top=np.median(wf[325:475])
    mid=0.5*(bl+top)
    return np.argmin(np.abs(wf[:400]-mid))

pmt=19
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
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

id=0
j=0
start_time = time.time()
rec=np.recarray(100000, dtype=[
    ('height', 'i8'),
    ('blw', 'f8'),
    ('area', 'f8'),
    ('init10', 'i8'),
    ('id', 'i8'),
    ('rise_time', 'i8'),
    ('maxi', 'i8'),
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
    # trig=find_trig(Data[2:1002,20])
    trig=np.argmin(Data[2:1002,0])
    wf=Data[2:1002, pmt]
    wf=np.roll(wf, 100-trig)
    bl=np.median(wf[:100])
    wf-=bl
    blw=np.sqrt(np.mean((wf[trig-100:trig])**2))
    wf-=BL
    rec[j]['blw']=blw
    h=-np.amin(wf[left:right])
    rec[j]['height']=h
    maxi=left+np.argmin(wf[left:right])
    rec[j]['area']=-(np.sum(wf[maxi-100:maxi+200])+np.sum(wf[maxi-50:maxi+150])+np.sum(wf[maxi-100:maxi+150])+np.sum(wf[maxi-50:maxi+200]))/4
    rec[j]['maxi']=maxi
    init10=np.amax(np.nonzero(wf[:maxi]>-0.1*h)[0])
    rec[j]['init10']=init10
    rec[j]['id']=id
    if h<height_cut:
        rec[j]['rise_time']=25
        rec[j]['spe']=np.zeros(1000)
    else:
        rec[j]['rise_time']=maxi-init10
        rec[j]['spe']=np.roll(wf, 200-init10)
    j+=1
    id+=1
np.savez(path+'PMT{}/spe'.format(pmt), rec=rec[:j-1])
