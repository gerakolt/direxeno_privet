import numpy as np
import time
import matplotlib.pyplot as plt


def find_trig(wf):
    bl=np.median(wf[:200])
    top=np.median(wf[325:475])
    mid=0.5*(bl+top)
    return np.argmin(np.abs(wf[:400]-mid))

pmt=18
path='/home/gerak/Desktop/DireXeno/130520/pulser/PMT{}/'.format(pmt)
data=np.load(path+'cuts.npz')
blw_cut=data['blw_cut']
height_cut=data['height_cut']
left=data['left']
right=data['right']
BL=np.load(path+'BL.npz')['BL']


pmts=np.array([0,0,0,0,0,5,10,11,13,15,16,18,19])
chns=[0,1,2,3,4,5,6,7,8,9,10,11,13]
PMT_num=12
time_samples=1024
path='/home/gerak/Desktop/DireXeno/130520/pulser/'.format(pmt)
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
    ('dh3', 'f8'),
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
    trig=find_trig(Data[2:1002,PMT_num+2])
    wf=Data[2:1002, chns[np.nonzero(pmts==pmt)[0][0]]]
    wf=np.roll(wf, 100-trig)
    bl=np.median(wf[:100])
    wf-=bl+BL
    blw=np.sqrt(np.mean((wf[:100])**2))
    rec[j]['blw']=blw
    h=-np.amin(wf[left:right])
    rec[j]['height']=h
    maxi=left+np.argmin(wf[left:right])
    rec[j]['area']=-(np.sum(wf[maxi-100:maxi+200])+np.sum(wf[maxi-50:maxi+150])+np.sum(wf[maxi-100:maxi+150])+np.sum(wf[maxi-50:maxi+200]))/4
    rec[j]['maxi']=maxi
    init10=np.amax(np.nonzero(wf[:maxi]>-0.1*h)[0])
    rec[j]['rise_time']=maxi-init10
    rec[j]['dh3']=(h+wf[maxi-3])/h
    rec[j]['init10']=init10
    rec[j]['id']=id

    # if maxi-init10>4 and maxi-init10<10 and h>50 and h<80:
    #     plt.figure()
    #     x=np.arange(1000)
    #     plt.plot(x, wf, 'k.')
    #     plt.plot(init10, wf[init10], 'ro')
    #     plt.show()
    if h<height_cut:
        rec[j]['spe']=np.zeros(1000)
    else:
        rec[j]['spe']=np.roll(wf, 200-init10)

    j+=1
    id+=1
np.savez(path+'PMT{}/spe'.format(pmt), rec=rec[:j-1])
