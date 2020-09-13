import numpy as np
import time
import matplotlib.pyplot as plt


def find_trig(wf):
    bl=np.median(wf[:200])
    top=np.median(wf[325:475])
    mid=0.5*(bl+top)
    return np.argmin(np.abs(wf[:400]-mid))

pmts=np.array([0,1,4,7,8,14])
chns=[2,3,6,9,10,15]
pmt=7
ID=32
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
data=np.load(path+'cuts.npz')
blw_cut=data['blw_cut']
height_cut=data['height_cut']
left=data['left']
right=data['right']
# BL=np.load(path+'BL.npz')['BL']
BL=np.zeros(1000)


PMT_num=20
time_samples=1024
path='/home/gerak/Desktop/DireXeno/190803/pulser/'.format(pmt)
file=open(path+'out.DXD', 'rb')

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
    wf=Data[2:1002, chns[np.nonzero(pmts==pmt)[0][0]]]
    wf=np.roll(wf, 100-trig)
    bl=np.median(wf[:100])
    wf-=bl+BL
    blw=np.sqrt(np.mean((wf[:100])**2))
    h=-np.amin(wf[left:right])
    maxi=np.argmin(wf)
    area=-(np.sum(wf[maxi-100:maxi+200])+np.sum(wf[maxi-50:maxi+150])+np.sum(wf[maxi-100:maxi+150])+np.sum(wf[maxi-50:maxi+200]))/4
    if len(np.nonzero(wf[:maxi]>-0.1*h)[0])>0:
        init10=np.amax(np.nonzero(wf[:maxi]>-0.1*h)[0])
    else:
        init10=0

    if id==ID:
        x=np.arange(1000)
        plt.figure()
        plt.plot(x, wf, 'k.', label='area={}'.format(area))
        plt.fill_between(x[maxi-100:maxi+200], y1=np.amin(wf), y2=0)
        plt.fill_between(x[maxi-50:maxi+150], y1=np.amin(wf), y2=0)
        plt.legend()
        plt.show()


    j+=1
    id+=1
