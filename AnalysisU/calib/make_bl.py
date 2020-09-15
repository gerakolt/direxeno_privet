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
pmt=14
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
data=np.load(path+'cuts.npz')
blw_cut=data['blw_cut']
height_cut=data['height_cut']
left=data['left']
right=data['right']



PMT_num=20
time_samples=1024
path='/home/gerak/Desktop/DireXeno/190803/pulser/'.format(pmt)
file=open(path+'out.DXD', 'rb')

id=0
j=0
start_time = time.time()
BL=np.zeros(1000)
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
    wf-=bl
    blw=np.sqrt(np.mean((wf[:100])**2))
    h=-np.amin(wf[left:right])
    maxi=np.argmin(wf)
    if blw<blw_cut and h<height_cut:
        BL+=wf

    j+=1
    id+=1
np.savez(path+'PMT{}/BL'.format(pmt), BL=BL/j)
plt.figure()
plt.plot(BL/j, 'k.')
plt.show()
