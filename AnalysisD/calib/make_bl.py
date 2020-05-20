import numpy as np
import time
import matplotlib.pyplot as plt


def find_trig(wf):
    bl=np.median(wf[:200])
    top=np.median(wf[325:475])
    mid=0.5*(bl+top)
    return np.argmin(np.abs(wf[:400]-mid))

PMT_num=12
time_samples=1024
path='/home/gerak/Desktop/DireXeno/130520/pulser/'
file=open(path+'out.DXD', 'rb')

pmts=np.array([0,0,0,0,0,5,10,11,13,15,16,18,19])
chns=[0,1,2,3,4,5,6,7,8,9,10,11,13]

pmt=18
BL=np.zeros(1000)
h_cut=40
data=np.load(path+'PMT{}/cuts.npz'.format(pmt))
blw_cut=data['blw_cut']
left=data['left']
right=data['right']

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
    trig=find_trig(Data[2:1002,PMT_num+2])
    wf=Data[2:1002, chns[np.nonzero(pmts==pmt)[0][0]]]
    wf=np.roll(wf, 100-trig)
    bl=np.median(wf[:100])
    wf=wf-bl
    blw=np.sqrt(np.mean(wf[:100]**2))
    if blw<blw_cut and np.amin(wf[left:right])>-h_cut:
        BL+=wf
        j+=1
    id+=1
x=np.arange(1000)
plt.plot(x, BL/j, 'k.')
plt.show()
np.savez(path+'PMT{}/BL'.format(pmt), BL=BL/j)
