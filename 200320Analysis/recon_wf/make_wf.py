import numpy as np
import time
from classes import WaveForm
from fun import find_hits

pmt=8
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
data=np.load(path+'spe.npz')
height_cut=data['height_cut']
rise_time_cut=data['rise_time_cut']

PMT_num=20
time_samples=1024
path='/home/gerak/Desktop/DireXeno/190803/Co57/'
file=open(path+'out.DXD', 'rb')

blw_cut=20
WF=np.zeros(1000)
BL=np.zeros(1000)

id=0
j=0
start_time = time.time()
REC=[]
rec=np.recarray(5000, dtype=[
    ('blw', 'i8'),
    ('hit_init', 'i8'),
    ('hit_area', 'i8')
    ])
n_WF=0
n_BL=0
init=70
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
    wf=wf-np.median(wf[:init])
    blw=np.sqrt(np.mean(wf[:init]**2))
    rec[j]['blw']=blw
    rec[j]['hit_init']=0
    rec[j]['hit_area']=0
    if blw<blw_cut:
        waveform=WaveForm(blw)
        find_hits(waveform, wf, init, height_cut, rise_time_cut)
        if len(waveform.hits)>0:
            rec[j]['hit_init']=waveform.hits[0].init
            rec[j]['hit_area']=waveform.hits[0].area
            # if waveform.hits[0].area<75000:
            #     print(id)
            #     show_wf(waveform, wf)
            WF+=wf
            n_WF+=1
        else:
            BL+=wf
            n_BL+=1

    j+=1
    id+=1
    if j==5000:
        REC.extend(rec)
        j=0

np.savez(path+'PMT{}/raw_wf'.format(pmt), rec=REC, WF=WF/n_WF, BL=BL/n_BL, init=init, blw_cut=blw_cut, height_cut=height_cut, rise_time_cut=rise_time_cut)
