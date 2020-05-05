import numpy as np
import time
from classes import WaveForm
from fun import Recon_wf
from scipy.stats import poisson, binom

pmt=8
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
data=np.load(path+'spe.npz')
height_cut=data['height_cut']
rise_time_cut=data['rise_time_cut']
spe=data['Sig_trig']

PMT_num=20
time_samples=1024
path='/home/gerak/Desktop/DireXeno/190803/Co57/'
data=np.load(path+'PMT{}/raw_wf.npz'.format(pmt))
blw_cut=data['blw_cut']
init=data['init']
file=open(path+'out.DXD', 'rb')



WF=np.zeros(1000)
recon_WF=np.zeros(1000)
BL=np.zeros(1000)

id=0
Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*id)
j=0
start_time = time.time()
REC=[]
rec=np.recarray(5000, dtype=[
    ('area', 'i8'),
    ('id', 'i8'),
    ('chi2', 'f8'),
    ('h', 'i8', 1000),
    ('init', 'i8'),
    ])
n=0
n_bl=0
while id<1e6:
    print(id)
    Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
    if len(Data)<(PMT_num+4)*(time_samples+2):
        break
    Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
    trig=np.argmin(Data[2:1002,0])
    wf=Data[2:1002, pmt+2]
    wf=wf-np.median(wf[:init])
    blw=np.sqrt(np.mean(wf[:init]**2))
    if blw>blw_cut or np.amin(wf)<-4000:
        id+=1
        continue
    waveform=WaveForm(blw)
    h, recon_wf=Recon_wf(waveform, wf, init, height_cut, rise_time_cut, spe, 6, 12)
    if np.any(h>0):
        rec[j]['area']=-np.sum(wf[init:])
        rec[j]['id']=id
        rec[j]['chi2']=np.sum((wf[init:]-recon_wf[init:])**2)
        rec[j]['init']=np.amin(np.nonzero(h>0)[0])
        rec[j]['h']=np.roll(h, -rec[j]['init'])
        WF+=wf
        recon_WF+=recon_wf
        n+=1
        j+=1
    else:
        BL+=wf
        n_bl+=1
    id+=1
    if j==5000:
        REC.extend(rec)
        j=0
if n_bl==0:
    n_bl=1
np.savez(path+'PMT{}/recon_wf'.format(pmt), rec=REC, WF=WF/n, recon_WF=recon_WF/n, BL=BL/n_bl)
