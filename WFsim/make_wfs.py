import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import poisson, binom
from classes import WaveForm, Hit
from fun import find_hits


pmt=1
events=15000
N=50
tau=45
St=0.7

Data=np.load('/home/gerak/Desktop/DireXeno/190803/pulser/SPEs{}.npz'.format(pmt))
SPEs=Data['SPE']
Data=np.load('/home/gerak/Desktop/DireXeno/190803/pulser/area{}.npz'.format(pmt))
Mpe=Data['Mpe']
Spe=Data['Spe']
mean_spe_area=Data['mean_spe_area']

spe=np.mean(SPEs, axis=0)*Mpe/mean_spe_area
for i in range(len(SPEs[:,0])):
    SPEs[i]=spe*np.random.normal(1,Spe/Mpe)
    print(i, 'out of', len(SPEs[:,0]))
# zeros=Data['zeros']
# SPEs[:,zeros]=0

def shoot_pes(SPEs, t):
    for i, spe in enumerate(SPEs):
        pe=np.zeros(1000)
        if t[i]>=1000:
            # yield pe
            continue
        elif np.argmin(spe)<t[i]:
            try:
                pe[t[i]-np.argmin(spe):]+=spe[:1000-(t[i]-np.argmin(spe))]
            except:
                print(t[i]-np.argmin(spe), t[i], np.argmin(spe))
                sys.exit()
        else:
            pe[:1000-(np.argmin(spe)-t[i])]+=spe[np.argmin(spe)-t[i]:]
        yield pe

def make_event(events, SPEs, n, tau, St):
    id=0
    while events>0:
        N=np.random.poisson(n)
        Rec=[]
        print(events)
        wf=np.zeros(1000)
        events-=1
        t=np.round(np.random.normal(200+np.random.exponential(tau*5, N), St*5, N)).astype(int)
        h, bins=np.histogram(t, bins=1000, range=[-0.5, 999.5])
        SPEids=np.random.randint(0,len(SPEs),N)
        for pe in shoot_pes(SPEs[SPEids], t):
            wf+=pe
        wf=wf-np.median(wf[:150])
        blw=np.sqrt(np.mean(wf[:150]**2))
        WF=WaveForm(100, blw)
        find_hits(WF, wf)
        rec=np.recarray(len(WF.hits), dtype=[
                        ('id', 'i8'),
                        ('blw', 'f8'),
                        ('init', 'i8'),
                        ('height', 'f8')])
        for i, hit in enumerate(WF.hits):
            rec[i]=id, blw, hit.init, hit.height
        yield [h, wf, rec]
        id+=1


H=np.zeros((events, 1000))
WF=np.zeros((events, 1000))
Rec=[]
for i, [h, wf, rec] in enumerate(make_event(events, SPEs, N, tau, St)):
    H[i]=h
    WF[i]=wf
    Rec.extend(rec)
np.savez('PMT{}/simWFs'.format(pmt), H=H, WF=WF, N=N, tau=tau, St=St, hits=Rec)
