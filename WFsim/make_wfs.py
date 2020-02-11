import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import poisson, binom
from classes import WaveForm, Hit
from fun import find_hits

height_cut=26

pmt=4
events=5000
N=50
tau=45
St=0.7

Data=np.load('/home/gerak/Desktop/DireXeno/pulser_190803_46211/PMT{}/AllSPEs.npz'.format(pmt))
SPEs=Data['SPE']
SPEs=SPEs[np.nonzero(np.amin(SPEs, axis=1)<-height_cut)[0]]
zeros=Data['zeros']
SPEs[:,zeros]=0

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

def make_event(events, SPEs, N, tau, St):
    id=0
    while events>0:
        Rec=[]
        print(events)
        wf=np.zeros(1000)
        events-=1
        t=np.round(np.random.normal(200+np.random.exponential(tau*5, N), St*5, N)).astype(int)
        h, bins=np.histogram(t, bins=1000, range=[-0.5, 999.5])
        for pe in shoot_pes(SPEs[np.random.randint(0,len(SPEs),N)], t):
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
