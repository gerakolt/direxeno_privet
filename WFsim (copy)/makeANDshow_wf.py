import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import poisson, binom
from classes import WaveForm, Hit
from fun import find_hits, Recon_WF


height_cut=35
# height_cut=0
pmt=1
events=5000
N=50
tau=45
St=0.7

Data=np.load('/home/gerak/Desktop/DireXeno/190803/pulser/SPEs{}.npz'.format(pmt))
SPEs=Data['SPE']
zeros=Data['zeros']
SPEs[:,zeros]=0
factor=Data['factor']

spe=np.mean(SPEs, axis=0)
spe=(spe-np.median(spe[:150]))/factor

plt.figure()
plt.plot(np.mean(SPEs, axis=0), 'k.', label='Shooted SPEs')
plt.plot(spe, 'r.', label='spe for reconstruction')
plt.legend()
plt.show()

while True:
    wf=np.zeros(1000)
    t=np.round(np.random.normal(200+np.random.exponential(tau*5, N), St*5, N)).astype(int)
    I=np.random.randint(0,len(SPEs),N)

    def shoot_pes(SPEs, t):
        for i, spe in enumerate(SPEs):
            pe=np.zeros(1000)
            if t[i]>=1000:
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


    PEs=np.zeros((len(I), 1000))
    for i, pe in enumerate(shoot_pes(SPEs[I], t)):
        wf+=pe
        PEs[i]=pe
    wf=wf-np.median(wf[:40])

    [recon_wf, chi2, recon_H]=next(Recon_WF([wf], spe, 12, 6, 100, 0))
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(wf, 'k.')
    ax.plot(recon_wf, 'r--')
    ax.legend()

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(np.histogram(t, bins=1000, range=[-0.5, 999.5])[0], 'k1', label=len(t))
    ax.plot(recon_H, 'r2', label=np.sum(recon_H))
    ax.legend()
    plt.show()



# h=np.histogram(t, bins=1000, range=[-0.5, 999.5])[0]
# ax=fig.add_subplot(212)
# ax.plot(h, 'k1', label=np.sum(h))
# ax.plot(recon_H, 'r2', label=np.sum(recon_H))
# ax.legend()

plt.show()
