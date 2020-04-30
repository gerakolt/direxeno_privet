import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import poisson, binom
from classes import WaveForm, Hit
from fun import find_hits, Recon_WF, find_peaks


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
factor=Data['factor']

# print(np.round(np.random.normal(200+np.random.exponential(tau*5, N), St*5, N)).astype(int))
# print(np.random.randint(0,len(SPEs),N))

wf=np.zeros(1000)
t=[400,  380, 1151,  390,  318,  226,  506,  288,  223,  270,  382,  670,  251,  363,
  417,  198,  560,  224,  416,  518,  236,  326, 781,  399,  283,  425,  249,  350,
  613,  488,  256,  343,  319,  316, 203,  249,  267,  267,  308,  461,  415,  245,
  291,  711,  218,  544,  335,  223,  305,  430]
I=[3557, 7069, 6653, 3887, 5910, 8067, 1844, 2069,  901, 5913, 2918, 2872, 4890, 6657,
 6882,   76,   92, 4789, 3090, 3378, 6136, 4754, 4159,  734, 6504, 6960,  528, 3240,
 1565, 6981, 5701, 6789, 4628, 2590, 1299, 6840, 1115, 4698, 4951, 4129, 3018, 6330,
 3392, 6109, 7243, 2263, 6345, 4837, 3105, 6713]



def shoot_pes(SPEs, t):
    for i, spe in enumerate(SPEs):
        # print(i, np.median(spe[:150]))
        area=0
        for peak in find_peaks(np.array(spe), np.sqrt(np.mean(spe[:150]**2)), [0]):
            if peak.area>area:
                area=peak.area
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
        yield pe, area

fig=plt.figure()

ax=fig.add_subplot(311)

areas=[]
for pe, area in shoot_pes(SPEs[I], t):
    wf+=pe
    ax.plot(pe, 'k--', alpha=0.2)
    areas.append(area)
wf=wf-np.median(wf[:150])

spe=np.sum(SPEs, axis=0)
spe=(spe-np.median(spe))/factor
[recon_wf, chi2, recon_H]=next(Recon_WF([wf], spe, 12, 6, 100))



ax.plot(wf, 'k.')
ax.plot(recon_wf, 'r--')
ax.plot(np.roll(spe, np.argmin(wf)-np.argmin(spe)), 'g--')


h=np.histogram(t, bins=1000, range=[-0.5, 999.5])[0]
ax=fig.add_subplot(312)
ax.plot(h, 'ko', label=np.sum(h))
ax.plot(recon_H, 'ro', label=np.sum(recon_H))
ax.legend()

ax=fig.add_subplot(313)
ax.hist(areas, histtype='step')
area=0
for peak in find_peaks(np.array(spe), np.sqrt(np.mean(spe[:150]**2)), [0]):
    if peak.area>area:
        area=peak.area
ax.axvline(x=area, ymin=0, ymax=1, color='k')
plt.show()
