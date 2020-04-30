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
pmt=0
events=5000
N=50
tau=45
St=0.7

Data=np.load('/home/gerak/Desktop/DireXeno/190803/pulser/SPEs{}.npz'.format(pmt))
SPEs=Data['SPE']

spe=np.mean(SPEs, axis=0)

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

while True:
    wf=np.zeros(1000)
    t=np.round(np.random.normal(200+np.random.exponential(tau*5, N), St*5, N)).astype(int)
    I=np.random.randint(0,len(SPEs),N)
    for i in range(len(SPEs[:,0])):
        SPEs[i]=spe
    # I=[15111,   918,  6463, 17149,  6096,  1526,  6606,  5866, 11504,  5395,  4728,  2008,
    #      11152,  5884, 16139,  7228, 16560,  4567, 11819,  5956,  4842, 13198, 10266, 13636,
    #      14259,  2441, 14658, 14576, 14091,  1512,  3233,  6116, 13543,   318, 12924,  9349,
    #       2230, 16273, 13504,  6740, 15178, 14938,  1307,  2287, 12190,  9439,  5513, 11685,
    #       1866, 12717]
    # t=np.array([384,  422,  399,  238,  353,  404,  226,  223,  275,  241,  402,  696,  416,  617,
    #     414, 1015,  522,  508,  375,  250,  237, 1326,  736,  313,  358,  251,  561,  246,
    #         605,  685,  258,  350,  396,  514,  336,  760,  300 , 208 , 440,  256,  383,  319,
    #         278, 315 , 325,  271,  264,  297,  201,  442])

    print(t)
    print(I)


    PEs=np.zeros((len(I), 1000))
    for i, pe in enumerate(shoot_pes(SPEs[I], t)):
        wf+=pe
        PEs[i]=pe
    wf=wf-np.median(wf[:40])

    plt.figure()
    plt.plot(np.mean(SPEs[I], axis=0), 'k.', label='Shooted SPEs')
    plt.plot(spe, 'r.', label='spe for reconstruction')
    plt.legend()
    plt.show()

    [recon_wf, chi2, recon_H]=next(Recon_WF([wf], spe, 12, 6, 100, height_cut))

    l=196
    r=210
    # r2=295
    #
    x=np.arange(1000)
    # sub_wf=np.zeros(1000)
    # fig=plt.figure()
    # ax=fig.add_subplot(211)
    # for i in range(l,r):
    #     J=np.nonzero(t==i)[0]
    #     if len(J)>0:
    #         for j in J:
    #             temp_wf=next(shoot_pes([SPEs[I[j]]], [i]))
    #             sub_wf+=temp_wf
    #             ax.plot(x, temp_wf, 'k.-', label=i)
    # ax.plot(x, sub_wf, 'r.-')
    # ax.legend()
    #
    # ax=fig.add_subplot(212)
    # sub_wf=np.zeros(1000)
    # for i in range(r,r2):
    #     J=np.nonzero(t==i)[0]
    #     if len(J)>0:
    #         for j in J:
    #             temp_wf=next(shoot_pes([SPEs[I[j]]], [i]))
    #             sub_wf+=temp_wf
    #             ax.plot(x, temp_wf, 'k.-', label=i)
    # ax.plot(x, sub_wf, 'r.-', alpha=0.2)
    # ax.legend()


    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(x, wf, 'k.')
    ax.plot(x, recon_wf, 'r--')
    ax.fill_between(x[l:r], y1=wf[l:r], y2=0, color='y')
    ax.legend()

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(np.histogram(t, bins=1000, range=[-0.5, 999.5])[0], 'k1', label=len(t[t<1000]))
    ax.plot(recon_H, 'r2', label=np.sum(recon_H))
    ax.fill_between(x[l:r], y1=0, y2=recon_H[l:r], color='y')
    ax.legend()
    plt.show()



# h=np.histogram(t, bins=1000, range=[-0.5, 999.5])[0]
# ax=fig.add_subplot(212)
# ax.plot(h, 'k1', label=np.sum(h))
# ax.plot(recon_H, 'r2', label=np.sum(recon_H))
# ax.legend()

plt.show()
