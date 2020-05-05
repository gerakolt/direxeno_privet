import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import poisson, binom
from fun import Recon_WFs

pmts=np.array([0,1,4,7,8,11])
spe=np.zeros((len(pmts), 1000))
height_cut=np.zeros(len(pmts))
delay=np.zeros(len(pmts))
for i, pmt in enumerate(pmts):
    Data=np.load('/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/area{}.npz'.format(pmt, pmt))
    Mpe=Data['Mpe']
    mean_spe_area=Data['mean_spe_area']
    Data=np.load('/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/cuts{}.npz'.format(pmt, pmt))
    height_cut[i]=Data['height_cut']
    Data=np.load('/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/SPEs{}.npz'.format(pmt, pmt))
    spe[i]=np.mean(Data['SPE'], axis=0)*(Mpe/mean_spe_area)
    if i>0:
        delay[i]=np.load('/home/gerak/Desktop/DireXeno/190803/pulser/Delays/delay_0_{}.npz'.format(pmt))['M']


dn=12
up=6
path='/home/gerak/Desktop/DireXeno/190803/Co57/'
file=open(path+'out.DXD', 'rb')

first_id=0
id=first_id

start_time = time.time()
spectrum=np.zeros((len(pmts),1000, 1000))
mean_WF=np.zeros((len(pmts),1000))
Recon_wf=np.zeros((len(pmts),1000))
Chi2=np.zeros((len(pmts),1000))
ID=np.zeros(1000)
first_pmt=np.zeros(len(pmts))

x=np.arange(1000)
for i, [recon_wfs, chi2, recon_Hs, wfs, BLW] in enumerate(Recon_WFs(file, spe, delay, dn, up, 100, height_cut, pmts, first_id)):
    init=np.amin(np.nonzero(recon_Hs>0)[1])
    first_pmt[recon_Hs[:,init]>0]+=1
    recon_wfs=np.roll(recon_wfs, 200-init, axis=1)
    wfs=np.roll(wfs, 200-init, axis=1)
    recon_Hs=np.roll(recon_Hs, 200-init, axis=1)
    event=np.sum(wfs, axis=0)
    recon_event=np.sum(recon_wfs, axis=0)
    for j in range(len(pmts)):
        plt.figure()
        plt.plot(x, wfs[j], 'k.', label='oreginal wf')
        plt.plot(x, recon_wfs[j], 'r.-', label='recon wf')
        plt.fill_between(x, y1=-BLW[j], y2=0)
        plt.title('PMT{}'.format(pmts[j]))
        plt.legend()
    plt.figure()
    plt.plot(x, event, 'k.')
    plt.plot(x, recon_event, 'r.-')
    plt.title('Event')

    plt.figure()
    plt.plot(pmts, first_pmt, 'ko')
    plt.show()
