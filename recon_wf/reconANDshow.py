import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import poisson, binom
from fun import Recon_WFs, find_hits
from classes import WaveForm

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

first_id=13
id=first_id

start_time = time.time()
N=100
spectrum=np.zeros((len(pmts),1000, N))
mean_WF=np.zeros((len(pmts),1000))
Recon_wf=np.zeros((len(pmts),1000))
Chi2=np.zeros(N)
ID=np.zeros(N)
first_pmt=np.zeros(len(pmts))
BLW=np.zeros((len(pmts), N))

x=np.arange(1000)
plt.figure()
for i, [recon_wfs, chi2, recon_Hs, wfs, blw] in enumerate(Recon_WFs(file, spe, delay, dn, up, 100, height_cut, pmts, first_id)):
    if (first_id+i)%10==0:
        print('in', path, '. ID=', first_id+i)

    WF=WaveForm(100, np.sqrt(np.mean(np.sum(wfs, axis=0)[:40]**2)))
    find_hits(WF, np.sum(wfs, axis=0))
    init=sorted(WF.hits, key=lambda hit: hit.area)[-1].init
    wfs=np.roll(wfs, 100-init, axis=1)
    plt.plot(x, np.sum(wfs, axis=0), '.-')
    # plt.plot(x, np.sum(recon_wfs, axis=0), 'r.-')
    if i==2:
        break
plt.show()
