import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import poisson, binom
from fun import Recon_WF

pmt=0
Data=np.load('/home/gerak/Desktop/DireXeno/190803/pulser/SPEs{}.npz'.format(pmt))
SPEs=Data['SPE']
Data=np.load('/home/gerak/Desktop/DireXeno/190803/pulser/area{}.npz'.format(pmt))
Mpe=Data['Mpe']
Spe=Data['Spe']
mean_spe_area=Data['mean_spe_area']
spe=np.mean(SPEs, axis=0)*Mpe/mean_spe_area

h_init=100
height_cut=0
blw_cut=60
dn=12
up=6
Data=np.load('PMT{}/simWFs.npz'.format(pmt))
WFs=Data['WF']
H=Data['H']

first_id=0
id=first_id

start_time = time.time()
spectrum=np.zeros((1000, 1000))
mean_WF=np.zeros(1000)
PE_by_area=np.zeros(1000)
Recon_wf=np.zeros(1000)
dif=np.zeros(1000)
Chi2=np.zeros(1000)
ID=np.zeros(1000)

j=0
for i, [recon_wf, chi2, recon_H, area, wf] in enumerate(Recon_WF(WFs, spe, dn, up, h_init, height_cut)):
    if id%10==0:
        print('In SIM, PMT {}, Event number {} ({} sec for event).'.format(pmt, id, (time.time()-start_time)/10))
        start_time = time.time()
    dif+=recon_H-H[i]
    spectrum[j]=recon_H
    PE_by_area[j]=area/Mpe
    mean_WF+=wf
    Recon_wf+=recon_wf
    Chi2[j]=chi2
    ID[j]=id
    id+=1
    j+=1
    if j==1000:
        np.savez('PMT{}/spectra{}to{}'.format(pmt, first_id, id-1), spectrum=spectrum, mean_WF=mean_WF, Recon_wf=Recon_wf, Chi2=Chi2, ID=ID, dif=dif, PE_by_area=PE_by_area)
        first_id=id
        j=0
        mean_WF=np.zeros(1000)
        Recon_wf=np.zeros(1000)
        dif=np.zeros(1000)

#
# if j>0:
#     np.savez('PMT{}/spectra{}to{}'.format(pmt, first_id, id-1), spectrum=spectrum[:j], mean_WF=mean_WF[:j], Recon_wf=Recon_wf[:j], Chi2=Chi2[:j], ID=ID[:j], ID=ID[:j])
