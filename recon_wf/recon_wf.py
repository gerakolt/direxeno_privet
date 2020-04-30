import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import poisson, binom
from fun import Recon_WF

PMT_num=20
time_samples=1024
pmts=np.array([0,1,2,3,4,5,6,7,8,9,10,11,17,14,15,16,18,19])
def read_data(file, pmt):
    stop=0
    while stop==0:
        Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
        if len(Data)<(PMT_num+4)*(time_samples+2):
            break
        Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
        data=Data[2:1002,2+np.nonzero(pmt==pmts)[0]][:,0]
        bl=np.median(data[:40])
        yield np.array(data-bl)

pmt=0
Data=np.load('/home/gerak/Desktop/DireXeno/190803/pulser/SPEs{}.npz'.format(pmt))
spe=np.sum(Data['SPE'], axis=0)
spe=(spe-np.median(spe[:150]))/Data['factor']
spe[Data['zeros']]=0

blw_cut=20
height_cut=60
dn=12
up=6

id0=1051
source='Co57'
type=''
path='/home/gerak/Desktop/DireXeno/190803/'+source+type+'/'
file=open(path+'out.DXD', 'rb')
np.fromfile(file, np.float32, (PMT_num+4)*(2+time_samples)*id0)

start_time = time.time()
spectrum=np.zeros((1000, 1000))
mean_WF=np.zeros(1000)
Recon_wf=np.zeros(1000)
Chi2=np.zeros(1000)
ID=np.zeros(1000)

j=0
for id, wf in enumerate(read_data(file, pmt)):
    if id%10==0:
        print('In '+source+type+', ID=', id0+id, '({} events were made.)'.format(j))
    if np.sqrt(np.mean(wf[:40]**2))>blw_cut:
        continue
    recon_wf, chi2, recon_H=Recon_WF(wf, spe, dn, up, height_cut)
    if np.all(recon_H==0):
        continue
    spectrum[j]=recon_H
    mean_WF+=wf
    Recon_wf+=recon_wf
    Chi2[j]=chi2
    ID[j]=id0+id
    j+=1
    if j==len(ID):
        np.savez(path+'PMT{}/spectra{}'.format(pmt, id0+id), spectrum=spectrum, mean_WF=mean_WF, Recon_wf=Recon_wf, Chi2=Chi2, ID=ID)
        j=0
        mean_WF=np.zeros(1000)
        Recon_wf=np.zeros(1000)
if j>1:
    np.savez(path+'PMT{}/spectra{}'.format(pmt, id), spectrum=spectrum[:j-1], mean_WF=mean_WF[:j-1], Recon_wf=Recon_wf[:j-1], Chi2=Chi2[:j-1], ID=ID[:j-1])

for file in os.listdir(path+'PMT{}'.format(pmt)):
    if file.startswith('spectra'):
        Data=np.load(path+'PMT{}/'.format(pmt)+file)
        spectrum=np.vstack((spectrum, Data['spectrum']))
        Recon_wf+=Data['Recon_wf']
        mean_WF+=Data['mean_WF']
        Chi2=np.append(Chi2, Data['Chi2'])
        ID=np.append(ID, Data['ID'])
        os.remove(path+'PMT{}/'.format(pmt)+file)
np.savez(path+'PMT{}/spectra.npz'.format(pmt), spectrum=spectrum[1:], mean_WF=mean_WF, Recon_wf=Recon_wf, Chi2=Chi2, ID=ID)
