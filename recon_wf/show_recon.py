import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import poisson, binom
from fun import Recon_WF, Show_Recon_WF

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
p=1
Data=np.load('/home/gerak/Desktop/DireXeno/190803/pulser/SPEs{}.npz'.format(pmt))
spe=np.sum(Data['SPE'], axis=0)
spe=(spe-np.median(spe[:150]))/Data['factor']
spe[Data['zeros']]=0

blw_cut=20
height_cut=60
dn=12
up=6

id0=13538
source='Co57'
type=''
path='/home/gerak/Desktop/DireXeno/190803/'+source+type+'/'
file=open(path+'out.DXD', 'rb')
np.fromfile(file, np.float32, (PMT_num+4)*(2+time_samples)*id0)

wf=next(read_data(file, pmt))
recon_wf, chi2, recon_H=Show_Recon_WF(wf, spe, dn, up, height_cut, p)
x=np.arange(1000)
plt.figure()
plt.plot(x, wf, 'k.-', label='wf')
plt.plot(x, recon_wf, 'r.-', label='recon_wf')
plt.legend()
plt.show()
