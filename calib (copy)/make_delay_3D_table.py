import matplotlib.pyplot as plt
import os, sys
import numpy as np
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
import itertools


path='/home/gerak/Desktop/DireXeno/190803/pulser/'
pmts=[0,1,2,3,4,5,6,7,8,9,10,11,17,14,15,16,18,19]

blw_cut=20
left=85
right=115
height_cut=27
d_cut=4

PEAKS=np.load(path+'AllPeaks.npz')['rec']
Peaks=PEAKS[np.logical_and(PEAKS['blw']<blw_cut, np.logical_and(PEAKS['t']>left, np.logical_and(PEAKS['t']<right,
    np.logical_and(PEAKS['h']>height_cut, PEAKS['d']>d_cut))))]

def make_d(Peaks):
    D=np.zeros((len(pmts), len(pmts)))
    PMTs, n=np.unique(Peaks['pmt'], return_counts=True)
    for pmt0, pmt1 in itertools.product(PMTs[n==1], PMTs[n==1]):
        if pmt1>pmt0:
            D[pmt0==pmts, pmt1==pmts]=Peaks[Peaks['pmt']==pmt1]['t']-Peaks[Peaks['pmt']==pmt0]['t']
    return D



D=np.zeros((len(pmts), len(pmts), 1000))
j=0
for i, id in enumerate(np.unique(Peaks['id'])):
    d=make_d(Peaks[Peaks['id']==id])
    if len(np.nonzero(np.abs(d)>0)[0])>1:
        D[:,:,j]=d
        j+=1
        if j==len(D[0,0,:]):
            np.savez(path+'subDelay3DTable{}'.format(i), D=D)
            j=0
    if i%100==0:
        print(i, 'out of', len(Peaks['id']), '({} events were saved)'.format(j))
if j>2:
    np.savez(path+'subDelay3DTable{}'.format(i), D=D[:,:,:j-1])

D=np.zeros((len(pmts), len(pmts), 1))
for filename in os.listdir(path):
    if filename.startswith('subDelay3DTable'):
        D=np.dstack((D, np.load(path+filename)['D']))
        os.remove(path+filename)
        print(filename)
np.savez(path+'Delay3DTable', D=D[:,:,1:])
