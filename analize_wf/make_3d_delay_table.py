import matplotlib.pyplot as plt
import os, sys
import numpy as np
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
import itertools


source='Co57'
type=''
path='/home/gerak/Desktop/DireXeno/190803/'+source+type+'/'
pmts=[0,1,2,3,4,5,6,7,8,9,10,11,17,14,15,16,18,19]

blw_cut=20
left=25
right=200
height_cut=110

HITs=np.load(path+'Allhits.npz')['rec']
Hits=HITs[np.logical_and(HITs['blw']<blw_cut, HITs['height_first_hit']>height_cut)]

def make_d(Hits):
    D=np.zeros((len(pmts), len(pmts)))
    PMTs, n=np.unique(Hits['pmt'], return_counts=True)
    for pmt0, pmt1 in itertools.product(PMTs[n==1], PMTs[n==1]):
        if pmt1>pmt0:
            D[pmt0==pmts, pmt1==pmts]=Hits[Hits['pmt']==pmt1]['init10']-Hits[Hits['pmt']==pmt0]['init10']
    return D



D=np.zeros((len(pmts), len(pmts), 1000))
j=0
for i, id in enumerate(np.unique(Hits['id'])):
    d=make_d(Hits[Hits['id']==id])
    if len(np.nonzero(np.abs(d)>0)[0])>1:
        D[:,:,j]=d
        j+=1
        if j==len(D[0,0,:]):
            np.savez(path+'subDelay3DTable{}'.format(i), D=D)
            j=0
    if i%100==0:
        print(i, 'out of', len(Hits['id']), '({} events were saved)'.format(j))
if j>2:
    np.savez(path+'subDelay3DTable{}'.format(i), D=D[:,:,:j-1])

D=np.zeros((len(pmts), len(pmts), 1))
for filename in os.listdir(path):
    if filename.startswith('subDelay3DTable'):
        D=np.dstack((D, np.load(path+filename)['D']))
        os.remove(path+filename)
        print(filename)
np.savez(path+'Delay3DTable', D=D[:,:,1:])
