import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

source='Co57'
type=''
pmt=0
spectrum=np.zeros(1000)
mean_WF=np.zeros(1000)
Recon_wf=np.zeros(1000)
Chi2=np.array([])
ID=np.array([])
path='/home/gerak/Desktop/DireXeno/190803/'+source+type+'/'
for file in os.listdir(path+'PMT{}'.format(pmt)):
    if file.startswith('spectra'):
        Data=np.load(path+'PMT{}/'.format(pmt)+file)
        spectrum=np.vstack((spectrum, Data['spectrum']))
        Recon_wf+=Data['Recon_wf']
        mean_WF+=Data['mean_WF']
        Chi2=np.append(Chi2, Data['Chi2'])
        ID=np.append(ID, Data['ID'])
        print(path+'PMT{}/'.format(pmt)+file)
        os.remove(path+'PMT{}/'.format(pmt)+file)
np.savez(path+'PMT{}/spectra.npz'.format(pmt), spectrum=spectrum[1:], mean_WF=mean_WF, Recon_wf=Recon_wf, Chi2=Chi2, ID=ID)
