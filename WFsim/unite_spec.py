import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pmt=4
spectrum=np.zeros(1000)
mean_WF=np.zeros(1000)
Recon_wf=np.zeros(1000)
Chi2=np.array([])
ID=np.array([])
for file in os.listdir('PMT{}'.format(pmt)):
    if file.startswith('spectra'):
        Data=np.load('PMT{}/'.format(pmt)+file)
        spectrum=np.vstack((spectrum, Data['spectrum']))
        Recon_wf+=Data['Recon_wf']
        mean_WF+=Data['mean_WF']
        Chi2=np.append(Chi2, Data['Chi2'])
        ID=np.append(ID, Data['ID'])
        os.remove('PMT{}/'.format(pmt)+file)
np.savez('PMT{}/spectra.npz'.format(pmt), spectrum=spectrum[1:], mean_WF=mean_WF, Recon_wf=Recon_wf, Chi2=Chi2, ID=ID)
