import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pmt=0
spectrum=np.zeros(1000)
mean_WF=np.zeros(1000)
Recon_wf=np.zeros(1000)
dif=np.zeros(1000)
Chi2=np.array([])
ID=np.array([])
PE_by_area=np.array([])
for file in os.listdir('PMT{}'.format(pmt)):
    if file.startswith('spectra'):
        Data=np.load('PMT{}/'.format(pmt)+file)
        spectrum=np.vstack((spectrum, Data['spectrum']))
        Recon_wf+=Data['Recon_wf']
        mean_WF+=Data['mean_WF']
        dif+=Data['dif']
        Chi2=np.append(Chi2, Data['Chi2'])
        ID=np.append(ID, Data['ID'])
        PE_by_area=np.append(PE_by_area, Data['PE_by_area'])
        os.remove('PMT{}/'.format(pmt)+file)
np.savez('PMT{}/spectra.npz'.format(pmt), spectrum=spectrum[1:], mean_WF=mean_WF, Recon_wf=Recon_wf, Chi2=Chi2, ID=ID, dif=dif, PE_by_area=PE_by_area)
