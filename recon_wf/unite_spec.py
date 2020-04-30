import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path='/home/gerak/Desktop/DireXeno/190803/Co57/'

for file in os.listdir(path+'Spectra/'):
    if file.startswith('spectra'):
        Data=np.load(path+'Spectra/'+file)
        pmts=Data['pmts']
        break

spectrum=np.zeros((len(pmts),1000))
mean_WF=np.zeros((len(pmts),1000))
Recon_wf=np.zeros((len(pmts),1000))
Chi2=[]
ID=[]
first_pmt=np.zeros(len(pmts))
BLW=np.zeros(len(pmts))

for file in os.listdir(path+'Spectra/'):
    if file.startswith('spectra'):
        Data=np.load(path+'Spectra/'+file)
        spectrum=np.dstack((spectrum, Data['spectrum']))
        Recon_wf+=Data['Recon_wf']
        mean_WF+=Data['mean_WF']
        Chi2=np.append(Chi2, Data['Chi2'])
        ID=np.append(ID, Data['ID'])
        first_pmt+=Data['first_pmt']
        try:
            BLW=np.vstack((BLW, Data['BLW'].T))
        except:
            BLW=np.vstack((BLW, Data['BLW']))

        print('remove', file)
        os.remove(path+'Spectra/'+file)
np.savez(path+'Spectra/spectra.npz', spectrum=spectrum[:,:,1:], mean_WF=mean_WF, Recon_wf=Recon_wf, Chi2=Chi2, ID=ID,
            first_pmt=first_pmt, BLW=BLW[1:], pmts=pmts)
