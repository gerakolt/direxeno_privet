import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fun import import_spe

pmts=np.array([0,1,2,3,4,5,6,7,8,9,10,11,17,14,15,16,18,19])
spe=import_spe(pmts)
source='BG'
type=''
pmt=0
Chi2=[]
ID=np.array([])
path='/home/gerak/Desktop/DireXeno/'+source+'_190803'+type+'/spectra{}/'.format(pmt)
Spec=np.zeros(1000)
WF=np.zeros(1000)
recon_WF=np.zeros(1000)
for filename in os.listdir(path):
    if filename.startswith('spectrum'):
        Data = np.load(path+filename)
        Spec=np.vstack((Spec, Data['spectrum']))
        WF+=Data['mean_WF']
        recon_WF+=Data['Recon_wf']
        Chi2.extend(Data['Chi2'])
        ID=np.append(ID,Data['ID'])

Spec=Spec[1:,:]
np.savez(path+'Spec', Spec=Spec, WF=WF, recon_WF=recon_WF, Chi2=Chi2, ID=ID, spe=spe[np.nonzero(pmts==pmt)[0][0]])
