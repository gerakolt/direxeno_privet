import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def do_smd(X, Y, w):
    if not int(1000/w)==1000/w:
        print('Not good window')
        return np.zeros(1000), np.zeros(1000)
    n=int(1000/w)
    x=np.zeros(n)
    y=np.zeros(n)
    for i in range(n):
        x[i]=np.mean(X[w*i:w*(i+1)])
        y[i]=np.mean(Y[w*i:w*(i+1)])
    return x,y



source='Co57'
type=''
pmt=0
wind=1
path='/home/gerak/Desktop/DireXeno/190803/'+source+type+'/PMT{}/'.format(pmt)
data=np.load(path+'spectra.npz')
spectrum=np.mean(data['spectrum'], axis=0)
Recon_wf=data['Recon_wf']/len(data['ID'])
mean_wf=data['mean_WF']/len(data['ID'])
Chi2=data['Chi2']
ID=data['ID']

x=np.arange(1000)/5
x2,spectrum2=do_smd(x, spectrum, wind*5)


fig=plt.figure()
fig.suptitle('PMT{}-'.format(pmt)+source+type)

ax=fig.add_subplot(311)
ax.plot(x, spectrum, label='{}')
ax.plot(x2, spectrum2, '.-')
ax.legend()

ax=fig.add_subplot(312)
ax.plot(x, Recon_wf, 'r.-', label='Recon WF - Chi2={}'.format(np.sqrt(np.mean(Recon_wf-mean_wf)**2)))
ax.plot(x, mean_wf, 'k.-', label='Mean WF')
ax.legend()

ax=fig.add_subplot(313)
ax.hist(Chi2, bins=100, range=[0,4e7])
ax.set_yscale('log')
print(ID[Chi2>0.5e7][:5])

plt.show()
