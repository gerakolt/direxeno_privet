import numpy as np
import matplotlib.pyplot as plt

dt=29710 #Co57
dt_BG=48524

pmt=0
path='/home/gerak/Desktop/DireXeno/190803/Co57/PMT{}/'.format(pmt)
data=np.load(path+'H.npz')
H=data['H']
H_BG=data['H_BG']*dt/dt_BG
ns=data['ns']
spec=data['spec']
spec_BG=data['spec_BG']*dt/dt_BG

decay=np.mean(H.T*np.arange(len(H[:,0])), axis=1)
decay_BG=np.mean(H_BG.T*np.arange(len(H[:,0])), axis=1)

x=np.arange(1000)/5
plt.plot(x, decay, 'r.', label='data')
plt.plot(x, decay_BG, 'k.', label='data_BG')
plt.legend()
plt.show()
