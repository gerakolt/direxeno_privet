import numpy as np
import time
import matplotlib.pyplot as plt


pmt=19
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
BL=np.load(path+'BL.npz')['BL']
x=np.arange(1000)

l=120
ly=-0.02
r=300
ry=0.4

bl=np.array(BL)
bl[l]=ly
bl[r]=ry

a=(ly-ry)/(l-r)
b=ry-a*r
bl[l:r]=a*x[l:r]+b

plt.plot(x, BL, 'k.')
plt.plot(x, bl, 'r.')
np.savez(path+'BL'.format(pmt), BL=bl)

plt.show()
