import numpy as np
import time
import matplotlib.pyplot as plt


pmt=18
path='/home/gerak/Desktop/DireXeno/130520/pulser/PMT{}/'.format(pmt)
BL=np.load(path+'BL.npz')['BL']
x=np.arange(1000)

l=125
ly=0.26
r=224
ry=0.635

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
