import numpy as np
import time
import matplotlib.pyplot as plt


pmt=7
path='/home/gerak/Desktop/DireXeno/190803/NEWpulser/NEWPMT{}/'.format(pmt)
BL=np.load(path+'BL.npz')['BL']
x=np.arange(1000)

l=140
ly=2.5
r=740
ry=9.5

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
