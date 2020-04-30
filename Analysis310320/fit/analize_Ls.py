import numpy as np
import matplotlib.pyplot as plt

Ls=np.load('Ls.npz')
L1=Ls['L1']
L2=Ls['L2']
L3=Ls['L3']
L4=Ls['L4']
L5=Ls['L5']

Ls=np.load('Ls2.npz')
A1=Ls['L1']
A2=Ls['L2']
A3=Ls['L3']
A4=Ls['L4']
A5=Ls['L5']


plt.plot(-L1, '.', label='L1')
plt.plot(-L2, '.', label='L2')
plt.plot(-L3, '.', label='L3')
plt.plot(-L4, '.', label='L4')
plt.plot(-L5, '.', label='L5')

plt.plot(-A1, '+', label='L1')
plt.plot(-A2, '+', label='L2')
plt.plot(-A3, '+', label='L3')
plt.plot(-A4, '+', label='L4')
plt.plot(-A5, '+', label='L5')

plt.legend()
# plt.yscale('log')
plt.show()
