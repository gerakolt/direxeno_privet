import numpy as np
from scipy.special import erfc, erf, expi
import matplotlib.pyplot as plt

t=np.arange(1000)/5
dt=t[1]-t[0]
tau=5
T=40
s=1
beta=45

A=(1/tau-beta/tau**2*expi(beta/tau)*np.exp(-beta/tau))*np.exp(-t/tau)+beta/tau**2*expi((beta+t)/tau)*np.exp(-(beta+t)/tau)-beta/(tau*(beta+t))
B=np.exp(-0.5*(t-T)**2/s**2)
C=np.convolve(A,B)[:1000]
plt.figure()
plt.plot(t, C, '-.')
plt.show()
