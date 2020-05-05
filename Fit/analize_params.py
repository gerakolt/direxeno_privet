import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf
from scipy.special import comb

pmt=0
path='/home/gerak/Desktop/DireXeno/WFsim/PMT{}/'.format(pmt)
Data=np.load(path+'NQs.npz')
prm=Data['prm']
q=Data['q']

L=np.exp(-q)/np.amax(np.exp(-q))
plt.figure()
plt.plot(prm, L, 'k.')
plt.show()
