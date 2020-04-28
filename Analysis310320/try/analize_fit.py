import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from scipy.stats import poisson, binom
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.special import erf as erf
import sys
import random
from datetime import datetime
random.seed(datetime.now())


data=np.load('fit.npz')
r0=data['r0']
spe=np.linspace(0.1, 2, 500)

data=np.load('fit2.npz')
r2=data['r0']
spe2=np.linspace(1.165, 1.19, 100)

data=np.load('fit3.npz')
r3=data['r0']
spe3=data['spe']

data=np.load('fit4.npz')
r4=data['r0']
spe4=data['spe']

plt.plot(spe, r0, 'k.')
plt.plot(spe2, r2, 'k.')
plt.plot(spe3, r3, 'k.')
plt.plot(spe4, r4, 'k.')

plt.show()
