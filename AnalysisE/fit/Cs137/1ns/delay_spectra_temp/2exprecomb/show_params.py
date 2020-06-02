from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf, expi
from scipy.special import comb
from PMTgiom import make_pmts

names=['F', 'Tf', 'Ts', 'R', 'b', 'l']
data=np.load('params.npz')['params']
data=data[1:]

fig, ax=plt.subplots(2,3)
for i in range(len(np.ravel(ax))):
    np.ravel(ax)[i].plot(data[:,i], 'ko', label=names[i])
    np.ravel(ax)[i].legend()
plt.show()
