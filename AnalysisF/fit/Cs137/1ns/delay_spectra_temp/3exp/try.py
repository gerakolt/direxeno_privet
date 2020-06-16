from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf
from scipy.special import comb
from PMTgiom import make_pmts


a=np.array([[1,2,3], [4,5,6]])
b=np.array([[11,12,13], [14,15,16]])
c=np.dstack((a, b))
print(c)
print(np.array(c*np.array([1,-1])).reshape(2,3,2))
