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

a=np.array([1,10,100])

b=np.array([[1,2,3,4], [-1,-2,-3,-4], [0.5, 1.5, 2.5,3.5]])
print(a*b.T)
