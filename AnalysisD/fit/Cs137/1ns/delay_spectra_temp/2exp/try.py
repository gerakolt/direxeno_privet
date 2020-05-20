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


ind=np.random.choice(3, size=3, replace=False, p=[0.3,0.3,0.4])
print(ind)
