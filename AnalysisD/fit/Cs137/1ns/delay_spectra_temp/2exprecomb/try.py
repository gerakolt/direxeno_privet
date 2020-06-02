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



def f(x,b):
    t=27
    return 1/t*np.exp(-x/t)-b/(t*(b+x))+b/t**2*np.exp(-(x+b)/t)*(expi((x+b)/t)-expi(b/t))

x=np.arange(1,200)
y=np.exp(-x/34)
y=y/np.sum(y)

p0=[1]
p, cov=curve_fit(f, x[25:],y[25:], p0=p0)

print(p)
plt.plot(x,y,'k.')
plt.plot(x,f(x, *p),'r--')
plt.show()
