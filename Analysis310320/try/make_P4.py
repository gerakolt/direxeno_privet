import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import poisson, binom
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.special import erf as erf
import sys
import random
from datetime import datetime
random.seed(datetime.now())


def make_P(Spe, p00, p01):
    n=100
    P=np.zeros((n,n))
    R=np.zeros((n,n))
    P[0,0]=p00
    P[1:,0]=((1-p00)/(2-p00))**(np.arange(1, n))
    P[0,1:]=((1-p00)/(2-p00))**(np.arange(1, n))

    R[0,1]=p01
    R[1,1]=0.5*(1+erf(0.5/(np.sqrt(2)*Spe)))-p01
    for i in np.arange(2, n):
        R[i,1]=0.5*(erf((i-0.5)/(np.sqrt(2)*Spe))-erf((i-1.5)/(np.sqrt(2)*Spe)))

    
    return P


P=make_P(0.5,0.9,0.1)
plt.plot(np.sum(P, axis=0), '.', label='axis 0')
plt.plot(np.sum(P, axis=1), '.', label='axis 1')
plt.legend()
plt.yscale('log')
plt.show()
