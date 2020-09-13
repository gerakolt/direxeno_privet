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

data=np.load('Rec.npz')
Rec=data['Rec']
ls=data['ls']

names=Rec.dtype.names
for name in names:
    try:
        for i in range(len(Rec[name][0])):
            plt.figure()
            plt.title(name+'{}'.format(i))
            plt.plot(Rec[name][:len(ls),i], 'ko')
    except:
        plt.figure()
        plt.title(name)
        plt.plot(Rec[name][:len(ls)], 'ko')

plt.show()
