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

Tf=1
b=2

t=np.arange(25)+0.5
y=1/Tf*np.exp(-t/Tf)-b/(Tf*(b+t))+b/Tf**2*np.exp(-(t+b)/Tf)*(expi((t+b)/Tf)-expi(b/Tf))
z=[]
k=[]
for i in range(len(t)):
    r=np.linspace(t[i]-0.5, t[i]+0.5,100)
    dr=r[1]-r[0]
    z.append(np.sum(1/Tf*np.exp(-r/Tf)-b/(Tf*(b+r))+b/Tf**2*np.exp(-(r+b)/Tf)*(expi((r+b)/Tf)-expi(b/Tf)))*dr)

    r=np.linspace(t[i]-0.5, t[i]+0.5,1000)
    dr=r[1]-r[0]
    k.append(np.sum(1/Tf*np.exp(-r/Tf)-b/(Tf*(b+r))+b/Tf**2*np.exp(-(r+b)/Tf)*(expi((r+b)/Tf)-expi(b/Tf)))*dr)

plt.plot(t,y,'ko')
plt.plot(t,z,'ro')
plt.plot(t,k,'go')

plt.show()
