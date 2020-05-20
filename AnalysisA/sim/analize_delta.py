import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

data=np.load('p.npz')
t0=data['t0']
t_val=data['t_val']
t_maxi=data['t_maxi']
t_h=data['t_h']
St=data['St']


def lin(x, a,b):
    return a*x+b

def exp(x, a,b,c):
    return a*np.exp(-x/b)+c

def r(x, a,b,c):
    return a/(b+x)+c

p0=[10, 0]
p0, cov=curve_fit(lin, St, t_maxi, p0=p0)

p1=[30, 0.1, 5]
p1, cov=curve_fit(exp, St, t_h, p0=p1)

p2=[30, 0.1, 5]
p2, cov=curve_fit(r, St, t_h, p0=p2)

plt.plot(St, t0, '.', label='t0')
plt.plot(St, t_val, '+', label='t_val')
plt.plot(St, t_maxi, 'g^', label='t_maxi')
plt.plot(St, t_h, 'r*', label='t_h')

plt.plot(St, lin(St, *p0), 'g--', label='{}, {}'.format(p0[0], p0[1]))
plt.plot(St, r(St, *p2), 'r--', label='{}, {}, {}'.format(p2[0], p2[1], p2[2]))

plt.legend()
plt.show()
