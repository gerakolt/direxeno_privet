from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf
from scipy.special import comb

def make_data(N_events, NQ, R, F, Tf, Ts, Spe, s_pad, St, T, Strig):
    d=np.zeros((N_events, 1000))
    for i in range(N_events):
        n=np.random.poisson(NQ)
        ch=np.random.choice(3, size=n, replace=True, p=[R, (1-R)*F, (1-R)*(1-F)])
        nd=len(np.nonzero(ch==0)[0])
        nf=len(np.nonzero(ch==1)[0])
        ns=len(np.nonzero(ch==2)[0])
        trig=np.random.normal(T*5, Strig*5, 1)
        tf=np.random.normal(trig+np.random.exponential(Tf*5, nf), St*5, nf)
        ts=np.random.normal(trig+np.random.exponential(Ts*5, ns), St*5, ns)
        td=np.random.normal(trig, St*5, nd)
        t=np.append(td, np.append(ts, tf))
        h, bins=np.histogram(t-np.amin(t), bins=np.arange(1001)-0.5)
        d[i]=h
    return d


t0=[]
t_val=[]
t_maxi=[]
t_h=[]
St=[]
for i, st in enumerate(np.linspace(0.1,2,100)):
    print(i)
    D=make_data(10000, 36, 1, 0, 5, 45, 0.3, 0.3, st, 30, 5)
    temp=np.mean(D, axis=0)
    t0.append(temp[0])
    t_val.append(temp[1])
    t_maxi.append(1+np.argmax(temp[1:]))
    t_h.append(np.amax(temp[1:]))
    St.append(st)

np.savez('p', t0=t0, t_val=t_val, t_maxi=t_maxi, t_h=t_h, St=St)

plt.plot(St, t0, '.', label='t0')
plt.plot(St, t_val, '+', label='t_val')
plt.plot(St, t_maxi, '^', label='t_maxi')
plt.plot(St, t_h, '*', label='t_h')
plt.legend()
plt.show()
