import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import poisson, binom
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.special import erf as erf
import sys

p01=0.1
s=0.8
n=300
P=np.zeros((n,n))
P[0,1]=p01
P[1,1]=0.5*(1+erf(0.5/(np.sqrt(2)*s)))-P[0,1]

for i in np.arange(2, n):
    P[i,1]=0.5*(erf((i-0.5)/(np.sqrt(2)*s))-erf((i-1.5)/(np.sqrt(2)*s)))

for i in range(n):
    for j in range(2,n):
        P[i,j]=np.sum(P[:i+1,1]*np.flip(P[:i+1,j-1]))

P[1,0]=1-np.sum(P[1,1:])
P[2:,0]=P[1,0]**(np.arange(2,n))
P[0,0]=1-np.sum(P[1:,0])



pmt=0
path='/home/gerak/Desktop/DireXeno/190803/Co57/PMT{}/'.format(pmt)
data=np.load(path+'recon_wf.npz')
rec=data['rec']
chi2_cut=1.7e6
rng=np.nonzero(np.logical_and(rec['chi2']<chi2_cut, rec['init']>=70))
spectrum=np.sum(rec[rng]['h'], axis=1)

NQ=50
ns=np.arange(20,100)
h=poisson.pmf(ns, NQ)
H=np.matmul(P[:,ns], h.reshape(len(ns),1))[ns]
a=np.amax(spectrum)/np.amax(H)
print(a)
plt.figure()
plt.hist(spectrum, bins=np.arange(300)-0.5)
plt.plot(ns, h/np.amax(h)*np.amax(spectrum), 'r+')
plt.plot(ns, a*H, 'g.-')

plt.show()
