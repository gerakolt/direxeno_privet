from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf
from scipy.special import comb

tau=45
nq=50
St=1
T=50
t=np.arange(2000)/5
Spe=0.5
S_trig=1


def make_data(N, nq, tau, T, St, Spe, S_trig):
    d=np.zeros((N,1000))
    for i in range(N):
        n=np.random.poisson(nq)
        t=np.random.normal(np.random.normal(T*5, S_trig*5, 1)+np.random.exponential(tau*5, n), St*5, n)
        h,bins=np.histogram(t-np.amin(t), bins=np.arange(1001)-0.5)
        d[i]=h
        # d[i]=np.roll(h, -np.amin(np.nonzero(h>0)[0]))
    return d


def Const(tau,T,s):
    C=1/(1-erf(s/(np.sqrt(2)*tau)+T/(np.sqrt(2)*s))+np.exp(-s**2/(2*tau**2)-T/tau)*(1+erf(T/(np.sqrt(2)*s))))
    return C

def Int(t, tau, T, s):
    dt=t[1]-t[0]
    return np.exp(-t/tau)*(1-erf(s/(np.sqrt(2)*tau)-(t-T)/(np.sqrt(2)*s)))*dt*Const(tau,T,s)/tau

def make_P(Spe, ns):
    P=np.zeros((ns[-1]+10, ns[-1]+10))
    P[0,0]=1
    for i in range(len(P[:,0])):
        r=np.linspace(i-0.5,i+0.5,1000)
        dr=r[1]-r[0]
        P[i,1]=dr*np.sum(np.exp(-0.5*(r-1)**2/Spe**2))/(np.sqrt(2*np.pi)*Spe)
    for j in range(2, len(P[0,:])):
        for i in range(len(P[:,0])):
            P[i,j]=np.sum(P[:i+1,1]*np.flip(P[:i+1,j-1]))
    return P




def make_z(h):
    z=np.zeros((np.shape(h)[0], 1000))
    p0=make_P0(h)
    C=0
    for k in range(np.shape(z)[1]):
        C+=p0[k]*(1-h[0,k])
    for i in range(1, np.shape(z)[0]):
        for k in range(np.shape(z)[1]):
            z[i,0]+=p0[k]*h[i,k]
    for i in range(1, np.shape(z)[0]):
        for j in range(1, np.shape(z)[1]):
            for k in range(np.shape(z)[1]):
                z[i,j]+=p0[k]*(1-h[0,k])*h[i,j+k]
    return z/C

def make_P0(h):
    p0=np.zeros(2000)
    for i in range(2000):
        p0[i]=np.prod(h[0,:i])
    return p0



m=Int(t,tau,T,St)
H=np.zeros((5,len(t)))
ns=np.arange(np.shape(H)[0])
I=np.arange(H.size)
M=poisson.pmf(np.floor(I/len(t)), nq*m[I%len(t)])
H=M.reshape(H.shape)
P=make_P(Spe, ns)
h=np.matmul(P[:,:np.shape(H)[0]], H)
z=make_z(h)
d1=make_data(15000, nq, 45, T, St, Spe, S_trig)
d2=make_data(15000, nq, 45, 0, St, Spe, S_trig)


plt.figure()
# plt.plot(t[:1000], m[:1000], 'k--', label='m')
# plt.plot(t[:1000], np.sum(H.T*ns, axis=1)[:1000], 'r--', label='<H>')
# plt.plot(t[:1000], np.sum(h.T*np.arange(np.shape(h)[0]), axis=1)[:1000], 'g--', label='<h> ({})'.format(np.sum(np.sum(h.T*np.arange(np.shape(h)[0]), axis=1)[:1000])))
plt.plot(t[:1000], np.sum(z.T*np.arange(np.shape(z)[0]), axis=1), 'r.', label='<z> ({})'.format(np.sum(np.sum(z.T*np.arange(np.shape(z)[0]), axis=1))))
plt.plot(t[:1000], np.mean(d1, axis=0), '.', label='data 45')
plt.plot(t[:1000], np.mean(d2, axis=0), '.', label='data 5')


plt.legend()
# plt.yscale('log')

plt.show()
