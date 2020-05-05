import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf
import sys
from scipy.optimize import minimize

def get_data(pmts, source):
    Hs=[]
    Ns=[]
    N_events=[]
    H_specs=[]
    path='/home/gerak/Desktop/DireXeno/190803/{}/'.format(source)
    for pmt in pmts:
        data=np.load(path+'PMT{}/H.npz'.format(pmt))
        Hs.append(data['H'])
        N_events.append(data['N_events'])
    return Hs, N_events

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
    z=np.zeros((np.shape(h)[0], shp[1]))
    p0=make_P0(h)
    C=0
    for k in range(np.shape(z)[1]):
        C+=p0[k]*(1-h[0,k])
    for i in range(1, np.shape(z)[0]):
        for k in range(np.shape(z)[1]):
            z[i,0]+=p0[k]*h[i,k]
    for i in range(np.shape(z)[0]):
        for j in range(1, np.shape(z)[1]):
            for k in range(np.shape(z)[1]):
                z[i,j]+=p0[k]*(1-h[0,k])*h[i,j+k]
    return z/C

def make_P0(h):
    p0=np.zeros(shp[1]*2)
    for i in range(shp[1]*2):
        p0[i]=np.prod(h[0,:i])
    return p0

def model1(p):
    T=50
    [NQ, F, tf, ts, St, Spe]=p
    t=np.arange(shp[1]*2)/5
    P=make_P(Spe, np.arange(shp[0]))
    I=np.arange(np.shape(P)[1]*len(t))
    m=NQ*(F*Int(t,tf,T,St)+(1-F)*Int(t,ts,T,St))
    M=poisson.pmf(np.floor(I/len(t)), m[I%len(t)])
    h=np.matmul(P,M.reshape((np.shape(P)[1], len(t))))
    return make_z(h)

def model2(p):
    [NQ, R, F, tf, ts, St, Spe]=p
    T=50
    t=np.arange(shp[1]*2)/5
    dt=t[1]-t[0]
    P=make_P(Spe, np.arange(shp[0]))
    I=np.arange(np.shape(P)[1]*len(t))
    m=NQ*(R*dt*(np.exp(-0.5*(t-T)**2/St**2)/(np.sqrt(2*np.pi)*St))+(1-R)*(F*Int(t,tf,T,St)+(1-F)*Int(t,ts,T,St)))
    M=poisson.pmf(np.floor(I/len(t)), m[I%len(t)])
    h=np.matmul(P,M.reshape((np.shape(P)[1], len(t))))
    return make_z(h)
