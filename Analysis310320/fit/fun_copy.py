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

def rec_to_p(rec):
    p=np.array([])
    for name in rec.dtype.names:
        p=np.append(p, np.array(rec[name][0]))
    return p

def p_to_rec(p, pmts):
    rec=np.recarray(1, dtype=[
        ('NQ', 'f8', len(pmts)),
        ('Spe', 'f8', len(pmts)),
        ('N_events', 'f8', len(pmts)),
        ('a_pad', 'f8', len(pmts)),
        ('a_spe', 'f8', len(pmts)),
        ('a_dpe', 'f8', len(pmts)),
        ('a_trpe', 'f8', len(pmts)),
        ('s_pad', 'f8', len(pmts)),
        ('m_pad', 'f8', len(pmts)),
        ('q', 'f8', len(pmts)),
        ('a0', 'f8', len(pmts)),
        ('St', 'f8', len(pmts)),
        ('F', 'f8', 1),
        ('Tf', 'f8', 1),
        ('Ts', 'f8', 1),
    ])
    for i, name in enumerate(rec.dtype.names):
        if np.shape(rec[name][0])==(len(pmts),):
            rec[name][0]=p[i*len(pmts):(i+1)*len(pmts)]
        elif name=='F':
            rec[name][0]=p[-3]
        elif name=='Tf':
            rec[name][0]=p[-2]
        elif name=='Ts':
            rec[name][0]=p[-1]
    return rec




def make_P(Spe, s_pad, r0, Q):
    n=100
    P=np.zeros((n,n))
    F=np.zeros((n,n))

    P[0,1:]=0.5*(1+erf((r0-np.arange(1,n))/(np.sqrt(2*(np.arange(1,n)*Spe**2+s_pad**2)))))
    P[1,1:]=0.5*(erf((1.5-np.arange(1,n))/(np.sqrt(2*(np.arange(1,n)*Spe**2+s_pad**2))))-erf((r0-np.arange(1,n))/(np.sqrt(2*(np.arange(1,n)*Spe**2+s_pad**2)))))
    for i in range(2,n):
        P[i,1:]=0.5*(erf((i+0.5-np.arange(1,n))/(np.sqrt(2*(np.arange(1,n)*Spe**2+s_pad**2))))-erf((i-0.5-np.arange(1,n))/(np.sqrt(2*(np.arange(1,n)*Spe**2+s_pad**2)))))

    F[1:,0]=binom.pmf(np.arange(1,n), np.arange(1,n), Q)
    for j in range(1,n):
        for i in range(1,n):
            F[i,j]=np.sum(binom.pmf(np.arange(i+1), i, Q)*np.flip(P[:i+1,j]))
    F[0,0]=1-np.sum(F[1:,0])
    F[0,1:]=F[0,0]*P[0,1:]
    for j in range(n):
        F[:,j]=F[:,j]/np.sum(F[:,j])
    return F


def model_spec(ns, NQ, P):
    h=poisson.pmf(np.arange(len(P[0,:])), NQ)
    return np.ravel(np.matmul(P, h.reshape(len(h),1))[ns])

def model_area(areas, Mpad, Spad, Spe, a_pad, a_spe, a_dpe, a_trpe):
    da=areas[1]-areas[0]
    h=np.zeros(len(areas))
    for i, a in enumerate(areas):
        r=np.linspace(a-0.5*da, a+0.5*da, 100)
        dr=r[1]-r[0]
        h[i]=dr*np.sum(a_pad*np.exp(-0.5*(r-Mpad)**2/(Spad)**2)+a_spe*np.exp(-0.5*(r-(1+Mpad))**2/(Spe**2+Spad**2))+a_dpe*np.exp(-0.5*(r-(2+Mpad))**2/(2*Spe**2+Spad**2))+a_trpe*np.exp(-0.5*(r-(3+Mpad))**2/(3*Spe**2+Spad**2)))
    return h




def Const(tau,T,s):
    C=1/(1-erf(s/(np.sqrt(2)*tau)+T/(np.sqrt(2)*s))+np.exp(-s**2/(2*tau**2)-T/tau)*(1+erf(T/(np.sqrt(2)*s))))
    return C

def Int(t, tau, T, s):
    dt=t[1]-t[0]
    return np.exp(-t/tau)*(1-erf(s/(np.sqrt(2)*tau)-(t-T)/(np.sqrt(2)*s)))*dt*Const(tau,T,s)/tau

def make_z(h):
    print('In make z')
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
    print('Out of make z')
    return z/C


def make_P0(h):
    p0=np.zeros(1000)
    for i in range(1000):
        p0[i]=np.prod(h[0,:i])
    return p0

def make_T0(Ms):
    p0=np.append(1, np.prod(np.cumprod(Ms, axis=0), axis=1))
    t0=p0[:-1]*(np.shape(Ms)[1]-np.sum(Ms, axis=1))
    return t0/np.sum(t0)

def Model2(NQs, F, Tf, Ts, Sts, Ps):
    T=50
    n=15
    t=np.arange(2000)/5
    I=np.arange(2*n*len(t))
    Ms=np.zeros((2*n, len(t), len(Ps)))
    for i in range(len(Ps)):
        m=(NQs[i]*(F*Int(t,Tf,T,Sts[i])+(1-F)*Int(t,Ts,T,Sts[i])))
        Ms[:,:,i]=(poisson.pmf(np.floor(I/len(t)), m[I%len(t)]).reshape((2*n, len(t))))
    T0=make_T0(Ms[0,:1000,:])
    temporal=np.zeros((n, len(T0), len(Ps)))
    for i, P in enumerate(Ps):
        h=np.zeros((np.shape(Ms)[0], len(T0)))
        for j in range(len(T0)):
            h[:,j]=np.sum(T0*Ms[:,j:j+len(T0), i], axis=1)
        temporal[:,:,i]=np.matmul(Ps[i][:n,:np.shape(h)[0]], h)
    return temporal


def Model3(NQs,  F, Tf, Ts, Sts, Ps):
    R=0.0
    T=50
    n=15
    t=np.arange(2000)/5
    dt=t[1]-t[0]
    I=np.arange(2*n*len(t))
    Ms=np.zeros((2*n, len(t), len(Ps)))
    for i in range(len(Ps)):
        m=NQs[i]*((1-R)*(F*Int(t,Tf,T,Sts[i])+(1-F)*Int(t,Ts,T,Sts[i]))+R*dt*np.exp(-0.5*(t-T)**2/Sts[i]**2)/np.sqrt(2*np.pi*Sts[i]**2))
        Ms[:,:,i]=(poisson.pmf(np.floor(I/len(t)), m[I%len(t)]).reshape((2*n, len(t))))
    T0=make_T0(Ms[0,:1000,:])
    temporal=np.zeros((n, len(T0), len(Ps)))
    for i, P in enumerate(Ps):
        h=np.zeros((np.shape(Ms)[0], len(T0)))
        for j in range(len(T0)):
            h[:,j]=np.sum(T0*Ms[:,j:j+len(T0), i], axis=1)
        temporal[:,:,i]=np.matmul(Ps[i][:n,:np.shape(h)[0]], h)
        plt.figure()
        plt.plot(np.mean(temporal[:,:,i].T*np.arange(np.shape(temporal)[0]), axis=1), 'k.')
        plt.show()



def model_h(X, a,t,s):
    y=np.zeros(len(X))
    dx=X[1]-X[0]
    for i, x in enumerate(X):
        r=np.linspace(x-0.5*dx,x+0.5*dx,100)
        dr=r[1]-r[0]
        y[i]=np.sum(a*np.exp(-0.5*(np.log(r/t))**2/s**2))*dr
    return y
