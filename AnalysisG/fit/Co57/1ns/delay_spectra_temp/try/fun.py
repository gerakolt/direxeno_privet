from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf
from scipy.special import comb

# def Sim(NQ, T, Strig, R, F, Tf, Ts, St):
#     N_events=10000
#     d=np.zeros((10000, 1000, len(NQ)))
#     H=np.zeros((25, 100, len(NQ)))
#     G=np.zeros((50,100))
#     for i in range(N_events):
#         t0=np.zeros(len(NQ))
#         trig=np.random.normal(0, Strig*5, 1)
#         for j in range(len(NQ)):
#             n=np.random.poisson(NQ[j])
#             ch=np.random.choice(3, size=n, replace=True, p=[R[j], (1-R[j])*F, (1-R[j])*(1-F)])
#             nd=len(np.nonzero(ch==0)[0])
#             nf=len(np.nonzero(ch==1)[0])
#             ns=len(np.nonzero(ch==2)[0])
#             td=np.random.normal(trig+5*T[j], St[j]*5, nd)
#             tf=np.random.normal(trig+5*T[j]+np.random.exponential(Tf*5, nf), St[j]*5, nf)
#             ts=np.random.normal(trig+5*T[j]+np.random.exponential(Ts*5, ns), St[j]*5, ns)
#             t=np.append(td, np.append(tf, ts))
#             h, bins=np.histogram(t, bins=np.arange(1001)-0.5)
#             t0[j]=np.amin(np.nonzero(h>0)[0])
#             d[i,:,j]=h
#         for j in range(len(NQ)):
#             d[i,:,j]=np.roll(d[i,:,j], -int(np.amin(t0)))
#     for k in range(100):
#         G[:,k]=np.histogram(np.sum(np.sum(d[:,10*k:10*(k+1),:], axis=1), axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
#         for j in range(len(NQ)):
#             H[:,k,j]=np.histogram(np.sum(d[:,10*k:10*(k+1),j], axis=1), bins=np.arange(np.shape(H)[0]+1)-0.5)[0]
#     return H/N_events, G/N_events


def Sim(NQ, T, Strig, R, F, Tf, Ts, St):
    N_events=10000
    d=np.zeros((10000, 200, len(NQ)))
    H=np.zeros((25, 200, len(NQ)))
    G=np.zeros((50,200))
    for i in range(N_events):
        t0=np.zeros(len(NQ))
        trig=np.random.normal(0, 5*Strig, 1)
        for j in range(len(NQ)):
            n=np.random.poisson(NQ[j])
            ch=np.random.choice(3, size=n, replace=True, p=[R[j], (1-R[j])*F, (1-R[j])*(1-F)])
            nd=len(np.nonzero(ch==0)[0])
            nf=len(np.nonzero(ch==1)[0])
            ns=len(np.nonzero(ch==2)[0])
            td=np.random.normal(trig+5*T[j], 5*St[j], nd)
            tf=np.random.normal(trig+5*T[j]+np.random.exponential(5*Tf, nf), 5*St[j], nf)
            ts=np.random.normal(trig+5*T[j]+np.random.exponential(5*Ts, ns), 5*St[j], ns)
            t=np.append(td, np.append(tf, ts))
            h, bins=np.histogram(t, bins=np.arange(201)*5)
            t0[j]=np.amin(np.nonzero(h>0)[0])
            d[i,:,j]=h
        for j in range(len(NQ)):
            d[i,:,j]=np.roll(d[i,:,j], -int(np.amin(t0)))
    for k in range(200):
        G[:,k]=np.histogram(np.sum(d[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        for j in range(len(NQ)):
            H[:,k,j]=np.histogram(d[:,k,j], bins=np.arange(np.shape(H)[0]+1)-0.5)[0]
    return H/N_events, G/N_events


def Global_Sim(NQ, T, Strig, R, F, Tf, Ts, St):
    N_events=10000
    d=np.zeros((10000, 1000, len(NQ)))
    G=np.zeros((15, 1000))
    for i in range(N_events):
        t0=np.zeros(len(NQ))
        trig=np.random.normal(0, Strig*5, 1)
        for j in range(len(NQ)):
            n=np.random.poisson(NQ[j])
            ch=np.random.choice(3, size=n, replace=True, p=[R[j], (1-R[j])*F, (1-R[j])*(1-F)])
            nd=len(np.nonzero(ch==0)[0])
            nf=len(np.nonzero(ch==1)[0])
            ns=len(np.nonzero(ch==2)[0])
            td=np.random.normal(trig+5*T[j], St[j]*5, nd)
            tf=np.random.normal(trig+5*T[j]+np.random.exponential(Tf*5, nf), St[j]*5, nf)
            ts=np.random.normal(trig+5*T[j]+np.random.exponential(Ts*5, ns), St[j]*5, ns)
            t=np.append(td, np.append(tf, ts))
            h, bins=np.histogram(t, bins=np.arange(1001)-0.5)
            t0[j]=np.amin(np.nonzero(h>0)[0])
            d[i,:,j]=h
        for j in range(len(NQ)):
            d[i,:,j]=np.roll(d[i,:,j], -int(np.amin(t0)))
    for k in range(1000):
        G[:,k]=np.histogram(np.sum(d[:,k,:], axis=1), bins=np.arange(16)-0.5)[0]
    return G/N_events


def subGlobal_Sim(NQ, T, Strig, R, F, f, Tf, Ts, St):
    N_events=10000
    d=np.zeros((10000, 1000, len(NQ)))
    G=np.zeros((15, 1000))
    for i in range(N_events):
        t0=np.zeros(len(NQ))
        trig=np.random.normal(0, Strig*5, 1)
        for j in range(len(NQ)):
            n=np.random.poisson(NQ[j])
            ch=np.random.choice(4, size=n, replace=True, p=[R[j], (1-R[j])*F, (1-R[j])*f[j], 1-R[j]-(1-R[j])*F-(1-R[j])*f[j]])
            nd=len(np.nonzero(ch==0)[0])
            nf=len(np.nonzero(ch==1)[0])
            ns=len(np.nonzero(ch==2)[0])
            td=np.random.normal(trig+5*T[j], St[j]*5, nd)
            tf=np.random.normal(trig+5*T[j]+np.random.exponential(Tf*5, nf), St[j]*5, nf)
            ts=np.random.normal(trig+5*T[j]+np.random.exponential(Ts*5, ns), St[j]*5, ns)
            t=np.append(td, np.append(tf, ts))
            h, bins=np.histogram(t, bins=np.arange(1001)-0.5)
            if len(np.nonzero(h>0)[0])>0:
                t0[j]=np.amin(np.nonzero(h>0)[0])
            else:
                t0[j]=1000
            d[i,:,j]=h
        for j in range(len(NQ)):
            d[i,:,j]=np.roll(d[i,:,j], -int(np.amin(t0)))
    for k in range(1000):
        G[:,k]=np.histogram(np.sum(d[:,k,:], axis=1), bins=np.arange(16)-0.5)[0]
    return G/N_events


def Const(tau,T,s):
    C=1/(1-erf(s/(np.sqrt(2)*tau)+T/(np.sqrt(2)*s))+np.exp(-s**2/(2*tau**2)-T/tau)*(1+erf(T/(np.sqrt(2)*s))))
    return C

def Int(t, tau, T, s):
    y=np.zeros(len(t))
    dt=t[1]-t[0]
    dr=dt/100
    for i in range(len(t)):
        r=np.linspace(t[i]-0.5*dt, t[i]+0.5*dt, 100)
        y[i]=np.sum(np.exp(-r/tau)*(1-erf(s/(np.sqrt(2)*tau)-(r-T)/(np.sqrt(2)*s))))*Const(tau,T,s)/tau*dr
    return y


def make_T0(Ms):
    p0=np.append(1, np.prod(np.cumprod(Ms, axis=0), axis=1))
    #t0=p0[:-1]*(np.shape(Ms)[1]-np.sum(Ms, axis=1))
    return p0

def delta(t ,T, St):
    y=np.zeros(len(t))
    dt=t[1]-t[0]
    dr=dt/100
    for i in range(len(t)):
        r=np.linspace(t[i]-0.5*dt, t[i]+0.5*dt, 100)
        y[i]=np.sum(np.exp(-0.5*(r-T)**2/St**2)/(np.sqrt(2*np.pi)*St))*dr
    return y



def Model(NQ, T, R, F, Tf, Ts, St):
    n=50
    t=np.arange(400)
    dt=t[1]-t[0]
    I=np.arange(2*n*len(t))
    Ms=np.zeros((2*n, len(t), len(NQ)))
    for i in range(len(NQ)):
        m=NQ[i]*((1-R[i])*(F*Int(t, Tf, T[i], St[i])+(1-F)*Int(t, Ts, T[i], St[i]))+R[i]*delta(t, T[i], St[i]))
        Ms[:,:,i]=(poisson.pmf(np.floor(I/len(t)), m[I%len(t)]).reshape((2*n, len(t))))
    P0=make_T0(Ms[0,:200,:])
    temporal=np.zeros((n, len(P0[:-1]), len(NQ)))
    # PG=make_P_Global6(Ms[:,:200,:])
    for i in range(len(NQ)):
        h=np.zeros((np.shape(Ms)[0], len(P0[:-1])))
        # h[0,0]=np.sum(P0[:-1]*((np.shape(Ms)[2]-1)*np.ones(len(P0[:-1]))-np.sum(Ms[0,:len(P0[:-1]),:], axis=1)+Ms[0,:len(P0[:-1]),i])*Ms[0,:len(P0[:-1]),i])
        M0=np.prod(Ms[0,:len(P0[:-1]), np.nonzero(np.arange(len(NQ))!=i)[0]], axis=0)
        h[0,0]=np.sum(P0[:-1]*(1-M0)*Ms[0,:len(P0[:-1]),i])
        h[1:,0]=np.sum(P0[:-1]*Ms[1:,:len(P0[:-1]),i], axis=1)
        M0=np.prod(Ms[0,:len(P0[:-1]),:], axis=1)
        for j in range(1,len(P0[:-1])):
            # h[:,j]=np.sum(T0*Ms[:,j:j+len(T0), i], axis=1)
            h[:,j]=np.sum(P0[:-1]*(1-M0)*Ms[:,j:j+len(P0[:-1]), i], axis=1)
        temporal[:,:,i]=h[:n]
    return temporal



def q0_model(ns, q0):
    y=np.zeros(len(ns))
    y[0]=1-q0/(1-q0)
    y[1:]=q0**np.arange(1, len(y))
    return y

def make_P(a0, Spad, Spe, m_pad):
    n=100
    P=np.zeros((n,n,len(a0)))
    for i in range(len(a0)):
        P[0,0,i]=1
        P[0,1:,i]=0.5*(1+erf((a0[i]-np.arange(1,n)-m_pad[i])/(np.sqrt(2*(Spad[i]**2+np.arange(1,n)*Spe[i]**2)))))
        P[1,1:,i]=0.5*(erf((1.5-np.arange(1,n)-m_pad[i])/(np.sqrt(2*(Spad[i]**2+np.arange(1,n)*Spe[i]**2))))-erf((a0[i]-np.arange(1,n)-m_pad[i])/(np.sqrt(2*(Spad[i]**2+np.arange(1,n)*Spe[i]**2)))))
        for j in range(2,n):
            P[j,1:,i]=0.5*(erf((j+0.5-np.arange(1,n)-m_pad[i])/(np.sqrt(2*(Spad[i]**2+np.arange(1,n)*Spe[i]**2))))-erf((j-0.5-np.arange(1,n)-m_pad[i])/(np.sqrt(2*(Spad[i]**2+np.arange(1,n)*Spe[i]**2)))))
    return P


def model_area(areas, m_pad, a_pad, a_spe, a_dpe, a_trpe, Spad, Spe):
    model=[]
    for i in range(len(areas)):
        da=areas[i][1]-areas[i][0]
        pad=a_pad[i]*np.exp(-0.5*(m_pad[i]-areas[i])**2/Spad[i]**2)/(np.sqrt(2*np.pi)*Spad[i])
        spe=a_spe[i]*np.exp(-0.5*((m_pad[i]+1)-areas[i])**2/(Spad[i]**2+Spe[i]**2))/(np.sqrt(2*np.pi*(Spad[i]**2+Spe[i]**2)))
        dpe=a_dpe[i]*np.exp(-0.5*((m_pad[i]+2)-areas[i])**2/(Spad[i]**2+2*Spe[i]**2))/(np.sqrt(2*np.pi*(Spad[i]**2+2*Spe[i]**2)))
        trpe=a_trpe[i]*np.exp(-0.5*((m_pad[i]+3)-areas[i])**2/(Spad[i]**2+3*Spe[i]**2))/(np.sqrt(2*np.pi*(Spad[i]**2+3*Spe[i]**2)))
        model.append(da*(pad+spe+dpe+trpe))
    return model


# def make_global(m):
#     G=np.zeros(np.shape(m)[:-1])
#     g=np.zeros(np.shape(m)[0,2])
#     for j in range(np.shape(G)[1]):
