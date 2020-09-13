import multiprocessing
import numpy as np
import sys
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf, expi
from scipy.special import comb
from scipy.signal import convolve2d
import time
from admin import make_iter
from PMTgiom import whichPMT, make_dS
from memory_profiler import profile
from Sim import make_v, make_recomb

def Const(tau,T,s):
    C=1/(1-erf(s/(np.sqrt(2)*tau)+T/(np.sqrt(2)*s))+np.exp(-s**2/(2*tau**2)-T/tau)*(1+erf(T/(np.sqrt(2)*s))))
    return C

def Int(t, tau, T, s):
    return np.exp(-t/tau)*(1-erf(s/(np.sqrt(2)*tau)-(t-T)/(np.sqrt(2)*s)))


def Promt(t, F, Tf, Ts, T, St):
    dt=t[1]-t[0]
    I=np.arange(len(t)*len(T))
    If=Int(t[I//len(T)], Tf, T[I%len(T)], St[I%len(T)]).reshape(len(t), len(T))
    Is=Int(t[I//len(T)], Ts, T[I%len(T)], St[I%len(T)]).reshape(len(t), len(T))
    return  np.ravel(dt*(F*If*Const(Tf, T, St)/Tf+(1-F)*Is*Const(Ts, T, St)/Ts))

def Recomb(t, F, Tf, Ts, T, St, a, eta):
    dt=t[1]-t[0]
    y=make_recomb(t, a, eta)
    if np.sum(y)-1>1e-4:
        print('In recombenation model:')
        print('sum(y):', np.sum(y))
        sys.exit()
    elif np.sum(y)>1:
        y=y/np.sum(y)
    Y=np.zeros(len(t))
    Z=F*np.exp(-t/Tf)/Tf+(1-F)*np.exp(-t/Ts)/Ts
    for i in range(1,len(t)):
        Y[i]=np.sum(y[:i+1]*Z[i::-1])*dt
    I=np.arange(len(t)*len(T))
    B=(np.exp(-0.5*(t[I//len(T)]-T[I%len(T)])**2/St[I%len(T)]**2)/np.sqrt(2*np.pi*St[I%len(T)]**2)).reshape(len(t), len(T))
    C=convolve2d(B, Y.reshape(len(Y), 1), mode='full')*dt
    return np.ravel(C[:len(t),:])


def make_iter(V, dS, nu, n, model, Q):
    for i in range(len(dS)):
        np.random.seed(int(i*time.time()%2**32))
        yield [V[i], dS[i], nu, n, model, Q]

def make_B(iter):
    [V, dS, nu, n , model, Q]=iter
    t=np.arange(200)
    I=np.arange(len(nu)*len(Q)*len(t)*len(n))
    return binom.pmf(nu[I//(len(Q)*len(t)*len(n))], (n[I%len(n)]).astype(int),
            dS[(I//(len(n)*len(t)))%len(Q)]*Q[(I//(len(n)*len(t)))%len(Q)]*model[(I//len(n))%len(t),(I//(len(n)*len(t)))%len(Q)]).reshape(len(nu), len(Q), len(t), len(n))*V

def make_3D(x1, x2, gamma, Q, T, St, mu, W, F, Tf, Ts, R, a, eta, PEs, Xcov):
    v=make_v(10000, mu, x1, x2)
    V_mash, dS=make_dS(v)
    t=np.arange(200)
    dt=1
    r=np.arange(t[0], t[-1]+dt, dt/100)
    dr=r[1]-r[0]
    N=gamma*1000/W
    n=np.arange(np.floor(N-3*np.sqrt(N)), np.ceil(N+3*np.sqrt(N)))
    pois=poisson.pmf(n,N)
    nu=np.arange(20)
    model=np.sum(((1-R)*Promt(r, F, Tf, Ts, T, St)+R*(1-eta)*Recomb(r, F, Tf, Ts, T, St,a,eta)).reshape(len(t), 100, len(T)), axis=1)
    frac=np.sum(model[:int(np.mean(T)+100)], axis=0)
    s=np.zeros((len(PEs), len(Q), len(n)))

    p=multiprocessing.Pool(processes=2)
    Bs=p.map(make_B, make_iter(V_mash, dS, nu, n, model, Q))
    p.close()
    p.join()

    B=np.sum(np.array(Bs), axis=0)
    B[np.logical_and(B>1, B<1+1e-6)]=1
    I=np.arange(len(PEs)*len(Q)*len(n)*len(V_mash))
    s=binom.pmf(PEs[I//(len(Q)*len(n)*len(V_mash))], (n[(I//len(V_mash))%len(n)]).astype(int), dS[I%len(V_mash), (I//(len(n)*len(V_mash)))%len(Q)]*Q[(I//(len(n)*len(V_mash)))%len(Q)]
        *frac[(I//(len(n)*len(V_mash)))%len(Q)]).reshape(len(PEs),len(Q), len(n), len(V_mash))
    S=np.sum(np.sum(s*V_mash, axis=-1)*pois, axis=-1)

    M=np.sum(S.T*PEs, axis=1)
    Mcov=np.zeros((len(Xcov),15))
    k=0
    for i in range(5):
        for j in range(i+1,6):
            for p1, pe1 in enumerate(PEs):
                for p2, pe2 in enumerate(PEs):
                    c=(pe1-M[i])*(pe2-M[j])/(M[i]*M[j])
                    if np.abs(c)<=0.1:
                        Mcov[np.argmin(np.abs(c-Xcov)), k]+=np.sum(np.sum(s[p1,i]*s[p2,j]*V_mash, axis=-1)*pois, axis=-1)
            k+=1

    P0=np.vstack((np.ones(len(n)), np.cumprod(np.prod(B[0], axis=0), axis=0)[:-1]))
    P1=(P0*(1-np.prod(B[0], axis=0)))
    P=np.zeros((100,len(nu),len(Q)))
    for i in range(len(Q)):
        P2=P0*(1-np.prod(B[0, np.delete(np.arange(len(Q)), i)], axis=0))
        P[0,0,i]=np.sum(np.sum(B[0,i]*P2*pois, axis=1), axis=0)
    P[0,1:]=np.sum(np.sum(B[1:]*P0*pois, axis=3), axis=2)
    for i in range(1, 100):
        P[i]=np.sum(np.sum(B[:,:,i:,:]*P1[:len(t)-i,:]*pois, axis=3), axis=2)
    return np.transpose(P, (1,0,2)), S, Mcov
