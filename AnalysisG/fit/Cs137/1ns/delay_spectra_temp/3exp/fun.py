from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf
from scipy.special import comb
from PMTgiom import make_pmts


def Sim(N, Q, T, Strig, R, F, Tf, Ts, St):
    N_events=10000
    d=np.zeros((N_events, 200, len(Q)))
    H=np.zeros((25, 200, len(Q)))
    G=np.zeros((50,200))
    for i in range(N_events):
        t0=np.zeros(len(Q))
        trig=np.random.normal(0, 5*Strig, 1)
        N_glob=np.random.poisson(N)
        for j in range(len(Q)):
            n=np.random.binomial(N_glob, Q[j])
            ch=np.random.choice(3, size=n, replace=True, p=[R[j], (1-R[j])*F, (1-R[j])*(1-F)])
            nd=len(np.nonzero(ch==0)[0])
            nf=len(np.nonzero(ch==1)[0])
            ns=len(np.nonzero(ch==2)[0])
            td=np.random.normal(trig+5*T[j], 5*St[j], nd)
            tf=np.random.normal(trig+5*T[j]+np.random.exponential(5*Tf, nf), 5*St[j], nf)
            ts=np.random.normal(trig+5*T[j]+np.random.exponential(5*Ts, ns), 5*St[j], ns)
            t=np.append(td, np.append(tf, ts))
            h, bins=np.histogram(t, bins=np.arange(201)*5)
            if np.any(h>0):
                t0[j]=np.amin(np.nonzero(h>0)[0])
            d[i,:,j]=h
        for j in range(len(Q)):
            d[i,:,j]=np.roll(d[i,:,j], -int(np.amin(t0)))
    spectrum=np.histogram(np.sum(np.sum(d, axis=2), axis=1), bins=np.arange(400)-0.5)[0]
    for k in range(200):
        G[:,k]=np.histogram(np.sum(d[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        for j in range(len(Q)):
            H[:,k,j]=np.histogram(d[:,k,j], bins=np.arange(np.shape(H)[0]+1)-0.5)[0]
    return H/N_events, G/N_events, spectrum


def whichPMT(costheta, phi, mid, rt, up, r, d):
    x=np.sin(np.arccos(costheta))*np.cos(phi)
    y=np.sin(np.arccos(costheta))*np.sin(phi)
    z=costheta
    n=np.zeros(len(mid))
    for j in range(len(z)):
        hit=0
        for i in range(len(mid)):
            # print(i,j)
            b=np.sum(mid[i]*(mid[i]-d))
            R=np.array([x[j], y[j], z[j]])
            a=b/np.sum(R*mid[i])
            if a>0:
                v=mid[i]-d-a*R
                if np.abs(np.sum(v*up[i]))<(0.5*r)**2 and np.abs(np.sum(v*rt[i]))<(0.5*r)**2:
                    n[i]+=1
                    hit+=1
            if hit>1:
                print('FFFFFFFFFFFFFFFFFFuck!!!!')
                sys.exit()
    return n

PMTs=[0,1,4,7,8,14]
def Sim2(N, Q, T, Strig, R, F, Tf, Ts, St):
    pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn, r=make_pmts(PMTs)
    N_events=10000
    d=np.zeros((N_events, 200, len(PMTs)))
    H=np.zeros((25, 200, len(PMTs)))
    G=np.zeros((50,200))
    NGlob=(np.amax(NQ)*4*np.pi/r**2)*len(PMTs)
    Q=NQ/np.amax(NQ)
    for i in range(N_events):
        print(i)
        t0=np.zeros(len(PMTs))
        trig=np.random.normal(0, 5*Strig, 1)
        N=np.random.poisson(NGlob)
        costheta=np.random.uniform(-1,1,N)
        phi=np.random.uniform(0,2*np.pi,N)
        # n=np.unique(whichPMT(costheta, phi, pmt_mid, pmt_r, pmt_up), return_counts=True)
        n=whichPMT(costheta, phi, pmt_mid, pmt_r, pmt_up, r, [0,0,0])
        for j, pmt in enumerate(PMTs):
            ch=np.random.choice(3, size=np.random.binomial(n[j], Q[j]), replace=True, p=[R[j], (1-R[j])*F, (1-R[j])*(1-F)])
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
        for j in range(len(PMTs)):
            d[i,:,j]=np.roll(d[i,:,j], -int(np.amin(t0)))
    spectrum=np.histogram(np.sum(np.sum(d, axis=2), axis=1), bins=np.arange(400)-0.5)[0]
    for k in range(200):
        G[:,k]=np.histogram(np.sum(d[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        for j in range(len(PMTs)):
            H[:,k,j]=np.histogram(d[:,k,j], bins=np.arange(np.shape(H)[0]+1)-0.5)[0]
    return H/N_events, G/N_events, spectrum



def Const(tau,T,s):
    C=1/(1-erf(s/(np.sqrt(2)*tau)+T/(np.sqrt(2)*s))+np.exp(-s**2/(2*tau**2)-T/tau)*(1+erf(T/(np.sqrt(2)*s))))
    return C

# def Int(t, tau, T, s):
#     y=np.zeros(len(t))
#     dt=t[1]-t[0]
#     dr=dt/100
#     for i in range(len(t)):
#         r=np.linspace(t[i]-0.5*dt, t[i]+0.5*dt, 100)
#         y[i]=np.sum(np.exp(-r/tau)*(1-erf(s/(np.sqrt(2)*tau)-(r-T)/(np.sqrt(2)*s))))*Const(tau,T,s)/tau*dr
#     return y

def Int(t, tau, T, s):
    return np.exp(-t/tau)*(1-erf(s/(np.sqrt(2)*tau)-(t-T)/(np.sqrt(2)*s)))*Const(tau,T,s)/tau


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


def make_3D(t, dt, N, R, F, Tf, Ts, Q, T, St):
    n=np.arange(np.floor(N-3*np.sqrt(N)), np.ceil(N+3*np.sqrt(N)))
    nu=np.arange(50)
    # h=np.tile(np.arange(n_max), (len(t),1)).T
    I=np.arange(len(t)*len(Q))
    model=(R[I%len(Q)]*np.exp(-0.5*(t[I//len(Q)]-T[I%len(Q)])**2/St[I%len(Q)]**2)/np.sqrt(2*np.pi*St[I%len(Q)]**2)+
                                            (1-R[I%len(Q)])*(F*Int(t[I//len(Q)], Tf, T[I%len(Q)], St[I%len(Q)])+
                                            (1-F)*Int(t[I//len(Q)], Ts, T[I%len(Q)], St[I%len(Q)]))).reshape(len(t), len(Q))
    I=np.arange(len(nu)*len(t)*len(n)*len(Q))
    B=binom.pmf(nu[I//(len(Q)*len(t)*len(n))], n[(I//len(Q))%len(n)], Q[I%len(Q)]*dt*model[(I//(len(Q)*len(n)))%len(t), I%len(Q)]).reshape(len(nu), len(t), len(n), len(Q))
    poiss=poisson.pmf(n, N).reshape(len(n),1)
    P0=np.matmul(np.prod(B[0], axis=2), poiss)[:,0]
    # P0 holds the probabilities that at time t all PMTs saw 0 PEs
    K=np.arange(len(t))
    P=np.zeros((len(t),len(Q),len(nu)))
    P[0]=np.matmul(np.matmul(np.transpose(B, (3,0,1,2)), poiss)[:,:,:,0], np.cumprod(np.append(1, P0[:-1])))
    for i in range(1, len(t)):
        P[i]=np.matmul(np.matmul(np.transpose(B, (3,0,1,2)), poiss)[:,:,i:,0], (np.cumprod(np.append(1, P0)[:len(t)-i])*(1-P0[:len(t)-i])))
    return np.transpose(P, (2,0,1))

# def make_3D(t, dt, N, R, F, Tf, Ts, Q, T, St):
#     model=N*dt*(R+(1-R)*(F*np.exp(-t/Tf)/Tf+(1-F)*np.exp(-t/Ts)/Ts))
#     n=np.arange(np.amax(model)+3*np.sqrt(np.amax(model)))
#     n_max=int(np.amin((15, n[-1])))
#     h=np.tile(np.arange(n_max), (len(t),1)).T
#     I=np.arange(n_max*len(t)*len(n)*len(t)*len(Q))
#     B=binom.pmf(h[I//(len(Q)*len(t)**2*len(n)), (I//(len(Q)*len(t)*len(n)))%len(t)],
#                 n[(I//(len(t)*len(Q)))%len(n)],
#                 Q[I%len(Q)]*dt[(I//(len(Q)*len(t)*len(n)))%len(t)]*
#                 np.exp(-0.5*(t[(I//(len(Q)*len(t)*len(n)))%len(t)]-t[(I//len(Q))%len(t)]-T[I%len(Q)])**2/St[I%len(Q)]**2)/
#                 np.sqrt(2*np.pi*St[I%len(Q)]**2)).reshape(n_max, len(t), len(n)*len(t), len(Q))
#     J=np.arange(len(n)*len(t))
#     poiss=poisson.pmf(n[(J//len(t))], model[J%len(t)]).reshape(len(n)*len(t),1)
#     # return np.matmul(np.transpose(B, (0,1,3,2)), poiss)[:,:,:,0]
#     P0=np.matmul(np.prod(B[0], axis=2), poiss)[:,0]
#     plt.figure()
#     plt.plot(P0, 'k.')
#     plt.show()
#     # P0 holds the probabilities that at time t all PMTs saw 0 PEs
#     K=np.arange(len(t))
#     P=np.zeros((len(t),len(Q),n_max))
#     P[0]=np.matmul(np.matmul(np.transpose(B, (3,0,1,2)), poiss)[:,:,:,0], np.cumprod(np.append(1, P0[:-1])))
#     for i in range(1, len(t)):
#         P[i]=np.matmul(np.matmul(np.transpose(B, (3,0,1,2)), poiss)[:,:,i:,0], (np.cumprod(np.append(1, P0)[:len(t)-i])*P0[:len(t)-i]))
#     return np.transpose(P, (2,0,1))


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
