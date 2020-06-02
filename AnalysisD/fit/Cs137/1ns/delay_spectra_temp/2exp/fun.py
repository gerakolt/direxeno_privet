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
    H=np.zeros((50, 200, len(Q)))
    G=np.zeros((250,200))
    for i in range(N_events):
        t0=np.zeros(len(Q))
        trig=np.random.normal(0, 5*Strig, 1)
        N_glob=np.random.poisson(N)
        ch=np.random.choice(2, size=N_glob, replace=True, p=[F, 1-F])
        nf=len(np.nonzero(ch==0)[0])
        ns=len(np.nonzero(ch==1)[0])
        tf=np.random.exponential(5*Tf, nf)
        ts=np.random.exponential(5*Ts, ns)
        t=np.append(tf, ts)
        for j in range(len(Q)):
            n=np.random.binomial(N_glob, Q[j])
            ind=np.nonzero(1==np.random.choice(2, size=len(t), replace=True, p=[1-Q[j], Q[j]]))[0]
            tj=np.random.normal(trig+5*T[j]+t[ind], 5*St[j], len(ind))
            h, bins=np.histogram(tj, bins=np.arange(201)*5)
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

PMTs=[0,1,4,7,8,14]
mid, rt, pmt_l, up, pmt_dn, r=make_pmts(PMTs)
def Sim2(N, Q, T, Strig, R, F, Tf, Ts, St):
    N_events=10
    d=np.zeros((N_events, 200, len(Q)))
    H=np.zeros((50, 200, len(Q)))
    G=np.zeros((250,200))
    for i in range(N_events):
        print(i)
        t0=np.zeros(len(Q))
        trig=np.random.normal(0, 5*Strig, 1)
        N_glob=np.random.poisson(N)
        ch=np.random.choice(3, size=N_glob, replace=True, p=[R, (1-R)*F, (1-R)*(1-F)])
        nd=len(np.nonzero(ch==0)[0])
        nf=len(np.nonzero(ch==1)[0])
        ns=len(np.nonzero(ch==2)[0])
        tf=np.random.exponential(5*Tf, nf)
        ts=np.random.exponential(5*Ts, ns)
        t=np.append(tf, ts)
        costheta=np.random.uniform(-1,1,N_glob)
        phi=np.random.uniform(0,2*np.pi,N_glob)
        pmts=whichPMT(costheta, phi, mid, rt, up, r)
        for j in range(len(Q)):
            ind0=np.nonzero(pmts==j)
            ind=np.nonzero(1==np.random.choice(2, size=len(t[ind0]), replace=True, p=[1-Q[j], Q[j]]))[0]
            tj=np.random.normal(trig+5*T[j]+t[ind0][ind], 5*St[j], len(ind))
            h, bins=np.histogram(tj, bins=np.arange(201)*5)
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



def whichPMT(costheta, phi, mid, rt, up, r):
    d=np.array([0,0,0])
    x=np.sin(np.arccos(costheta))*np.cos(phi)
    y=np.sin(np.arccos(costheta))*np.sin(phi)
    z=costheta
    n=np.zeros(len(z))-1
    for j in range(len(z)):
        for i in range(len(mid)):
            b=np.sum(mid[i]*(mid[i]-d))
            R=np.array([x[j], y[j], z[j]])
            a=b/np.sum(R*mid[i])
            if a>0:
                v=mid[i]-d-a*R
                if np.abs(np.sum(v*up[i]))<(0.5*r)**2 and np.abs(np.sum(v*rt[i]))<(0.5*r)**2:
                    n[j]=i
                    break
    return n




def Const(tau,T,s):
    C=1/(1-erf(s/(np.sqrt(2)*tau)+T/(np.sqrt(2)*s))+np.exp(-s**2/(2*tau**2)-T/tau)*(1+erf(T/(np.sqrt(2)*s))))
    return C


def Int(t, dt, tau, T, s):
    return dt*np.exp(-t/tau)*(1-erf(s/(np.sqrt(2)*tau)-(t-T)/(np.sqrt(2)*s)))*Const(tau,T,s)/tau


def delta(t ,T, St):
    y=np.zeros(len(t))
    dt=t[1]-t[0]
    dr=dt/100
    for i in range(len(t)):
        r=np.linspace(t[i]-0.5*dt, t[i]+0.5*dt, 100)
        y[i]=np.sum(np.exp(-0.5*(r-T)**2/St**2)/(np.sqrt(2*np.pi)*St))*dr
    return y


def make_dS(d, pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn):
    dS=np.zeros(len(pmt_mid))
    ETA=np.linspace(-1,1,100, endpoint=True)
    zeta=np.linspace(-1,1,100, endpoint=True)
    deta=ETA[1]-ETA[0]
    for i in range(len(dS)):
        for eta in ETA:
            dS[i]+=np.sum(pmt_r[i]*pmt_r[i])*deta**2*np.sum((1-np.sum(d*pmt_mid[i]))/((1-2*np.sum(d*pmt_mid[i])+np.sum(d*d)-2*eta*np.sum(d*pmt_r[i])-2*zeta*np.sum(d*pmt_up[i])
                            +eta**2*np.sum(pmt_r[i]*pmt_r[i])+zeta**2*np.sum(pmt_up[i]*pmt_up[i]))**(3/2)))
    return dS/(4*np.pi)


pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn, r=make_pmts(PMTs)
dS=make_dS(np.array([0,0,0]),  pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn)



# def make_3D(t, dt, N, R, F, Tf, Ts, Q, T, St):
#     n=np.arange(np.floor(N-4*np.sqrt(N)), np.ceil(N+4*np.sqrt(N)))
#     pois=poisson.pmf(n,N)
#     nu=np.arange(50)
#     I=np.arange(len(t)*len(Q))
#     model=(R[I%len(Q)]*np.exp(-0.5*(t[I//len(Q)]-T[I%len(Q)])**2/St[I%len(Q)]**2)/np.sqrt(2*np.pi*St[I%len(Q)]**2)+
#                                             (1-R[I%len(Q)])*(F*Int(t[I//len(Q)], dt, Tf, T[I%len(Q)], St[I%len(Q)])+
#                                             (1-F)*Int(t[I//len(Q)], dt, Ts, T[I%len(Q)], St[I%len(Q)]))).reshape(len(t), len(Q))
#     I=np.arange(len(nu)*len(Q)*len(t)*len(n))
#     B=binom.pmf(nu[I//(len(Q)*len(t)*len(n))], n[I%len(n)],
#         dS[(I//(len(n)*len(t)))%len(Q)]*Q[(I//(len(n)*len(t)))%len(Q)]*dt*model[(I//len(n))%len(t),
#          (I//(len(n)*len(t)))%len(Q)]).reshape(len(nu), len(Q), len(t), len(n))
#     P0=np.vstack((np.ones(len(n)), np.cumprod(np.prod(B[0], axis=0), axis=0)[:-1]))
#     P1=(P0*(1-np.prod(B[0], axis=0)))
#     P=np.zeros((len(t),len(nu),len(Q)))
#     for i in range(len(Q)):
#         P2=P0*(1-np.prod(B[0, np.delete(np.arange(len(Q)), i)], axis=0))
#         P[0,0,i]=np.sum(np.sum(B[0,i]*P2*pois, axis=1), axis=0)
#     P[0,1:]=np.sum(np.sum(B[1:]*P0*pois, axis=3), axis=2)
#     for i in range(1, len(t)):
#         P[i]=np.sum(np.sum(B[:,:,i:,:]*P1[:len(t)-i,:]*pois, axis=3), axis=2)
#     return np.transpose(P, (1,0,2))[:,:100,:]

def make_3D(t, dt, N, R, F, Tf, Ts, Q, T, St):
    n=np.arange(np.floor(N-4*np.sqrt(N)), np.ceil(N+4*np.sqrt(N)))
    pois=poisson.pmf(n,N)
    nu=np.arange(50)
    I=np.arange(len(t)*len(Q))
    model=(R[I%len(Q)]*np.exp(-0.5*(t[I//len(Q)]-T[I%len(Q)])**2/St[I%len(Q)]**2)/np.sqrt(2*np.pi*St[I%len(Q)]**2)+
                                            (1-R[I%len(Q)])*(F*Int(t[I//len(Q)], dt, Tf, T[I%len(Q)], St[I%len(Q)])+
                                            (1-F)*Int(t[I//len(Q)], dt, Ts, T[I%len(Q)], St[I%len(Q)]))).reshape(len(t), len(Q))
    I=np.arange(len(nu)*len(Q)*len(t)*len(n))
    B=binom.pmf(nu[I//(len(Q)*len(t)*len(n))], n[I%len(n)],
        dS[(I//(len(n)*len(t)))%len(Q)]*Q[(I//(len(n)*len(t)))%len(Q)]*dt*model[(I//len(n))%len(t),
         (I//(len(n)*len(t)))%len(Q)]).reshape(len(nu), len(Q), len(t), len(n))
    P0=np.vstack((np.ones(len(n)), np.cumprod(np.prod(B[0], axis=0), axis=0)[:-1]))
    P1=(P0*(1-np.prod(B[0], axis=0)))
    P=np.zeros((100,len(nu),len(Q)))
    for i in range(len(Q)):
        P2=P0*(1-np.prod(B[0, np.delete(np.arange(len(Q)), i)], axis=0))
        P[0,0,i]=np.sum(np.sum(B[0,i]*P2*pois, axis=1), axis=0)
    P[0,1:]=np.sum(np.sum(B[1:]*P0*pois, axis=3), axis=2)
    for i in range(1, 100):
        P[i]=np.sum(np.sum(B[:,:,i:,:]*P1[:len(t)-i,:]*pois, axis=3), axis=2)
    return np.transpose(P, (1,0,2))


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
