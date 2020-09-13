from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf, expi
from scipy.special import comb
from PMTgiom import make_pmts
from make_mash import mash
from scipy.signal import convolve2d

pmts=[0,1,4,7,8,14]


pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn, r=make_pmts(pmts)
r_mash, V_mash=mash()

def Sim(t, N, F, Tf, Ts, R, a, eta, Q, T, St):
    f=np.sum(make_recomb(np.arange(1000*20)/100, a, eta).reshape(1000,20), axis=1)
    f[-1]=1-np.sum(f[:-1])
    N_events=2000
    Strig=2
    d=np.zeros((N_events, 200, len(Q)))
    H=np.zeros((30, 200, len(Q)))
    G=np.zeros((250,200))
    trp=np.zeros((N_events, 200, len(Q)))
    sng=np.zeros((N_events, 200, len(Q)))
    Rtrp=np.zeros((N_events, 200, len(Q)))
    Rsng=np.zeros((N_events, 200, len(Q)))
    Gtrp=np.zeros((250,200))
    Gsng=np.zeros((250,200))
    GRtrp=np.zeros((250,200))
    GRsng=np.zeros((250,200))
    for i in range(N_events):
        print('in sim', i)
        t0=np.zeros(len(Q))
        trig=np.random.normal(0, Strig, 1)
        N_glob=np.random.poisson(N)
        ex=np.random.binomial(N_glob, 1-R)
        recomb=np.random.binomial(N_glob-ex, 1-eta)
        t=np.zeros(recomb+ex)
        t[:recomb]+=np.random.choice(np.arange(1000)/5, size=recomb,  replace=True, p=f)
        ch=np.random.choice(2, size=recomb+ex, replace=True, p=[F, 1-F])
        t[ch==0]+=np.random.exponential(Tf, len(t[ch==0]))
        t[ch==1]+=np.random.exponential(Ts, len(t[ch==1]))
        slow_i=np.nonzero(ch==1)[0]
        fast_i=np.nonzero(ch==0)[0]
        for j in range(len(Q)):
            ind=np.nonzero(1==np.random.choice(2, size=len(t), replace=True, p=[1-Q[j]*dS[j], Q[j]*dS[j]]))[0]
            tj=np.random.normal(trig+T[j]+t[ind], St[j], len(ind))
            # tj=t[ind]
            h, bins=np.histogram(tj, bins=np.arange(201))
            if np.any(h>0):
                t0[j]=np.amin(np.nonzero(h>0)[0])
            d[i,:,j]=h
            trp[i,:,j]=np.histogram(tj[np.nonzero(np.logical_and(ind>=recomb, np.isin(ind, slow_i)))[0]], bins=np.arange(201))[0]
            sng[i,:,j]=np.histogram(tj[np.nonzero(np.logical_and(ind>=recomb, np.isin(ind, fast_i)))[0]], bins=np.arange(201))[0]
            Rtrp[i,:,j]=np.histogram(tj[np.nonzero(np.logical_and(ind<recomb, np.isin(ind, slow_i)))[0]], bins=np.arange(201))[0]
            Rsng[i,:,j]=np.histogram(tj[np.nonzero(np.logical_and(ind<recomb, np.isin(ind, fast_i)))[0]], bins=np.arange(201))[0]
        for j in range(len(Q)):
            d[i,:,j]=np.roll(d[i,:,j], -int(np.amin(t0)))
            trp[i,:,j]=np.roll(trp[i,:,j], -int(np.amin(t0)))
            sng[i,:,j]=np.roll(sng[i,:,j], -int(np.amin(t0)))
            Rtrp[i,:,j]=np.roll(Rtrp[i,:,j], -int(np.amin(t0)))
            Rsng[i,:,j]=np.roll(Rsng[i,:,j], -int(np.amin(t0)))

    spectrum=np.histogram(np.sum(np.sum(d, axis=2), axis=1), bins=np.arange(1000)-0.5)[0]
    for k in range(200):
        G[:,k]=np.histogram(np.sum(d[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]

        Gtrp[:,k]=np.histogram(np.sum(trp[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        Gsng[:,k]=np.histogram(np.sum(sng[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        GRtrp[:,k]=np.histogram(np.sum(Rtrp[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        GRsng[:,k]=np.histogram(np.sum(Rsng[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]

        for j in range(len(Q)):
            H[:,k,j]=np.histogram(d[:,k,j], bins=np.arange(np.shape(H)[0]+1)-0.5)[0]
    return H/N_events, G/N_events, spectrum, Gtrp/N_events, Gsng/N_events, GRtrp/N_events, GRsng/N_events




def Sim2(N, F, Tf, Ts, R, a, eta, Q, T, St, PEs):
    f=np.sum(make_recomb(np.arange(1000*20)/100, a, eta).reshape(1000,20), axis=1)
    f[-1]=1-np.sum(f[:-1])
    N_events=10000
    Strig=2
    d=np.zeros((N_events, 200, len(Q)))
    H=np.zeros((30, 200, len(Q)))
    G=np.zeros((250,200))
    trp=np.zeros((N_events, 200, len(Q)))
    sng=np.zeros((N_events, 200, len(Q)))
    Rtrp=np.zeros((N_events, 200, len(Q)))
    Rsng=np.zeros((N_events, 200, len(Q)))
    Gtrp=np.zeros((250,200))
    Gsng=np.zeros((250,200))
    GRtrp=np.zeros((250,200))
    GRsng=np.zeros((250,200))
    spectra=np.zeros((len(PEs),len(Q)))
    for i in range(N_events):
        print('in sim', i)
        t0=np.zeros(len(Q))
        trig=np.random.normal(0, Strig, 1)
        N_glob=np.random.poisson(N)
        ex=np.random.binomial(N_glob, 1-R)
        recomb=np.random.binomial(N_glob-ex, 1-eta)
        t=np.zeros(recomb+ex)
        t[:recomb]+=np.random.choice(np.arange(1000)/5, size=recomb,  replace=True, p=f)
        ch=np.random.choice(2, size=recomb+ex, replace=True, p=[F, 1-F])
        t[ch==0]+=np.random.exponential(Tf, len(t[ch==0]))
        t[ch==1]+=np.random.exponential(Ts, len(t[ch==1]))
        slow_i=np.nonzero(ch==1)[0]
        fast_i=np.nonzero(ch==0)[0]
        costheta=np.random.uniform(-1,1)
        phi=np.random.uniform(0,2*np.pi)
        r3=np.random.uniform(0,(10/40)**3)
        r=r3**(1/3)
        v=np.array([r*np.sin(np.arccos(costheta))*np.cos(phi), r*np.sin(np.arccos(costheta))*np.sin(phi), r*costheta])
        costheta=np.random.uniform(-1,1, len(t))
        phi=np.random.uniform(0,2*np.pi, len(t))
        us=np.vstack((np.vstack((np.sin(np.arccos(costheta))*np.cos(phi), np.sin(np.arccos(costheta))*np.sin(phi))), costheta))
        pmt_hit=whichPMT(v, us, pmt_mid, pmt_r, pmt_up)
        for j in range(len(Q)):
            hits=np.nonzero(pmt_hit==j)[0]
            ind=np.nonzero(1==np.random.choice(2, size=len(hits), replace=True, p=[1-Q[j], Q[j]]))[0]
            tj=np.random.normal(trig+T[j]+t[hits[ind]], St[j], len(ind))
            h, bins=np.histogram(tj, bins=np.arange(201))
            if np.any(h>0):
                t0[j]=np.amin(np.nonzero(h>0)[0])
            d[i,:,j]=h
            trp[i,:,j]=np.histogram(tj[np.nonzero(np.logical_and(hits[ind]>=recomb, np.isin(hits[ind], slow_i)))[0]], bins=np.arange(201))[0]
            sng[i,:,j]=np.histogram(tj[np.nonzero(np.logical_and(hits[ind]>=recomb, np.isin(hits[ind], fast_i)))[0]], bins=np.arange(201))[0]
            Rtrp[i,:,j]=np.histogram(tj[np.nonzero(np.logical_and(hits[ind]<recomb, np.isin(hits[ind], slow_i)))[0]], bins=np.arange(201))[0]
            Rsng[i,:,j]=np.histogram(tj[np.nonzero(np.logical_and(hits[ind]<recomb, np.isin(hits[ind], fast_i)))[0]], bins=np.arange(201))[0]
        for j in range(len(Q)):
            d[i,:,j]=np.roll(d[i,:,j], -int(np.amin(t0)))
            trp[i,:,j]=np.roll(trp[i,:,j], -int(np.amin(t0)))
            sng[i,:,j]=np.roll(sng[i,:,j], -int(np.amin(t0)))
            Rtrp[i,:,j]=np.roll(Rtrp[i,:,j], -int(np.amin(t0)))
            Rsng[i,:,j]=np.roll(Rsng[i,:,j], -int(np.amin(t0)))

    spectrum=np.histogram(np.sum(np.sum(d, axis=2), axis=1), bins=np.arange(1000)-0.5)[0]
    for i in range(len(Q)):
        spectra[:,i]=np.histogram(np.sum(d[:,:100,i], axis=1), bins=np.arange(len(PEs)+1)-0.5)[0]
    for k in range(200):
        G[:,k]=np.histogram(np.sum(d[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]

        Gtrp[:,k]=np.histogram(np.sum(trp[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        Gsng[:,k]=np.histogram(np.sum(sng[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        GRtrp[:,k]=np.histogram(np.sum(Rtrp[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        GRsng[:,k]=np.histogram(np.sum(Rsng[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]

        for j in range(len(Q)):
            H[:,k,j]=np.histogram(d[:,k,j], bins=np.arange(np.shape(H)[0]+1)-0.5)[0]
    return H/N_events, G/N_events, spectrum, spectra, Gtrp/N_events, Gsng/N_events, GRtrp/N_events, GRsng/N_events




def whichPMT(v, us, mid, rt, up):
    hits=np.zeros(len(us[0]))-1
    for i in range(len(mid)):
        a=(1-np.sum(mid[i]*v, axis=0))/np.sum(us.T*mid[i], axis=1)
        r=v+(a*us).T-mid[i]
        hits[np.nonzero(np.logical_and(a>0, np.logical_and(np.abs(np.sum(r*rt[i], axis=1))<np.sum(rt[i]**2), np.abs(np.sum(r*up[i], axis=1))<np.sum(up[i]**2))))[0]]=i
    return hits




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


# def make_dS(d, pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn):
#     if np.sum(d**2)>=1:
#         sys.exit()
#         return np.zeros(len(pmt_mid))
#     dS=np.zeros(len(pmt_mid))
#     ETA=np.linspace(-1,1,100, endpoint=True)
#     zeta=np.linspace(-1,1,100, endpoint=True)
#     deta=ETA[1]-ETA[0]
#     for i in range(len(dS)):
#         for eta in ETA:
#             dS[i]+=np.sum(pmt_r[i]*pmt_r[i])*deta**2*np.sum((1-np.sum(d*pmt_mid[i]))/((1-2*np.sum(d*pmt_mid[i])+np.sum(d*d)-2*eta*np.sum(d*pmt_r[i])-2*zeta*np.sum(d*pmt_up[i])
#                             +eta**2*np.sum(pmt_r[i]*pmt_r[i])+zeta**2*np.sum(pmt_up[i]*pmt_up[i]))**(3/2)))
#     return dS/(4*np.pi)

def make_dS(d, m, rt, up):
    r=np.sqrt(np.sum(rt[0]**2))
    dS=np.zeros(len(m))
    a=np.linspace(-1,1,1000, endpoint=True)
    I=np.arange(len(a)**2)
    for i in range(len(dS)):
        x=m[i,0]+a[I//len(a)]*rt[i,0]+a[I%len(a)]*up[i,0]-d[0]
        y=m[i,1]+a[I//len(a)]*rt[i,1]+a[I%len(a)]*up[i,1]-d[1]
        z=m[i,2]+a[I//len(a)]*rt[i,2]+a[I%len(a)]*up[i,2]-d[2]
        dS[i]=np.sum((1-np.sum(d*m[i]))/(np.sqrt(x**2+y**2+z**2)**3))*((a[1]-a[0])*r)**2
    return dS/(4*np.pi)


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

def make_recomb(t, a, eta):
    dt=t[1]-t[0]
    ni=np.ones(len(t))
    ne=np.ones(len(t))*(1-eta)
    for i in range(1, len(t)):
        ni[i]=ni[i-1]*(1-dt*a*ne[i-1])
        ne[i]=ne[i-1]*(1-dt*a*ni[i-1])
    return dt*a*(ni*ne)/(1-eta)

def Recomb(t, F, Tf, Ts, T, St, a, eta):
    dt=t[1]-t[0]
    y=make_recomb(t, a, eta)
    Y=np.zeros(len(t))
    Z=F*np.exp(-t/Tf)/Tf+(1-F)*np.exp(-t/Ts)/Ts
    for i in range(1,len(t)):
        # Y[i]=np.sum(y[:i]*Z[i-1::-1])*dt
        Y[i]=np.sum(y[:i+1]*Z[i::-1])*dt
    I=np.arange(len(t)*len(T))
    B=(np.exp(-0.5*(t[I//len(T)]-T[I%len(T)])**2/St[I%len(T)]**2)/np.sqrt(2*np.pi*St[I%len(T)]**2)).reshape(len(t), len(T))
    C=convolve2d(B, Y.reshape(len(Y), 1), mode='full')*dt
    return np.ravel(C[:len(t),:])


# def make_3D(t, N, F, Tf, Ts, R, a, eta, Q, T, St):
#     dS=make_dS(np.array([0, 0, 0]),  pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn)
#     dt=t[1]-t[0]
#     r=np.arange(t[0], t[-1]+dt, dt/100)
#     dr=r[1]-r[0]
#     n=np.arange(np.floor(N-4*np.sqrt(N)), np.ceil(N+4*np.sqrt(N)))
#     pois=poisson.pmf(n,N)
#     nu=np.arange(30)
#     model=np.sum(((1-R)*Promt(r, F, Tf, Ts, T, St)+R*(1-eta)*Recomb(r, F, Tf, Ts, T, St,a,eta)).reshape(len(t), 100, len(T)), axis=1)
#     if np.any(model<0):
#         return np.amin(model)*np.ones((len(nu), 100, len(Q)))
#     I=np.arange(len(nu)*len(Q)*len(t)*len(n))
#     B=binom.pmf(nu[I//(len(Q)*len(t)*len(n))], (n[I%len(n)]).astype(int), dS[(I//(len(n)*len(t)))%len(Q)]*Q[(I//(len(n)*len(t)))%len(Q)]*model[(I//len(n))%len(t),(I//(len(n)*len(t)))%len(Q)]).reshape(len(nu), len(Q), len(t), len(n))
#     # P=np.matmul(B, pois)
#     if np.any(np.isnan(B)):
#         print('B is nan')
#         sys.exit()
#     if np.any(np.isinf(B)):
#         print('B is inf')
#         sys.exit()
#     P0=np.vstack((np.ones(len(n)), np.cumprod(np.prod(B[0], axis=0), axis=0)[:-1]))
#     P1=(P0*(1-np.prod(B[0], axis=0)))
#     P=np.zeros((100,len(nu),len(Q)))
#     for i in range(len(Q)):
#         P2=P0*(1-np.prod(B[0, np.delete(np.arange(len(Q)), i)], axis=0))
#         P[0,0,i]=np.sum(np.sum(B[0,i]*P2*pois, axis=1), axis=0)
#     P[0,1:]=np.sum(np.sum(B[1:]*P0*pois, axis=3), axis=2)
#     for i in range(1, 100):
#         P[i]=np.sum(np.sum(B[:,:,i:,:]*P1[:len(t)-i,:]*pois, axis=3), axis=2)
#     return np.transpose(P, (1,0,2))


def make_3D(t, N, F, Tf, Ts, R, a, eta, Q, T, St):
    dt=t[1]-t[0]
    r=np.arange(t[0], t[-1]+dt, dt/100)
    dr=r[1]-r[0]
    n=np.arange(np.floor(N-4*np.sqrt(N)), np.ceil(N+4*np.sqrt(N)))
    pois=poisson.pmf(n,N)
    nu=np.arange(30)
    model=np.sum(((1-R)*Promt(r, F, Tf, Ts, T, St)+R*(1-eta)*Recomb(r, F, Tf, Ts, T, St,a,eta)).reshape(len(t), 100, len(T)), axis=1)
    if np.any(model<0):
        return np.amin(model)*np.ones((len(nu), 100, len(Q)))
    I=np.arange(len(nu)*len(Q)*len(t)*len(n))
    B=np.zeros((len(nu), len(Q), len(t), len(n)))
    # for i in range(len(r_mash)):
    for i in range(1):
        V_mash[0]=1
        # print(i)
        dS=make_dS(r_mash[i],  pmt_mid, pmt_r, pmt_up)
        B+=binom.pmf(nu[I//(len(Q)*len(t)*len(n))], (n[I%len(n)]).astype(int), dS[(I//(len(n)*len(t)))%len(Q)]*Q[(I//(len(n)*len(t)))%len(Q)]*model[(I//len(n))%len(t),(I//(len(n)*len(t)))%len(Q)]).reshape(len(nu),
         len(Q), len(t), len(n))*V_mash[i]
    if np.any(np.isnan(B)):
        print('B is nan')
        sys.exit()
    if np.any(np.isinf(B)):
        print('B is inf')
        sys.exit()
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


def make_spectra(m, PEs):
    spectra=np.zeros((len(PEs), np.shape(m)[-1]))
    P=np.zeros((len(PEs), np.shape(m)[1], np.shape(m)[-1]))
    P[:np.shape(m)[0]]=m
    spectra=np.array(P[:,0,:])
    for i in range(1, np.shape(m)[1]):
        temp=np.array(spectra)
        for pe in PEs:
            temp[pe]=np.sum(spectra[:pe+1]*P[pe::-1,i], axis=0)
        spectra=np.array(temp)
    return spectra
