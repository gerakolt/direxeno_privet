import numpy as np
import sys
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf, expi
from scipy.special import comb
from PMTgiom import whichPMT, make_dS
from scipy.signal import convolve2d


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

def make_v(N, mu, x1, x2):
    k=0
    V=np.zeros(3)
    while len(V)<=N:
        k+=1
        costheta=np.random.uniform(-1,1)
        phi=np.random.uniform(0,2*np.pi)
        r3=np.random.uniform(0,(10/40)**3)
        r=r3**(1/3)
        v=np.array([r*np.sin(np.arccos(costheta))*np.cos(phi), r*np.sin(np.arccos(costheta))*np.sin(phi), r*costheta])
        d=v[x1]+np.sqrt((10/40)**2-v[x2]**2-v[-1]**2)
        P=np.exp(-d/mu)
        if 1==np.random.choice([0,1], size=1,  p=[1-P, P]):
            V=np.vstack((V,v))
    return V[1:]

def Sim_fit(N, F, Tf, Ts, R, a, eta, Q, T, St, PEs, mu, x1, x2, left, right):
    f=np.sum(make_recomb(np.arange(1000*20)/100, a, eta).reshape(1000,20), axis=1)
    i=2
    while 1-np.sum(f)>1e-4:
        f=np.sum(make_recomb(np.arange(i*1000*20)/100, a, eta).reshape(i*1000,20), axis=1)
        i+=1
    if np.sum(f)-1>1e-4:
        print('In recombenation model:')
        print('sum(f):', np.sum(f))
        sys.exit()
    else:
        f=f/np.sum(f)
    N_events=10000
    Strig=2
    d=np.zeros((N_events, 200, len(Q)))
    H=np.zeros((50, 200, len(Q)))
    spectra=np.zeros((len(PEs),len(Q)))
    cov=np.zeros((11, 15))
    v=make_v(N_events, mu, x1, x2)
    for i in range(N_events):
        t0=np.zeros(len(Q))
        trig=np.random.normal(0, Strig, 1)
        N_glob=np.random.poisson(N)
        ex=np.random.binomial(N_glob, 1-R)
        recomb=np.random.binomial(N_glob-ex, 1-eta)
        t=np.zeros(recomb+ex)
        t[:recomb]+=np.random.choice(np.arange(len(f))/5, size=recomb,  replace=True, p=f)
        ch=np.random.choice(2, size=recomb+ex, replace=True, p=[F, 1-F])
        t[ch==0]+=np.random.exponential(Tf, len(t[ch==0]))
        t[ch==1]+=np.random.exponential(Ts, len(t[ch==1]))
        slow_i=np.nonzero(ch==1)[0]
        fast_i=np.nonzero(ch==0)[0]

        costheta=np.random.uniform(-1,1, len(t))
        phi=np.random.uniform(0,2*np.pi, len(t))
        us=np.vstack((np.vstack((np.sin(np.arccos(costheta))*np.cos(phi), np.sin(np.arccos(costheta))*np.sin(phi))), costheta))
        pmt_hit=whichPMT(v[i], us)
        for j in range(len(Q)):
            hits=np.nonzero(pmt_hit==j)[0]
            ind=np.nonzero(1==np.random.choice(2, size=len(hits), replace=True, p=[1-Q[j], Q[j]]))[0]
            tj=np.random.normal(trig+T[j]+t[hits[ind]], St[j], len(ind))
            h, bins=np.histogram(tj, bins=np.arange(201))
            if np.any(h>0):
                t0[j]=np.amin(np.nonzero(h>0)[0])
            d[i,:,j]=h
        for j in range(len(Q)):
            d[i,:,j]=np.roll(d[i,:,j], -int(np.amin(t0)))

    s=np.sum(np.sum(d[:,:100],axis=1), axis=1)
    ind=np.nonzero(np.logical_and(s>left, s<right))[0]
    if len(ind)==0:
        w=np.abs(np.mean(s)-(left+right)/2)
        return H-w, spectra-w, cov-w
    d=d[ind]

    for i in range(len(Q)):
        spectra[:,i]=np.histogram(np.sum(d[:,:100,i], axis=1), bins=np.arange(len(PEs)+1)-0.5)[0]

    k=0
    for i in range(5):
        Si=np.sum(d[:,:100,i], axis=1)
        Mi=np.mean(Si)
        for j in range(i+1,6):
            Sj=np.sum(d[:,:100,j], axis=1)
            Mj=np.mean(Sj)
            cov[:,k]=np.histogram((Si-Mi)*(Sj-Mj)/(Mi*Mj), bins=11, range=[-0.1,0.1])[0]
            k+=1

    for k in range(200):
        for j in range(len(Q)):
            H[:,k,j]=np.histogram(d[:,k,j], bins=np.arange(np.shape(H)[0]+1)-0.5)[0]
    return H/len(ind), spectra/len(ind), cov/len(ind)



def Sim(N, F, Tf, Ts, R, a, eta, Q, T, St, PEs, mu, x1, x2, left, right):
    f=np.sum(make_recomb(np.arange(1000*20)/100, a, eta).reshape(1000,20), axis=1)
    i=2
    while 1-np.sum(f)>1e-4:
        f=np.sum(make_recomb(np.arange(i*1000*20)/100, a, eta).reshape(i*1000,20), axis=1)
        i+=1
    if np.sum(f)-1>1e-4:
        print('In recombenation model:')
        print('sum(f):', np.sum(f))
        sys.exit()
    else:
        f=f/np.sum(f)
    N_events=10000
    Strig=2
    d=np.zeros((N_events, 200, len(Q)))
    H=np.zeros((50, 200, len(Q)))
    spectra=np.zeros((len(PEs),len(Q)))
    cov=np.zeros((11, 15))

    G=np.zeros((250,200))
    trp=np.zeros((N_events, 200, len(Q)))
    sng=np.zeros((N_events, 200, len(Q)))
    Rtrp=np.zeros((N_events, 200, len(Q)))
    Rsng=np.zeros((N_events, 200, len(Q)))
    Gtrp=np.zeros((250,200))
    Gsng=np.zeros((250,200))
    GRtrp=np.zeros((250,200))
    GRsng=np.zeros((250,200))

    v=make_v(N_events, mu, x1, x2)
    for i in range(N_events):
        # print(i, 'out of', N_events)
        t0=np.zeros(len(Q))
        trig=np.random.normal(0, Strig, 1)
        N_glob=np.random.poisson(N)
        ex=np.random.binomial(N_glob, 1-R)
        recomb=np.random.binomial(N_glob-ex, 1-eta)
        t=np.zeros(recomb+ex)
        t[:recomb]+=np.random.choice(np.arange(len(f))/5, size=recomb,  replace=True, p=f)
        ch=np.random.choice(2, size=recomb+ex, replace=True, p=[F, 1-F])
        t[ch==0]+=np.random.exponential(Tf, len(t[ch==0]))
        t[ch==1]+=np.random.exponential(Ts, len(t[ch==1]))
        slow_i=np.nonzero(ch==1)[0]
        fast_i=np.nonzero(ch==0)[0]

        costheta=np.random.uniform(-1,1, len(t))
        phi=np.random.uniform(0,2*np.pi, len(t))
        us=np.vstack((np.vstack((np.sin(np.arccos(costheta))*np.cos(phi), np.sin(np.arccos(costheta))*np.sin(phi))), costheta))
        pmt_hit=whichPMT(v[i], us)
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

    s=np.sum(np.sum(d[:,:100],axis=1), axis=1)
    ind=np.nonzero(np.logical_and(s>left, s<right))[0]
    if len(ind)==0:
        w=np.abs(np.mean(s)-(left+right)/2)
        return H-w, spectra-w, cov-w, G-w, Gtrp-w, Gsng-w, GRtrp-w, GRsng-w
    d=d[ind]
    trp=trp[ind]
    sng=sng[ind]
    Rtrp=Rtrp[ind]
    Rsng=Rsng[ind]

    for i in range(len(Q)):
        spectra[:,i]=np.histogram(np.sum(d[:,:100,i], axis=1), bins=np.arange(len(PEs)+1)-0.5)[0]

    k=0
    for i in range(5):
        Si=np.sum(d[:,:100,i], axis=1)
        Mi=np.mean(Si)
        for j in range(i+1,6):
            Sj=np.sum(d[:,:100,j], axis=1)
            Mj=np.mean(Sj)
            cov[:,k]=np.histogram((Si-Mi)*(Sj-Mj)/(Mi*Mj), bins=11, range=[-0.1,0.1])[0]
            k+=1

    for k in range(200):
        G[:,k]=np.histogram(np.sum(d[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        Gtrp[:,k]=np.histogram(np.sum(trp[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        Gsng[:,k]=np.histogram(np.sum(sng[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        GRtrp[:,k]=np.histogram(np.sum(Rtrp[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        GRsng[:,k]=np.histogram(np.sum(Rsng[:,k,:], axis=1), bins=np.arange(np.shape(G)[0]+1)-0.5)[0]
        for j in range(len(Q)):
            H[:,k,j]=np.histogram(d[:,k,j], bins=np.arange(np.shape(H)[0]+1)-0.5)[0]
    return H/len(ind), spectra/len(ind), cov/len(ind), G/len(ind), Gtrp/len(ind), Gsng/len(ind), GRtrp/len(ind), GRsng/len(ind), len(ind)



def make_3D(N, F, Tf, Ts, R, a, eta, Q, T, St, PEs, mu, x1, x2, Xcov):
    v=make_v(10000, mu, x1, x2)
    V_mash, dS=make_dS(v)
    t=np.arange(200)
    dt=1
    r=np.arange(t[0], t[-1]+dt, dt/100)
    dr=r[1]-r[0]
    n=np.arange(np.floor(N-3*np.sqrt(N)), np.ceil(N+3*np.sqrt(N)))
    pois=poisson.pmf(n,N)
    nu=np.arange(20)
    model=np.sum(((1-R)*Promt(r, F, Tf, Ts, T, St)+R*(1-eta)*Recomb(r, F, Tf, Ts, T, St,a,eta)).reshape(len(t), 100, len(T)), axis=1)
    frac=np.sum(model[:int(np.mean(T)+100)], axis=0)
    if np.any(model<0):
        return np.amin(model)*np.ones((len(nu), 100, len(Q)))
    I=np.arange(len(nu)*len(Q)*len(t)*len(n))
    B=np.zeros((len(nu), len(Q), len(t), len(n)))
    s=np.zeros((len(PEs), len(Q), len(n)))
    for i in range(len(V_mash)):
        print('in B', i)
        B+=binom.pmf(nu[I//(len(Q)*len(t)*len(n))], (n[I%len(n)]).astype(int),
            dS[i, (I//(len(n)*len(t)))%len(Q)]*Q[(I//(len(n)*len(t)))%len(Q)]*model[(I//len(n))%len(t),(I//(len(n)*len(t)))%len(Q)]).reshape(len(nu), len(Q), len(t), len(n))*V_mash[i]
        if np.any(np.isnan(B)):
            print('B is nan')
            print(i, N, F, Tf, Ts, R, a, eta, Q, T, St, dS, PEs, V_mash, V_mash)
            sys.exit()
        if np.any(np.isinf(B)):
            print('B is inf')
            print(i, N, F, Tf, Ts, R, a, eta, Q, T, St, dS, PEs, V_mash, V_mash)
            sys.exit()
    B[np.logical_and(B>1, B<1+1e-6)]=1
    if np.any(1-B<0):
        print('B>1')
        print(B[1-B<0])
        print(N, F, Tf, Ts, R, a, eta, Q, T, St, dS, PEs, V_mash, V_mash)
        sys.exit()

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
