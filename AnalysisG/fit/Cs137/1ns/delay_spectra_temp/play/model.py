from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf, expi
from scipy.special import comb
from PMTgiom import make_pmts
from make_mash import mash
from scipy.signal import convolve2d
from sim import Sim2

pmts=[0,1,4,7,8,14]
# pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn, r=make_pmts(pmts)
r_mash, V_mash, dS=mash(pmts)
V=np.load('V.npz')['V']
V_mash=V/np.sum(V)

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



def make_3D(t, N, F, Tf, Ts, R, a, eta, Q, T, St, dS, PEs):
    dt=t[1]-t[0]
    r=np.arange(t[0], t[-1]+dt, dt/100)
    dr=r[1]-r[0]
    n=np.arange(np.floor(N-4*np.sqrt(N)), np.ceil(N+4*np.sqrt(N)))
    pois=poisson.pmf(n,N)
    nu=np.arange(30)
    model=np.sum(((1-R)*Promt(r, F, Tf, Ts, T, St)+R*(1-eta)*Recomb(r, F, Tf, Ts, T, St,a,eta)).reshape(len(t), 100, len(T)), axis=1)
    frac=np.sum(model[:int(np.mean(T)+100)], axis=0)
    if np.any(model<0):
        return np.amin(model)*np.ones((len(nu), 100, len(Q)))
    I=np.arange(len(nu)*len(Q)*len(t)*len(n))
    B=np.zeros((len(nu), len(Q), len(t), len(n)))
    S=np.zeros((len(PEs), len(Q), len(n)))
    for i in range(len(r_mash)):
        print('in B', i)
    # for i in range(1):
        B+=binom.pmf(nu[I//(len(Q)*len(t)*len(n))], (n[I%len(n)]).astype(int), dS[i, (I//(len(n)*len(t)))%len(Q)]*Q[(I//(len(n)*len(t)))%len(Q)]*model[(I//len(n))%len(t),(I//(len(n)*len(t)))%len(Q)]).reshape(len(nu),
         len(Q), len(t), len(n))*V_mash[i]
    I=np.arange(len(PEs)*len(Q)*len(n))
    for i in range(len(r_mash)):
        print('In S', i)
    # for i in range(1):
        S+=binom.pmf(PEs[I//(len(Q)*len(n))], (n[I%len(n)]).astype(int), dS[i, (I//(len(n)))%len(Q)]*Q[(I//(len(n)))%len(Q)]*frac[(I//(len(n)))%len(Q)]).reshape(len(PEs),len(Q), len(n))*V_mash[i]
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
    return np.transpose(P, (1,0,2)), np.sum(S*pois, axis=2)



def make_3D2(t, N, F, Tf, Ts, R, a, eta, Q, T, St):
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
    for i in range(len(r_mash)):
        print(i)
        dS=make_dS(r_mash[i],  pmt_mid, pmt_r, pmt_up)
        b=binom.pmf(nu[I//(len(Q)*len(t)*len(n))], (n[I%len(n)]).astype(int), dS[(I//(len(n)*len(t)))%len(Q)]*Q[(I//(len(n)*len(t)))%len(Q)]*model[(I//len(n))%len(t),(I//(len(n)*len(t)))%len(Q)])
        B+=b.reshape(len(nu), len(Q), len(t), len(n))*V_mash[i]
        if np.any(b<0):
            print('B<0')
            sys.exit()
        if np.any(b>1):
            print('B>1')
            print(dS)
            print(r_mash[i])
            for j in np.nonzero(np.ravel(b)>1)[0]:
                print(np.ravel(b)[j]-1, nu[I//(len(Q)*len(t)*len(n))][j], (n[I%len(n)][j]).astype(int), dS[(I//(len(n)*len(t)))%len(Q)][j], Q[(I//(len(n)*len(t)))%len(Q)][j],
                    model[(I//len(n))%len(t),(I//(len(n)*len(t)))%len(Q)][j])
            sys.exit()
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



N=60*662
rec=np.recarray(1, dtype=[
    ('Q', 'f8', len(pmts)),
    ('T', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('F', 'f8', 1),
    ('Tf', 'f8', 1),
    ('Ts', 'f8', 1),
    ('R', 'f8', 1),
    ('a', 'f8', 1),
    ('eta', 'f8', 1),
    ])

rec[0]=([0.19657476,  0.13754033,  0.12743771,  0.18797336,  0.17835696,  0.3510241],
 [42.00312603, 42.07819445, 42.03561186, 42.05469875, 42.03181254 ,42.13596326],
  [0.94580769,  0.61208912,  0.84663691,  1.25148529,  0.78060014,  0.59422144],
  0.09440092,  2.06581567, 37.59474049,  0.68417731,  0.454177,    0.19242887)


t=np.arange(300)
dt=t[1]-t[0]

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

PEs=np.arange(350)
m, model=make_3D(t, N, rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0], rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0], dS, PEs)
s, GS, GS_spectrum, Sspectra, Gtrp, Gsng, GRtrp, GRsng=Sim2(N, rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0], rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0], PEs, 0)

#m2=make_3D2(t, N, rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0], rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0])



plt.show()
fig, ax=plt.subplots(2,3)
# model=make_spectra(m, PEs)
# model2=make_spectra(m2, PEs)
# Smodel=make_spectra(s[:,:100,:], PEs)

for i in range(len(pmts)):
    np.ravel(ax)[i].plot(PEs, model[:,i], 'r-.', label='model')
    # np.ravel(ax)[i].plot(PEs, Smodel[:,i], 'g-.', label='model2')
    np.ravel(ax)[i].plot(PEs, Sspectra[:,i]/np.sum(Sspectra[:,i]), 'b-.', label='sim')
    np.ravel(ax)[i].legend()
plt.show()
