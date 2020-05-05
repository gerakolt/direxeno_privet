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
    Hs_BG=[]
    N_events=[]
    Hs_spec=[]
    Hs_spec_BG=[]
    path='/home/gerak/Desktop/DireXeno/190803/{}/'.format(source)
    for pmt in pmts:
        data=np.load(path+'PMT{}/H.npz'.format(pmt))
        Hs.append(data['H'])
        Hs_BG.append(data['H_BG'])
        N_events.append(data['N_events'])
        Hs_spec.append(data['h_spec'])
        Hs_spec_BG.append(data['h_spec_BG'])
        Ns.append(data['Ns'])
    return Hs, N_events, Hs_BG, Ns, Hs_spec, Hs_spec_BG

def Const(tau,T,s):
    C=1/(1-erf(s/(np.sqrt(2)*tau)+T/(np.sqrt(2)*s))+np.exp(-s**2/(2*tau**2)-T/tau)*(1+erf(T/(np.sqrt(2)*s))))
    return C

def Int(t, tau, T, s):
    dt=t[1]-t[0]
    return np.exp(-t/tau)*(1-erf(s/(np.sqrt(2)*tau)-(t-T)/(np.sqrt(2)*s)))*dt*Const(tau,T,s)/tau

# def make_P(Spe, p0):
#     q=1-p0
#     P=np.zeros((300, 300))
#     P[0,0]=1
#     for i in range(len(P[:,0])):
#         r=np.linspace(i-0.5,i+0.5,1000)
#         dr=r[1]-r[0]
#         P[i,1]=dr*np.sum(np.exp(-0.5*(r-1)**2/Spe**2))/(np.sqrt(2*np.pi)*Spe)
#     for j in range(2, len(P[0,:])):
#         for i in range(len(P[:,0])):
#             P[i,j]=np.sum(P[:i+1,1]*np.flip(P[:i+1,j-1]))
#     for i in range(1, len(P[:,0])):
#         P[i]=np.sum(binom.pmf(np.arange(i+1), i, q)*P[:i+1].T, axis=1)
#     if np.isnan(np.any(P)):
#         print('P is nan')
#         sys.exit()
#     if np.isinf(np.any(P)):
#         print('P is inf')
#         sys.exit()
#     if np.any(np.sum(P, axis=0)==0):
#         print(Spe)
#         print('P=0')
#         sys.exit()
#     return P/np.sum(P, axis=0)


def make_P(Spe, p0):
    q=1-p0
    P=np.zeros((300, 300))
    P[0,0]=q
    P[1,0]=p0
    for i in range(len(P[:,0])):
        r=np.linspace(i-0.5,i+0.5,1000)
        dr=r[1]-r[0]
        P[i,1]=dr*np.sum(np.exp(-0.5*(r-1)**2/Spe**2))/(np.sqrt(2*np.pi)*Spe)
    P[:,1]=P[:,1]/np.sum(P[:,1])
    for j in range(2, len(P[0,:])):
        for i in range(len(P[:,0])):
            P[i,j]=np.sum(P[:i+1,1]*np.flip(P[:i+1,j-1]))
    # for i in range(1, len(P[:,0])):
    #     P[i]=np.sum(binom.pmf(np.arange(i+1), i, q)*P[:i+1].T, axis=1)
    if np.isnan(np.any(P)):
        print('P is nan')
        sys.exit()
    if np.isinf(np.any(P)):
        print('P is inf')
        sys.exit()
    if np.any(np.sum(P, axis=0)==0):
        print(Spe)
        print('P=0')
        sys.exit()
    P=P
    # print(np.sum(P, axis=0)[:5])
    # print(np.sum(P, axis=1)[:5])
    # sys.exit()
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
            print(i,j)
            for k in range(np.shape(z)[1]):
                z[i,j]+=p0[k]*(1-h[0,k])*h[i,j+k]
    print('make z done')
    return z/C


def make_P0(h):
    p0=np.zeros(1000)
    for i in range(1000):
        p0[i]=np.prod(h[0,:i])
    return p0

def model2(p, P, n):
    T=50
    [NQ, F, tf, ts, St]=p
    t=np.arange(2000)/5
    I=np.arange(np.shape(P)[1]*len(t))
    m=NQ*(F*Int(t,tf,T,St)+(1-F)*Int(t,ts,T,St))
    M=poisson.pmf(np.floor(I/len(t)), m[I%len(t)])
    h=np.matmul(P,M.reshape((np.shape(P)[1], len(t))))
    return h[:n, :1000]

def model3(p, P):
    [NQ, R, F, tf, ts, St, Spe]=p
    T=50
    t=np.arange(2000)/5
    dt=t[1]-t[0]
    I=np.arange(np.shape(P)[1]*len(t))
    m=NQ*(R*dt*(np.exp(-0.5*(t-T)**2/St**2)/(np.sqrt(2*np.pi)*St))+(1-R)*(F*Int(t,tf,T,St)+(1-F)*Int(t,ts,T,St)))
    M=poisson.pmf(np.floor(I/len(t)), m[I%len(t)])
    h=np.matmul(P,M.reshape((np.shape(P)[1], len(t))))
    return make_z(h)
