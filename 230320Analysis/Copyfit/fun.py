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


def make_P(Spe, p01):
    n=300
    P=np.zeros((n,n))
    P[0,1]=p01
    P[1,1]=0.5*(1+erf(0.5/(np.sqrt(2)*Spe)))-p01

    for i in np.arange(2, n):
        P[i,1]=0.5*(erf((i-0.5)/(np.sqrt(2)*Spe))-erf((i-1.5)/(np.sqrt(2)*Spe)))

    for i in range(n):
        for j in range(2,n):
            P[i,j]=np.sum(P[:i+1,1]*np.flip(P[:i+1,j-1]))

    P[1,0]=1-np.sum(P[1,1:])
    if P[1,0]>0.5:
        return 1+P[1,0]
    if P[1,0]<0:
        return 1-P[1,0]
    P[2:,0]=P[1,0]**(np.arange(2,n))
    P[0,0]=1-np.sum(P[1:,0])
    if np.any(P<0):
        print('P<0')
        print('Spe=', Spe, 'P01=',p01)
        print(np.nonzero(P<0))
        print(P[:2,:2])
        sys.exit()

    if np.any(P>=1):
        print('P=1')
        print('Spe=', Spe, 'P01=',p01)
        print(np.nonzero(P>=1))
        print(P[:2,:2])
        sys.exit()
    return P

def model_spec(ns, NQ, P, a):
    h=poisson.pmf(ns, NQ)
    return np.ravel(a*np.matmul(P[:,ns], h.reshape(len(ns),1))[ns])

def model_area(areas, Mpad, Spad, Mpe, Spe, a_pad, a_spe, a_dpe):
    return a_pad*np.exp(-0.5*(areas-Mpad)**2/(Spad)**2)+a_spe*np.exp(-0.5*(areas-(Mpe+Mpad))**2/((Mpe*Spe)**2+Spad**2))+a_dpe*np.exp(-0.5*(areas-(Mpe+2*Mpad))**2/(2*(Mpe*Spe)**2+Spad**2))

# def make_P(Spe):
#     P=np.zeros((300, 300))
#     P[0,0]=1
#     for i in range(len(P[:,0])):
#         r=np.linspace(i-0.5,i+0.5,1000)
#         dr=r[1]-r[0]
#         P[i,1]=dr*np.sum(np.exp(-0.5*(r-1)**2/Spe**2))/(np.sqrt(2*np.pi)*Spe)
#     P[:,1]=P[:,1]/np.sum(P[:,1])
#     for j in range(2, len(P[0,:])):
#         for i in range(len(P[:,0])):
#             P[i,j]=np.sum(P[:i+1,1]*np.flip(P[:i+1,j-1]))
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
#     return P

# def make_P(p0, p2):
#     P=np.zeros((300, 300))
#     P[0,0]=1
#     P[0,1]=p0
#     P[1,1]=1-p0-p2
#     P[2,1]=p2
#     # P[:,1]=P[:,1]/np.sum(P[:,1])
#     for j in range(2, len(P[0,:])):
#         for i in range(len(P[:,0])):
#             P[i,j]=np.sum(P[:i+1,1]*np.flip(P[:i+1,j-1]))
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
#     # print(np.sum(P, axis=0)[:6])
#     # print(np.sum(P, axis=1)[:6])
#     # sys.exit()
#     return P


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

# def make_z(h):
#     z=np.zeros((np.shape(h)[0], 1000))
#     p0=make_P0(h)
#     C=0
#     for k in range(np.shape(z)[1]):
#         C+=p0[k]*(1-h[0,k])
#     for i in range(1, np.shape(z)[0]):
#         for k in range(np.shape(z)[1]):
#             z[i,0]+=p0[k]*h[i,k]
#     for i in range(1, np.shape(z)[0]):
#         for j in range(1, np.shape(z)[1]):
#             for k in range(np.shape(z)[1]):
#                 z[i,j]+=p0[k]*(1-h[0,k])*h[i,j+k]
#     return z/C
#
# def make_P0(h):
#     p0=np.zeros(2000)
#     for i in range(2000):
#         p0[i]=np.prod(h[0,:i])
#     return p0

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
    return make_z(h[:n])

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

# def model_spec(Ns, NQ, P, a):
#     dn=Ns[1]-Ns[0]
#     h=np.zeros(len(Ns))
#     for i, N in enumerate(Ns):
#         for n in range(int(np.ceil(N-0.5*dn)), int(np.ceil(N+0.5*dn))):
#             h[i]+=a*np.matmul(P, poisson.pmf(np.arange(len(P)), NQ)).reshape(len(P),1))[n]
#     return h
