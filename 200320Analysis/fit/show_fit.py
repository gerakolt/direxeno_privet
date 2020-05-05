import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf
import sys
from scipy.optimize import minimize

pmt=8
path='/home/gerak/Desktop/DireXeno/190803/Co57/PMT{}/'.format(pmt)
data=np.load(path+'H.npz')
H_origin=data['H']
H=np.array(H_origin[:,:400])
shp=np.shape(H)
H=np.ravel(H)
N_events=data['N_events']

def Const(tau,T,s):
    C=1/(1-erf(s/(np.sqrt(2)*tau)+T/(np.sqrt(2)*s))+np.exp(-s**2/(2*tau**2)-T/tau)*(1+erf(T/(np.sqrt(2)*s))))
    return C

def Int(t, tau, T, s):
    dt=t[1]-t[0]
    return np.exp(-t/tau)*(1-erf(s/(np.sqrt(2)*tau)-(t-T)/(np.sqrt(2)*s)))*dt*Const(tau,T,s)/tau

def make_P(Spe, ns):
    P=np.zeros((ns[-1]+10, ns[-1]+10))
    P[0,0]=1
    for i in range(len(P[:,0])):
        r=np.linspace(i-0.5,i+0.5,1000)
        dr=r[1]-r[0]
        P[i,1]=dr*np.sum(np.exp(-0.5*(r-1)**2/Spe**2))/(np.sqrt(2*np.pi)*Spe)
    for j in range(2, len(P[0,:])):
        for i in range(len(P[:,0])):
            P[i,j]=np.sum(P[:i+1,1]*np.flip(P[:i+1,j-1]))
    return P




def make_z(h):
    # print('in make z')
    z=np.zeros((np.shape(h)[0], shp[1]))
    p0=make_P0(h)
    C=0
    for k in range(np.shape(z)[1]):
        C+=p0[k]*(1-h[0,k])
    for i in range(1, np.shape(z)[0]):
        for k in range(np.shape(z)[1]):
            z[i,0]+=p0[k]*h[i,k]
    for i in range(np.shape(z)[0]):
        for j in range(1, np.shape(z)[1]):
            for k in range(np.shape(z)[1]):
                z[i,j]+=p0[k]*(1-h[0,k])*h[i,j+k]
    return z/C

def make_P0(h):
    p0=np.zeros(shp[1]*2)
    for i in range(shp[1]*2):
        p0[i]=np.prod(h[0,:i])
    return p0

def model1(p):
    T=50
    [NQ, F, tf, ts, St, Spe]=p
    t=np.arange(shp[1]*2)/5
    P=make_P(Spe, np.arange(shp[0]))
    I=np.arange(np.shape(P)[1]*len(t))
    m=NQ*(F*Int(t,tf,T,St)+(1-F)*Int(t,ts,T,St))
    M=poisson.pmf(np.floor(I/len(t)), m[I%len(t)])
    h=np.matmul(P,M.reshape((np.shape(P)[1], len(t))))
    return make_z(h)

def model2(p):
    [NQ, R, F, tf, ts, St, Spe]=p
    T=50
    t=np.arange(shp[1]*2)/5
    dt=t[1]-t[0]
    P=make_P(Spe, np.arange(shp[0]))
    I=np.arange(np.shape(P)[1]*len(t))
    m=NQ*(R*dt*(np.exp(-0.5*(t-T)**2/St**2)/(np.sqrt(2*np.pi)*St))+(1-R)*(F*Int(t,tf,T,St)+(1-F)*Int(t,ts,T,St)))
    M=poisson.pmf(np.floor(I/len(t)), m[I%len(t)])
    h=np.matmul(P,M.reshape((np.shape(P)[1], len(t))))
    return make_z(h)

def L1(p):
    [NQ, F, tf, ts, St, Spe]=p

    if St<=0:
        return 1e8*(1-St)
    if Spe<=0:
        return 1e8*(1-Spe)
    if tf<St:
        return 1e8*(St-tf)

    h=np.ravel(model1(p)[:shp[0],:shp[1]])
    l1=0
    for i in range(len(h)):
        if H[i]==0 and h[i]>0:
            l1-=h[i]
        elif h[i]==0 and H[i]>0:
            l1+=H[i]-h[i]-H[i]*np.log(H[i])-H[i]*1e100
        elif h[i]>0 and H[i]>0:
            l1+=H[i]*np.log(h[i])-H[i]*np.log(H[i])+H[i]-h[i]
    print(-l1, p)
    return -l1


p0=[1.83879666e+02, 2.17205950e-02, 9.64126061e-02, 1.94396897e+01,
       3.92050222e+02, 7.89888533e-01, 2.40599886e-01]
p1=[1.83879666e+02, 1, 9.64126061e-02, 1.94396897e+01,
       3.92050222e+02, 0.2, 2.40599886e-01]
p2=[1.83879666e+02, 0, 1.5, 1,
       3.92050222e+02, 0.4, 2.40599886e-01]
p3=[1.83879666e+02, 0, 0, 1.94396897e+01,
       3.92050222e+02, 7.89888533e-01, 2.40599886e-01]

h0=N_events*model2(p0)
# h1=N_events*model2(p1)*p0[1]
# h2=N_events*model2(p2)*(1-p0[1])*p0[2]*0.45
# h3=N_events*model2(p3)*(1-p0[1])*(1-p0[2])

plt.figure(figsize=(20,10))
plt.plot(np.sum(H_origin.T*np.arange(np.shape(H_origin)[0]), axis=1), 'k.-', label='data')
plt.plot(np.sum(h0.T*np.arange(np.shape(h0)[0]), axis=1), 'r.-', label='delta')
# plt.plot(np.sum(h1.T*np.arange(np.shape(h1)[0]), axis=1), 'r.-', label='delta')
# plt.plot(np.sum(h2.T*np.arange(np.shape(h2)[0]), axis=1), 'g.-', label='fast')
# plt.plot(np.sum(h3.T*np.arange(np.shape(h3)[0]), axis=1), 'b.-', label='slow')
plt.legend()
plt.show()
