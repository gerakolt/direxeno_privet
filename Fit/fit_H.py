import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf
from scipy.special import comb

source='Cs137'
Type=''
pmt=0
peak=1
Chi2=[]
ID=np.array([])
# path='/home/gerak/Desktop/DireXeno/'+source+'_190803'+Type+'/spectra{}/'.format(pmt)
path='/home/gerak/Desktop/DireXeno/WFsim/PMT{}/'.format(pmt)
Data = np.load(path+'H.npz')
H=Data['H']
bins_spec=Data['bins_spec']
spec_x=0.5*(bins_spec[1:]+bins_spec[:-1])
spec_y=Data['h_spec']
left=Data['left']
right=Data['right']
n_events=Data['n_events']



Data=np.load('/home/gerak/Desktop/DireXeno/190803/pulser/area{}.npz'.format(pmt))
bins_area=Data['bins_area']
h_area=Data['h_area']

for filename in os.listdir('/home/gerak/Desktop/DireXeno/190803/pulser'):
    if filename.endswith("_{}.npz".format(pmt)) or filename.startswith("delay_{}".format(pmt)):
        Data=np.load('/home/gerak/Desktop/DireXeno/190803/pulser/'+filename)
        bins_delay=Data['bins_delay']
        h_delay=Data['h_delay']

x_spec=0.5*(bins_spec[1:]+bins_spec[:-1])
x_area=0.5*(bins_area[1:]+bins_area[:-1])
t=np.arange(1000)/5
ns=np.arange(len(H[:,0]))
Ns=np.array(x_spec).astype(int)
dt=t[1]-t[0]
I=np.arange(H.size)
J=I[np.nonzero(np.logical_and(I%len(t)<right, I%len(t)>left))[0]]
# delay_rng=np.nonzero(np.logical_and(delay_x>-1, delay_x<5))[0]
area_rng=np.nonzero(np.logical_and(x_area>700, x_area<4000))[0]
shp=np.shape(H)
H=np.ravel(H)




def Const(tau,T,s):
    C=1/(1-erf(s/(np.sqrt(2)*tau)+T/(np.sqrt(2)*s))+np.exp(-s**2/(2*tau**2)-T/tau)*(1+erf(T/(np.sqrt(2)*s))))
    return C

def Int(t, tau, T, s):
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
    return P[ns,:]

def model(NQ, F, ts, T, St, Spe):
    P=make_P(Spe, ns)
    I=np.arange(np.shape(P)[1]*len(t))
    m=NQ*(F*dt*np.exp(-0.5*(t-T)**2/St**2)/(np.sqrt(2*np.pi)*St)+(1-F)*Int(t,ts,T,St))
    M=poisson.pmf(np.floor(I/len(t)), m[I%len(t)])
    return n_events*np.ravel(np.matmul(P,M.reshape((np.shape(P)[1], len(t)))))


def Norm(x, a, b, c):
    return a*(np.exp(-0.5*(x-b)**2/c**2)/np.sum(np.exp(-0.5*(x-b)**2/c**2)))


def L(p):
    [T, St, Spe, Mpe]=p
    NQ=50
    ts=45
    if St<=0:
        return 1e8*(1-St)
    if Spe<=0:
        return 1e8*(1-Spe)
    if T<=0:
        return 1e8*(1-T)
    if np.isinf(Const(ts,T,St)):
        if np.exp(-St**2/(2*ts**2))==0:
            return 1e8*St**2/(2*ts**2)
        if erf(T/(np.sqrt(2)*St))==1:
            return 1e8*T/(np.sqrt(2)*St)
        if np.exp(-T/ts)==0:
            return 1e8*T/ts


    h=model(NQ, 0, ts, T, St, Spe)
    if np.any(np.isnan(h)):
        print('h is nan')
        return 1e8


    l1=0
    l2=0
    l3=0
    l4=0
    for i in J:
        if H[i]==0 and h[i]>0:
            l1-=h[i]
        elif h[i]==0 and H[i]>0:
            l1+=H[i]-h[i]-H[i]*np.log(H[i])-H[i]*1e100
        elif h[i]>0 and H[i]>0:
            l1+=H[i]*np.log(h[i])-H[i]*np.log(H[i])+H[i]-h[i]

    # for i in delay_rng:
    #     if delay_h[i]==0 and delay_y[i]>0:
    #         l2-=delay_y[i]
    #     elif delay_y[i]==0 and delay_h[i]>0:
    #         l2+=delay_h[i]-delay_y[i]-delay_h[i]*np.log(delay_h[i])-delay_h[i]*1e100
    #     elif delay_y[i]>0 and delay_h[i]>0:
    #         l2+=delay_h[i]*np.log(delay_y[i])-delay_h[i]*np.log(delay_h[i])+delay_h[i]-delay_y[i]

    mu=np.sum(np.sum(h.reshape(shp).T*ns, axis=1)[left:right])/n_events
    #y=np.sum(spec_y)*poisson.pmf(spec_x, mu)/np.sum(poisson.pmf(spec_x[spec_y>0], mu))
    y=poisson.pmf(np.arange(108), mu)
    # P=make_P(Spe, spec_x.astype(int))
    # y=np.ravel(np.matmul(P, Y.reshape(len(Y),1)))
    y=np.sum(spec_y)*y/np.sum(y)
    for i in np.nonzero(spec_y>0)[0]:
        if spec_y[i]==0 and y[i]>0:
            l3-=y[i]
        elif y[i]==0 and spec_y[i]>0:
            l3+=spec_y[i]-y[i]-spec_y[i]*np.log(spec_y[i])-spec_y[i]*1e100
        elif y[i]>0 and spec_y[i]>0:
            l3+=spec_y[i]*np.log(y[i])-spec_y[i]*np.log(spec_y[i])+spec_y[i]-y[i]

    y=Norm(x_area[area_rng], np.sum(h_area[area_rng]), Mpe, Spe*Mpe)
    for i in range(len(y)):
        if h_area[area_rng][i]==0 and y[i]>0:
            l4-=y[i]
        elif y[i]==0 and h_area[area_rng][i]>0:
            l4+=h_area[area_rng][i]-y[i]-h_area[area_rng][i]*np.log(h_area[area_rng][i])-h_area[area_rng][i]*1e100
        elif y[i]>0 and h_area[area_rng][i]>0:
            l4+=h_area[area_rng][i]*np.log(y[i])-h_area[area_rng][i]*np.log(h_area[area_rng][i])+h_area[area_rng][i]-y[i]
    print('In L' ,-(l1/len(J)+l4/len(x_area[area_rng])+l3/len(np.nonzero(spec_y>0)[0])))
    return -(l1/len(J)+10*l4/len(x_area[area_rng])+l3/len(np.nonzero(spec_y>0)[0]))

# p0=np.array([50, 45, 43, 0.7, 0.77, 2000])
p0=np.array([43, 0.7, 0.77, 2000])
NQ=50
ts=45

p1=minimize(L, p0, method='Nelder-Mead', options={'disp':True, 'maxfev':10000})
p1=minimize(L, p1.x, method='Nelder-Mead', options={'disp':True, 'maxfev':10000})

# [NQ, ts, T, St, Spe, Mpe]=p1.x
[T, St, Spe, Mpe]=p1.x
h=model(NQ, 0, ts, T, St, Spe).reshape(shp)
H=H.reshape(shp)


fig=plt.figure()

ax=fig.add_subplot(221)
ax.plot(t, np.average(H, axis=0, weights=np.arange(len(H[:,0]))), 'k.-')
ax.plot(t, np.average(h, axis=0, weights=np.arange(len(H[:,0]))), 'r.-', label='NQ={:3.2f}, ts={:3.2f} ns'.format(NQ, ts))
ax.fill_between(t[left:right], y1=0, y2=np.average(H, axis=0, weights=np.arange(len(H[:,0])))[left:right])
ax.legend()

ax=fig.add_subplot(222)
x_spec=0.5*(bins_spec[1:]+bins_spec[:-1])
ax.step(x_spec, spec_y, 'k')
mu=np.sum(np.sum(h.T*ns, axis=1)[left:right])/n_events
y=np.sum(spec_y)*poisson.pmf(spec_x, mu)/np.sum(poisson.pmf(spec_x[spec_y>0], mu))
ax.plot(x_spec, y, 'r.-')

ax=fig.add_subplot(223)
x_area=0.5*(bins_area[1:]+bins_area[:-1])
ax.step(x_area, h_area, 'k')
ax.plot(x_area[area_rng], Norm(x_area[area_rng], np.sum(h_area[area_rng]), Mpe, Mpe*Spe), 'r.-', label='{:3.2f}+-{:3.2f}'.format(Mpe, Mpe*Spe))
ax.legend()

ax=fig.add_subplot(224)
x_delay=0.5*(bins_delay[1:]+bins_delay[:-1])
ax.step(x_delay, h_delay, 'k')

plt.show()
