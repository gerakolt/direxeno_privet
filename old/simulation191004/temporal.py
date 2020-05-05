from classes import WF, Event
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fun import do_smd, do_dif, Find_Peaks, Fix_Peaks, Analize_Peaks, find_peaks, Fit_Decay, fix_peaks, analize_peaks
from hit_fun_order import find_peaks, Reconstruct_WF
from scipy.optimize import curve_fit
from scipy import ndimage
from scipy import special
import sys
from scipy.stats import chi2
from scipy.stats import poisson


delay_sigma=1

def func(t, tau, T):
    s=delay_sigma
    y=T/(np.sqrt(2)*s)+s/(np.sqrt(2)*tau)
    a=np.exp(-t/tau)*(special.erf(y)-special.erf(y-t/(np.sqrt(2)*s)))
    N=tau*(1+special.erf(y-s/(np.sqrt(2)*tau)))*np.exp(-s**2/(2*tau**2)-np.sqrt(2)*s*y/tau)
    if np.sum(a)==0 and type(t)=='numpy.float64':
        print(type(t))
        plt.plot(t, a, 'k.')
        plt.title('sum=0, tau={:3.2f}, T={:3.2f}, sigma={:3.2f}'.format(tau,T,s))
        plt.show()
    if np.isnan(np.sum(a)):
        plt.plot(t, a, 'k.')
        plt.title('sum=nan, tau={:3.2f}, T={:3.2f}, sigma={:3.2f}'.format(tau,T,s))
        plt.show()
    return a, N


def func2exp(t, tau_f, tau_s, frac_f, T):
    [a_f, N_f]=func(t, tau_f, T)
    [a_s, N_s]=func(t, tau_s, T)
    return frac_f*a_f/N_f+(1-frac_f)*a_s/N_s




def find_chi2_T(x, dx, y, func, p0, bounds, PE_num):
    T=np.linspace(bounds[0][3], bounds[1][3], 50)
    Chi=np.array([])
    for t in T:
        p=[p0[0], p0[1], p0[2], t]
        chi=0
        for i in range(len(x)):
            if func(x[i], *p)>0:
                chi+=(y[i]-dx[i]*PE_num*func(x[i], *p))**2/(dx[i]*PE_num*func(x[i], *p))
            if y[i]>0 and func(x[i], *p)<=0:
                chi+=(y[i]-dx[i]*PE_num*func(x[i], *p))**2/(y[i])
        Chi=np.append(Chi, chi)
    return T[np.argsort(Chi)[:5]], chi2.pdf(Chi[np.argsort(Chi)[:5]], np.amin(Chi))


def find_Q(x, dx, y, func, p, bounds, PE_num, Delay, P_Delay):
    Q=0
    for j in range(len(Delay)):
        p[3]=Delay[j]
        P=1
        for i in range(len(x)):
            P=P*poisson.pmf(y[i], dx[i]*PE_num*func(x[i], *p))
            if dx[i]*PE_num*func(x[i], *p)<0:
                print('negative expectation')
                print('p=', p)
                print(dx[i]*PE_num*func(x[i], *p))
                sys.exit()
        Q+=P*P_Delay[j]
    if np.isnan(Q):
        print('Q is nan')
        print('p=', p)
        print('p_delay=', P_Delay)
        sys.exit()
    return Q


def find_max_likelihood(x, dx, y, func, p0, bounds, PE_num):
    #Nelder–Mead method
    alpha=1
    gamma=1
    rho=0.5
    sigma=0.5
    for k in range(5):
        p1=0.5*(bounds[0]+p0)
        p2=0.5*(bounds[1]+p0)
        p3=0.5*(p1+p0)
        p=np.array([p0, p1, p2, p3])
        Q=[0, 0, 0, 0]
        Delay, P_Delay=find_chi2_T(x, dx, y, func, p0, bounds, PE_num)
        Q[0]=find_Q(x, dx, y, func, p[0], bounds, PE_num, Delay, P_Delay)
        if Q[0]==0:
            print('Q of p0 = 0')
            sys.exit()
        for i in range(1, len(Q)):
            Q[i]=find_Q(x, dx, y, func, p[i], bounds, PE_num, Delay, P_Delay)
            while Q[i]<=0:
                p[i]=p[i]+0.5*(p[0]-p[i])
                Q[i]=Q[i]=find_Q(x, dx, y, func, p[i], bounds, PE_num, Delay, P_Delay)
                print('Srinking the intial ps dis=', np.sqrt(np.sum((p[i]-p[0])**2)))
        for j in range(5):
            ind=np.argsort(Q)
            cm=(p[ind[-1]]*Q[ind[-1]]+p[ind[-2]]*Q[ind[-2]]+p[ind[-3]]*Q[ind[-3]])/(Q[ind[-1]]+Q[ind[-2]]+Q[ind[-3]])
            if np.any(cm<bounds[0]) or np.any(cm>bounds[1]):
                print('cm is out if bounds')
                print('cm = ', cm)
                print('upper bound = ', bounds[1])
                print('lower bound = ', bounds[0])
                sys.exit()
            pr=cm+alpha*(cm-p[ind[0]])
            while np.any(pr<bounds[0]) or np.any(pr>bounds[1]):
                pr=cm+0.5*(pr-cm)
            Qr=find_Q(x, dx, y, func, pr, bounds, PE_num, Delay, P_Delay)

            if Qr>Q[ind[-1]]:
                pe=pr+gamma*(pr-cm)
                while np.any(pe<bounds[0]) or np.any(pe>bounds[1]):
                    pe=pr+0.5*(pe-pr)
                Qe=find_Q(x, dx, y, func, pe, bounds, PE_num, Delay, P_Delay)
                if Qe>Qr:
                    p[ind[0]]=pe
                    Q[ind[0]]=Qe
                else:
                    p[ind[0]]=pr
                    Q[ind[0]]=Qr

            elif Qr<Q[ind[1]]:
                pc=cm+rho*(cm-p[ind[0]])
                while np.any(pc<bounds[0]) or np.any(pc>bounds[1]):
                    pc=cm+0.5*(pc-cm)
                Qc=find_Q(x, dx, y, func, pc, bounds, PE_num, Delay, P_Delay)
                if Qc>Q[ind[1]]:
                    p[ind[0]]=pc
                    Q[ind[0]]=Qc
                else:
                    p[ind[0]]=cm+sigma*(cm-p[ind[0]])
                    while np.any(p[ind[0]]<bounds[0]) or np.any(p[ind[0]]>bounds[1]):
                        p[ind[0]]=cm+0.5*(p[ind[0]]-cm)
                    p[ind[1]]=cm+sigma*(cm-p[ind[1]])
                    while np.any(p[ind[1]]<bounds[0]) or np.any(p[ind[1]]>bounds[1]):
                        p[ind[1]]=cm+0.5*(p[ind[1]]-cm)
                    p[ind[2]]=cm+sigma*(cm-p[ind[2]])
                    while np.any(p[ind[2]]<bounds[0]) or np.any(p[ind[2]]>bounds[1]):
                        p[ind[2]]=cm+0.5*(p[ind[2]]-cm)
                    Q[ind[0]]=find_Q(x, dx, y, func, p[ind[0]], bounds, PE_num, Delay, P_Delay)
                    while Q[ind[0]]==0:
                        p[ind[0]]=p[ind[0]]+0.5*(cm-p[ind[0]])
                        Q[ind[0]]=find_Q(x, dx, y, func, p[ind[0]], bounds, PE_num, Delay, P_Delay)
                    while Q[ind[1]]==0:
                        p[ind[1]]=p[ind[1]]+0.5*(cm-p[ind[1]])
                        Q[ind[1]]=find_Q(x, dx, y, func, p[ind[1]], bounds, PE_num, Delay, P_Delay)
                    while Q[ind[2]]==0:
                        p[ind[2]]=p[ind[2]]+0.5*(cm-p[ind[2]])
                        Q[ind[2]]=find_Q(x, dx, y, func, p[ind[2]], bounds, PE_num, Delay, P_Delay)
            else:
                p[ind[0]]=pr
                Q[ind[0]]=Qr

        p0=p[ind[-1]]
    return p0


max_like_tau_f=[]
oreginal_tau_f=[]
max_like_tau_s=[]
oreginal_tau_s=[]
max_like_frac_f=[]
oreginal_frac_f=[]
max_like_delay=[]
oreginal_delay=[]
bounds=np.array([[1,30,0,0],[20,70,1,20]])
while len(oreginal_delay)<200:
    np.random.seed(int(int(time.time()*10000)%1e9))
    PE_num=int(np.random.normal(150, 50))
    while PE_num<1:
        PE_num=int(np.random.normal(35, 10))

    np.random.seed(int(int(time.time()*10000)%1e9))
    tau_f=np.random.uniform(bounds[0][0], bounds[1][0])

    np.random.seed(int(int(time.time()*10000)%1e9))
    tau_s=np.random.uniform(bounds[0][1], bounds[1][1])

    np.random.seed(int(int(time.time()*10000)%1e9))
    delay=np.random.uniform(bounds[0][3], bounds[1][3])

    np.random.seed(int(int(time.time()*10000)%1e9))
    frac_f=np.random.uniform(bounds[0][2], bounds[1][2])

    print(len(oreginal_delay), tau_f, tau_s, frac_f, delay)

    np.random.seed(int(int(time.time()*10000)%1e9))
    PE0_s=np.random.exponential(tau_s, PE_num-int(PE_num*frac_f))

    np.random.seed(int(int(time.time()*10000)%1e9))
    PE0_f=np.random.exponential(tau_f, int(PE_num*frac_f))

    np.random.seed(int(int(time.time()*10000)%1e9))
    PE_f=PE0_f+np.random.normal(delay, delay_sigma, len(PE0_f))

    np.random.seed(int(int(time.time()*10000)%1e9))
    PE_s=PE0_s+np.random.normal(delay, delay_sigma, len(PE0_s))

    PE=np.append(PE_s, PE_f)

    bins=np.array([0])
    j=0
    while bins[-1]<200:
        j+=1
        bins=np.append(bins, bins[-1]+1/8*j+3/8)

    x=0.5*(bins[1:]+bins[:-1])
    dx=bins[1:]-bins[:-1]
    h, bins=np.histogram(PE, bins=bins, range=[0,200])

    np.random.seed(int(int(time.time()*10000)%1e9))
    tau_f0=np.random.uniform(bounds[0][0], bounds[1][0])
    np.random.seed(int(int(time.time()*10000)%1e9))
    tau_s0=np.random.uniform(bounds[0][1], bounds[1][1])
    np.random.seed(int(int(time.time()*10000)%1e9))
    frac_f0=np.random.uniform(bounds[0][2], bounds[1][2])
    np.random.seed(int(int(time.time()*10000)%1e9))
    delay0=np.random.uniform(bounds[0][3], bounds[1][3])

    p0=np.array([tau_f0, tau_s0, frac_f0, delay0])

    try:
        p_max=find_max_likelihood(x, dx, h, func2exp, p0, bounds, PE_num)

        max_like_tau_f.append(p_max[0])
        max_like_tau_s.append(p_max[1])
        max_like_frac_f.append(p_max[2])
        max_like_delay.append(p_max[3])
        oreginal_tau_f.append(tau_f)
        oreginal_tau_s.append(tau_s)
        oreginal_frac_f.append(frac_f)
        oreginal_delay.append(delay)
    except:
        temp=1

def lin(x, a,b):
    return a*x+b

fig=plt.figure()
ax=fig.add_subplot(221)
ax.plot(oreginal_tau_f, max_like_tau_f, 'k.', label='fast')
p0=[1,0]
ax.plot(oreginal_tau_f, lin(np.array(oreginal_tau_f), *p0), 'g--')
try:
    p, pch = curve_fit(lin, oreginal_tau_f, max_like_tau_f, p0=p0)
    ax.plot(oreginal_tau_f, lin(np.array(oreginal_tau_f), *p), 'r--', label='{:3.2f}x+{:3.2f}'.format(p[0], p[1]))
    ax.legend()
except:
    temp=1

ax=fig.add_subplot(222)
ax.plot(oreginal_tau_s, max_like_tau_s, 'k.', label='slow')
p0=[1,0]
ax.plot(oreginal_tau_s, lin(np.array(oreginal_tau_s), *p0), 'g--')
try:
    p, pch = curve_fit(lin, oreginal_tau_s, max_like_tau_s, p0=p0)
    ax.plot(oreginal_tau_s, lin(np.array(oreginal_tau_s), *p), 'r--', label='{:3.2f}x+{:3.2f}'.format(p[0], p[1]))
    ax.legend()
except:
    temp=1

ax=fig.add_subplot(223)
ax.plot(oreginal_frac_f, max_like_frac_f, 'k.', label='frac f')
p0=[1,0]
ax.plot(oreginal_frac_f, lin(np.array(oreginal_frac_f), *p0), 'g--')
try:
    p, pch = curve_fit(lin, oreginal_frac_f, max_like_frac_f, p0=p0)
    ax.plot(oreginal_frac_f, lin(np.array(oreginal_frac_f), *p), 'r--', label='{:3.2f}x+{:3.2f}'.format(p[0], p[1]))
    ax.legend()
except:
    temp=1

ax=fig.add_subplot(224)
ax.plot(oreginal_delay, max_like_delay, 'k.', label='delay')
p0=[1,0]
ax.plot(oreginal_delay, lin(np.array(oreginal_delay), *p0), 'g--')
try:
    p, pch = curve_fit(lin, oreginal_delay, max_like_delay, p0=p0)
    ax.plot(oreginal_delay, lin(np.array(oreginal_delay), *p), 'r--', label='{:3.2f}x+{:3.2f}'.format(p[0], p[1]))
    ax.legend()
except:
    temp=1
fig.suptitle('5 itterations')
plt.show()
