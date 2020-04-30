import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import random
from fit_decay import Fit_Decay
from scipy.optimize import curve_fit
from scipy.special import sph_harm


r=0.25
pmt=np.array([]).astype(int)
good_chns=np.array([0,1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,19]).astype(int)
pmt_mid=np.array([[1/np.sqrt(2)*np.cos(120*np.pi/180), 1/np.sqrt(2)*np.sin(120*np.pi/180), 1/np.sqrt(2)],
                [-1/np.sqrt(2), 0, 1/np.sqrt(2)],
                [1/np.sqrt(2)*np.cos(240*np.pi/180), 1/np.sqrt(2)*np.sin(240*np.pi/180), 1/np.sqrt(2)],
                [0,1,0],
                [-1/np.sqrt(2), 1/np.sqrt(2), 0],
                [-1,0,0],
                [-1/np.sqrt(2), -1/np.sqrt(2),0],
                [1/np.sqrt(2)*np.cos(120*np.pi/180), 1/np.sqrt(2)*np.sin(120*np.pi/180), -1/np.sqrt(2)],
                [-1/np.sqrt(2), 0, -1/np.sqrt(2)],
                [1/np.sqrt(2)*np.cos(240*np.pi/180), 1/np.sqrt(2)*np.sin(240*np.pi/180), -1/np.sqrt(2)],
                [1/np.sqrt(2)*np.cos(300*np.pi/180), 1/np.sqrt(2)*np.sin(300*np.pi/180), 1/np.sqrt(2)],
                [1/np.sqrt(2), 0,1/np.sqrt(2)],
                [1/np.sqrt(2)*np.cos(60*np.pi/180), 1/np.sqrt(2)*np.sin(60*np.pi/180), 1/np.sqrt(2)],
                [0, -1, 0],
                [1/np.sqrt(2), -1/np.sqrt(2), 0],
                [1, 0, 0],
                [1/np.sqrt(2), 1/np.sqrt(2), 0],
                [1/np.sqrt(2)*np.cos(300*np.pi/180), 1/np.sqrt(2)*np.sin(300*np.pi/180), -1/np.sqrt(2)],
                [1/np.sqrt(2), 0, -1/np.sqrt(2)],
                [1/np.sqrt(2)*np.cos(60*np.pi/180), 1/np.sqrt(2)*np.sin(60*np.pi/180), -1/np.sqrt(2)],
                ])

pmt_r=np.zeros((len(pmt_mid),3))
pmt_l=np.zeros((len(pmt_mid),3))
pmt_up=np.zeros((len(pmt_mid),3))
pmt_dn=np.zeros((len(pmt_mid),3))

for i in range(len(pmt_mid)):
    a=pmt_mid[i][0]
    b=pmt_mid[i][1]
    pmt_r[i]=[-r*b/np.sqrt(a**2+b**2), r*a/np.sqrt(a**2+b**2), 0]
    pmt_l[i]=[r*b/np.sqrt(a**2+b**2),-r*a/np.sqrt(a**2+b**2), 0]
    pmt_up[i]=np.cross(pmt_mid[i], pmt_r[i])
    pmt_dn[i]=np.cross(pmt_mid[i], pmt_l[i])


t0_dn=10
t0=20
t0_up=30

a0_dn=0
a0=500
a0_up=5000

tau_f_dn=0
tau_f=27
tau_f_up=200

tau_s_dn=0
tau_s=45
tau_s_up=200

R_s_dn=0
R_s=0.5
R_s_up=1

T_dn=0
T=0.5
T_up=10

s_dn=0
s=1.5
s_up=10


def shootPEs(PE_num, R_s, strng):
    fig = plt.figure(figsize=[15,10])
    ax = fig.add_subplot(111, projection='3d')
    t1=np.random.exponential(scale=45, size=PE_num)
    t2=t1+t0+np.random.normal(loc=T, scale=s, size=PE_num)
    pmt=[]
    t=[]
    i=0

    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_zlim(-2,2)
    if strng=='plot':
        for J in range(20):
            if J in good_chns:
                ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='g')
                ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='g')
                ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='g')
                ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='g')

                ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='g')
                ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='g')
                ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='g')
                ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='g')

                ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='g')
                ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='g')
                ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='g')
                ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='g')
            else:
                ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='k', alpha=0.2)
                ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='k', alpha=0.2)
                ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='k', alpha=0.2)
                ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='k', alpha=0.2)

                ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='k', alpha=0.2)
                ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='k', alpha=0.2)
                ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='k', alpha=0.2)
                ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='k', alpha=0.2)

                ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='k', alpha=0.2)
                ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='k', alpha=0.2)
                ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='k', alpha=0.2)
                ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='k', alpha=0.2)

    while i<int(R_s*(PE_num)):
        #print('in iso', i , 'out of', int(R_s*(PE_num)))
        costheta=np.random.uniform(-1,1)
        phi=np.random.uniform(0,2*np.pi)
        v=np.array([np.sin(np.arccos(costheta))*np.cos(phi), np.sin(np.arccos(costheta))*np.sin(phi), costheta])
        if i%50==0 and strng=='plot':
            ax.quiver(0,0,0,v[0],v[1],v[2], color='y', arrow_length_ratio=0)
        alpha=np.zeros(len(pmt_mid))+1e6
        for j in range(len(pmt_mid)):
            if np.sum(v*pmt_mid[j])>0:
                alpha[j]=1/(np.sum(v*pmt_mid[j]))
        J=np.argmin(alpha)
        Alpha=np.amin(alpha)
        if (np.sum((pmt_mid[J]-Alpha*v)*pmt_r[J])<r**2 and np.sum((pmt_mid[J]-Alpha*v)*pmt_r[J])>-r**2 and
                np.sum((pmt_mid[J]-Alpha*v)*pmt_up[J])<r**2 and np.sum((pmt_mid[J]-Alpha*v)*pmt_up[J])>-r**2 and J in good_chns):
            pmt=np.append(pmt,J)
            t=np.append(t, t2[i])
        i+=1

    Costheta=np.random.uniform(-1,1)
    Phi=np.random.uniform(0,2*np.pi)
    while i<PE_num:
        #print('in an iso', i, 'out of', PE_num)
        costheta=np.random.normal(loc=Costheta, scale=0.25, size=1)[0]
        phi=np.random.normal(loc=Phi, scale=0.5, size=1)[0]
        while not (costheta>-1 and costheta<1):
            costheta=np.random.normal(loc=Costheta, scale=0.25, size=1)
        while not (phi>0 and phi<2*np.pi):
            phi=np.random.normal(loc=Phi, scale=0.5, size=1)
        #v=random.choice([1,-1])*np.array([np.sin(np.arccos(costheta))*np.cos(phi), np.sin(np.arccos(costheta))*np.sin(phi), costheta])
        v=random.choice([1,-1])*np.array([np.sin(np.arccos(costheta))*np.cos(phi), np.sin(np.arccos(costheta))*np.sin(phi), costheta])
        if i%50==0 and strng=='plot':
            ax.quiver(0,0,0,1.5*v[0],1.5*v[1],1.5*v[2], color='r', arrow_length_ratio=0)
        for j in range(len(pmt_mid)):
            dis=pmt_mid[j]-v/np.sum(v*pmt_mid[j])
            if (np.sum(dis*pmt_r[j])/(np.sqrt(np.sum(pmt_r[j]*pmt_r[j])))<r and np.sum(dis*pmt_l[j])/(np.sqrt(np.sum(pmt_l[j]*pmt_l[j])))<r and
                    np.sum(dis*pmt_up[j])/(np.sqrt(np.sum(pmt_up[j]*pmt_up[j])))<r and
                        np.sum(dis*pmt_dn[j])/(np.sqrt(np.sum(pmt_dn[j]*pmt_dn[j])))<r):
                pmt=np.append(pmt,j)
                t=np.append(t, t2[i])
        i+=1
    if strng=='plot':
        plt.title(r'$\theta=$'+'{:3.1f}, '.format(np.arccos(Costheta)/np.pi*180)+r'$\phi=$'+'{:3.1f}'.format(Phi/np.pi*180))
        plt.show()
    return t, pmt



def show_event(event, posible_angles):
    def func(x, t0, tau, T, s):
        y=s/(np.sqrt(2)*tau)+T/(np.sqrt(2)*s)
        t=x-t0
        f=np.exp(-t/tau)*(special.erf(y)-special.erf(y-t/(np.sqrt(2)*s)))/tau
        f[f<0]=0
        return f

    def func2(x, a, t0, tau_s, tau_f, R_s, T, s):
        return a*((1-R_s)*func(x, *[t0, tau_f, T, s])+R_s*func(x, *[t0, tau_s, T, s]))

    t=event.wf[0].T
    PE=event.wf[0].PE
    PEs=np.zeros(len(good_chns))
    PE_var=np.ones(len(PE))
    PEs[event.wf[0].pmt==good_chns]=np.sum(PE)
    for wf in event.wf[1:]:
        PE=PE+wf.PE
        PEs[np.nonzero(good_chns==wf.pmt)[0]]=np.sum(wf.PE)
    fig = plt.figure(figsize=[15,10])
    ax0=fig.add_subplot(4,1,1)
    ax0.plot(t, PE, 'k.', label='{} PEs'.format(np.sum(PE)), markersize=20)
    ax0.errorbar(t, PE, yerr=np.sqrt(PE_var), ls='', elinewidth=6, ecolor='k')
    p, short = Fit_Decay(t, PE, PE_var)
    x=np.linspace(t[0], t[-1], 500)
    ax0.plot(x, func2(x, *p), 'r.', label='tau = {} ns'.format(p[2]), markersize=5)
    ax0.legend()
    show_pmt_hit(fig, 412, event, np.sum(PE))
    ax2=fig.add_subplot(4,1,3)
    ax2.plot(good_chns, PEs, 'ro')
    ax3=fig.add_subplot(4,1,4)
    ax3.plot(posible_angles, event.angles/np.sum(PE), 'ro')
    plt.show()


def show_pmt_hit(fig, cnvs, event, tot_PE):
    ax = fig.add_subplot(cnvs, projection='3d')
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_zlim(-2,2)
    for J in range(20):
        if J in good_chns:
            ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='g')
            ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='g')
            ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='g')
            ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='g')

            ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='g')
            ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='g')
            ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='g')
            ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='g')

            ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='g')
            ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='g')
            ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='g')
            ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='g')
        else:
            ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='k', alpha=0.2)
            ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='k', alpha=0.2)
            ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='k', alpha=0.2)
            ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='k', alpha=0.2)

            ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='k', alpha=0.2)
            ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='k', alpha=0.2)
            ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='k', alpha=0.2)
            ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='k', alpha=0.2)

            ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='k', alpha=0.2)
            ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='k', alpha=0.2)
            ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='k', alpha=0.2)
            ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='k', alpha=0.2)

    for wf in event.wf:
        r=1+np.sum(wf.PE)/tot_PE
        J=wf.pmt
        ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2], r*pmt_mid[J,0], r*pmt_mid[J,1], r*pmt_mid[J,2], color='r')
        

def find_posible_angels(self):
    for i in range(len(good_chns)):
        for j in range(i, len(good_chns)):
            if not int(180/np.pi*np.arccos(np.sum(pmt_mid[i]*pmt_mid[j]))) in self.posible_angles:
                self.posible_angles=np.append(self.posible_angles,int(180/np.pi*np.arccos(np.sum(pmt_mid[i]*pmt_mid[j]))))

def find_angels(event, posible_angles):
    event.angles=np.zeros(len(posible_angles))
    for i in range(len(event.wf)):
        wf=event.wf[i]
        PE=np.sum(wf.PE)
        event.angles[np.argmin(np.abs(posible_angles-0))]+=PE/2*(PE-1)
        for j in range(i+1,len(event.wf)):
            event.angles[np.argmin(np.abs(posible_angles-int(180/np.pi*np.arccos(np.sum(pmt_mid[i]*pmt_mid[j])))))]+=\
                PE*np.sum(event.wf[j].PE)


def ylm(A, event):
    for l in range(int(np.floor(len(A[:,0])/2))):
        for m in range(-l, l+1):
            for i in range(len(event.wf)):
                theta=np.arcsin(pmt_mid[event.wf[i].pmt,2])
                phi=np.arccos(pmt_mid[event.wf[i].pmt,0]/pmt_mid[event.wf[i].pmt,2])
                A[l,int(np.floor(len(A[:,0])/2))+m]+=np.sum(event.wf[i].PE)*sph_harm(l, -m, theta, phi)*(-1)**m


def Dipol(event):
    dipol=0
    PMT=0
    PEs=0
    for wf in event.wf:
        if np.sum(wf.PE)>PEs:
            PMT=wf.pmt
            PEs=np.sum(wf.PE)
    Z=pmt_mid[PMT,:]
    for wf in event.wf:
        dipol+=np.sum(wf.PE)/event.PEs*np.abs(np.sum(pmt_mid[wf.pmt]*Z))
    return dipol, PMT
