from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf
from scipy.special import comb


def Sim(NQ, T, Strig, R, F, Tf, Ts, St, q0, a0, Spad, Spe, m_pad):
    N_events=10000
    d=np.zeros((N_events, 200, len(NQ)))
    H=np.zeros((15, 200, len(NQ)))
    for i in range(N_events):
        print(i)
        t0=np.zeros(len(NQ))
        trig=np.random.normal(0, Strig, 1)
        for j in range(len(NQ)):
            n=np.random.poisson(NQ[j])
            ch0=np.random.choice(15, size=200, replace=True, p=np.append(1-q0[j]/(1-q0[j]), q0[j]**np.arange(1, 15)))
            ch=np.random.choice(3, size=n, replace=True, p=[R[j], (1-R[j])*F, (1-R[j])*(1-F)])
            nd=len(np.nonzero(ch==0)[0])
            nf=len(np.nonzero(ch==1)[0])
            ns=len(np.nonzero(ch==2)[0])
            td=np.random.normal(trig+T[j], St[j], nd)
            tf=np.random.normal(trig+T[j]+np.random.exponential(Tf, nf), St[j], nf)
            ts=np.random.normal(trig+T[j]+np.random.exponential(Ts, ns), St[j], ns)
            t=np.append(td, np.append(tf, ts))
            h, bins=np.histogram(t, bins=np.arange(201)-0.5)
            for k in np.nonzero(h>0)[0]:
                a=np.random.normal(h[k]+m_pad[j], np.sqrt(Spad[j]**2+h[k]*Spe[j]**2))
                if a<a0[j]:
                    h[k]=0
                elif a>a0[j] and a<1.5:
                    h[k]=1
                else:
                    h[k]=int(np.round(a))
            h+=ch0
            t0[j]=np.amin(np.nonzero(h>0)[0])
            d[i,:,j]=h
        for j in range(len(NQ)):
            d[i,:,j]=np.roll(d[i,:,j], -int(np.amin(t0)))
    for j in range(len(NQ)):
        for k in range(200):
            H[:,k,j]=np.histogram(d[:,k,j], bins=np.arange(16)-0.5)[0]
    return H/N_events


def Sim1(NQ, T, Strig, R, F, Tf, Ts, St, q0, a0, Spad, Spe, m_pad):
    N_events=10000
    d=np.zeros((N_events, 1000, len(NQ)))
    H=np.zeros((15, 1000, len(NQ)))
    G=np.zeros((15, 1000))
    for i in range(N_events):
        t0=np.zeros(len(NQ))
        trig=np.random.normal(0, Strig*5, 1)
        for j in range(len(NQ)):
            n=np.random.poisson(NQ[j])
            # ch0=np.random.choice(15, size=200, replace=True, p=np.append(1-q0[j]/(1-q0[j]), q0[j]**np.arange(1, 15)))
            ch=np.random.choice(3, size=n, replace=True, p=[R[j], (1-R[j])*F, (1-R[j])*(1-F)])
            nd=len(np.nonzero(ch==0)[0])
            nf=len(np.nonzero(ch==1)[0])
            ns=len(np.nonzero(ch==2)[0])
            td=np.random.normal(trig+5*T[j], 5*St[j], nd)
            tf=np.random.normal(trig+5*T[j]+np.random.exponential(5*Tf, nf), 5*St[j], nf)
            ts=np.random.normal(trig+5*T[j]+np.random.exponential(5*Ts, ns), 5*St[j], ns)
            t=np.append(td, np.append(tf, ts))
            h, bins=np.histogram(t, bins=np.arange(1001)-0.5)
            # for k in np.nonzero(h>0)[0]:
            #     a=np.random.normal(h[k]+m_pad[j], np.sqrt(Spad[j]**2+h[k]*Spe[j]**2))
            #     if a<a0[j]:
            #         h[k]=0
            #     elif a>a0[j] and a<1.5:
            #         h[k]=1
            #     else:
            #         h[k]=int(np.round(a))
            # h+=ch0
            t0[j]=np.amin(np.nonzero(h>0)[0])
            d[i,:,j]=h
        for j in range(len(NQ)):
            d[i,:,j]=np.roll(d[i,:,j], -int(np.amin(t0)))
    for j in range(len(NQ)):
        for k in range(1000):
            H[:,k,j]=np.histogram(d[:,k,j], bins=np.arange(16)-0.5)[0]
    for k in range(1000):
        G[:,k]=np.histogram(np.sum(d[:,k,:], axis=1), bins=np.arange(16)-0.5)[0]
    return H/N_events, G/N_events

def Sim2(NQ, T, Strig, R, F, Tf, Ts, St, q0, a0, Spad, Spe, m_pad):
    N_events=10000
    d=np.zeros((N_events, 1000, len(NQ)))
    H=np.zeros((15, 1000, len(NQ)))
    G=np.zeros((15, 1000))
    for i in range(N_events):
        t0=np.zeros(len(NQ))
        trig=np.random.normal(0, Strig*5, 1)
        for j in range(len(NQ)):
            n=np.random.poisson(NQ[j])
            # ch0=np.random.choice(15, size=200, replace=True, p=np.append(1-q0[j]/(1-q0[j]), q0[j]**np.arange(1, 15)))
            ch=np.random.choice(3, size=n, replace=True, p=[R[j], (1-R[j])*F, (1-R[j])*(1-F)])
            nd=len(np.nonzero(ch==0)[0])
            nf=len(np.nonzero(ch==1)[0])
            ns=len(np.nonzero(ch==2)[0])
            td=np.random.normal(trig+5*T[j], 5*St[j], nd)
            tf=np.random.normal(trig+5*T[j]+np.random.exponential(5*Tf, nf), 5*St[j], nf)
            ts=np.random.normal(trig+5*T[j]+np.random.exponential(5*Ts, ns), 5*St[j], ns)
            t=np.append(td, np.append(tf, ts))
            h, bins=np.histogram(t, bins=np.arange(1001)-0.5)
            # for k in np.nonzero(h>0)[0]:
            #     a=np.random.normal(h[k]+m_pad[j], np.sqrt(Spad[j]**2+h[k]*Spe[j]**2))
            #     if a<a0[j]:
            #         h[k]=0
            #     elif a>a0[j] and a<1.5:
            #         h[k]=1
            #     else:
            #         h[k]=int(np.round(a))
            # h+=ch0
            t0[j]=np.amin(np.nonzero(h>0)[0])
            d[i,:,j]=h
        # for j in range(len(NQ)):
        #     d[i,:,j]=np.roll(d[i,:,j], -int(np.amin(t0)))
    for j in range(len(NQ)):
        for k in range(1000):
            H[:,k,j]=np.histogram(d[:,k,j], bins=np.arange(16)-0.5)[0]
    for k in range(1000):
        G[:,k]=np.histogram(np.sum(d[:,k,:], axis=1), bins=np.arange(16)-0.5)[0]
    return H/N_events, G/N_events

pmts=[0,1]
rec=np.recarray(1, dtype=[
    ('NQ', 'f8', len(pmts)),
    ('T', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('R', 'f8', len(pmts)),
    ('q0', 'f8', len(pmts)),
    ('a0', 'f8', len(pmts)),
    ('Spad', 'f8', len(pmts)),
    ('Spe', 'f8', len(pmts)),
    ('a_pad', 'f8', len(pmts)),
    ('a_spe', 'f8', len(pmts)),
    ('a_dpe', 'f8', len(pmts)),
    ('a_trpe', 'f8', len(pmts)),
    ('m_pad', 'f8', len(pmts)),
    ('a_spec', 'f8', len(pmts)),
    ('F', 'f8', 1),
    ('Tf', 'f8', 1),
    ('Ts', 'f8', 1),
    ('a_delay', 'f8', 1),
    ])

rec[0]=([34.3884471 , 34.55879039], [37.94533241, 38.0146521 ], [0.38741811, 0.57078376], [5.32071203e-05, 1.61793834e-02], [5.46801032e-05, 2.34126233e-05], [5.64455335e-05, 4.18362118e-11], [0.2886672 , 0.25148966],
 [7.53253959e-08, 8.59598191e-14], [77211.6489643 , 79512.73384661], [13446.83345397, 11839.49233702], [1398.52958763, 1152.94084535], [69.44435858,  2.98888058], [-0.03808736, -0.02661984], [34447.41818585, 36784.14944601],
  0.04588877, 2.45177921, 39.37174056, 367.56308697)



x=np.arange(1000)
s1, S1=Sim1(rec['NQ'][0], rec['T'][0], 5, rec['R'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0], rec['q0'][0], rec['a0'][0], rec['Spad'][0], rec['Spe'][0], rec['m_pad'][0])
s2, S2=Sim2(rec['NQ'][0], rec['T'][0], 5, rec['R'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0], rec['q0'][0], rec['a0'][0], rec['Spad'][0], rec['Spe'][0], rec['m_pad'][0])

fig, (ax1, ax2)=plt.subplots(1,2, sharey=False)
ax2.plot(x[:100], np.sum(s1[:,:100,0].T*np.arange(np.shape(s1)[0]), axis=1), 'k.-', linewidth=7, label='PMT0')
ax2.plot(x[:100], np.sum(s1[:,:100,1].T*np.arange(np.shape(s1)[0]), axis=1), 'r.-', linewidth=7, label='PMT1')

ax1.plot(x, np.sum(s2[:,:,0].T*np.arange(np.shape(s2)[0]), axis=1), 'k.-', linewidth=7, label='PMT0')
ax1.plot(x, np.sum(s2[:,:,1].T*np.arange(np.shape(s2)[0]), axis=1), 'r.-', linewidth=7, label='PMT1')

# ax1.title.set_text('Trigger Aligned', fontdict={'fontsize': 25})
# ax2.title.set_text('First PE in Event Aligned', fontdict={'fontsize': 25})

ax1.set_title('Trigger Aligned', fontdict={'fontsize': 25, 'fontweight': 'medium'})
ax2.set_title('First PE in Event Aligned', fontdict={'fontsize': 25, 'fontweight': 'medium'})


ax1.legend(fontsize=15)
ax2.legend(fontsize=15)
ax1.tick_params(axis='both', which='major', labelsize=22)
ax2.tick_params(axis='both', which='major', labelsize=22)
fig.text(0.5, 0.04, 'Time [ns]', ha='center', fontsize=25)
fig.text(0.04, 0.5, r'$\sum_n nD_{ni}$', va='center', rotation='vertical', fontsize=25)

plt.show()
