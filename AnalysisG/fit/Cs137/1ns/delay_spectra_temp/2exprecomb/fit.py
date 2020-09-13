import numpy as np
import matplotlib.pyplot as plt
import time
import os
from fun import Sim, Sim2, q0_model, make_P, model_area, make_3D, make_spectra
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf
import warnings
from minimize import minimize, make_ps


pmts=[0,1,4,7,8,14]



path='/home/gerak/Desktop/DireXeno/190803/pulser/DelayRecon/'
delay_hs=[]
names=[]
delays=[]
for i in range(len(pmts)-1):
    for j in range(i+1, len(pmts)):
        data=np.load(path+'delay_hist{}-{}.npz'.format(pmts[i], pmts[j]))
        delays.append(data['x']-data['m'])
        delay_hs.append(data['h'])
        names.append('{}_{}'.format(pmts[i], pmts[j]))

T_CsB=1564825612162-1564824285761
T_BG=1564874707904-1564826183355
path='/home/gerak/Desktop/DireXeno/190803/Cs137B/EventRecon/'
data=np.load(path+'H.npz')
H=data['H'][:30,:,:]
G=data['G']
spectrum=data['spectrum']
spectra=data['spectra']
left=data['left']
right=data['right']
t=np.arange(200)
dt=t[1]-t[0]


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

Rec=np.recarray(5000, dtype=[
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


def rec_to_p(rec):
    p=np.array([])
    for name in rec.dtype.names:
        p=np.append(p, np.array(rec[name][0]))
    return p

def p_to_rec(p):
    for i, name in enumerate(rec.dtype.names):
        if np.shape(rec[name][0])==(len(pmts),):
            rec[name][0]=p[i*len(pmts):(i+1)*len(pmts)]
        else:
            if name=='eta':
                rec[name][0]=p[-1]
            elif name=='a':
                rec[name][0]=p[-2]
            elif name=='R':
                rec[name][0]=p[-3]
            elif name=='Ts':
                rec[name][0]=p[-4]
            elif name=='Tf':
                rec[name][0]=p[-5]
            elif name=='F':
                rec[name][0]=p[-6]
            else:
                print('fuck')
                sys.exit()
    return rec

counter=0
PEs=np.arange(len(spectra[:,0]))
GPEs=np.arange(len(spectrum))
l_min=1e10
ls=[]
params=np.zeros(6)
def L(p):
    rec=p_to_rec(p)
    global counter, l_min, params, ls
    counter+=1

    nams=['Q', 'Ts', 'T', 'F', 'Tf', 'Ts', 'R', 'a', 'eta']
    for name in nams:
        if np.any(rec[name]<0):
            Rec[counter-1]=rec[0]
            ls.append(1e10*(1-np.amin(rec[name])))
            np.savez('Rec', Rec=Rec, ls=ls)
            return 1e10*(1-np.amin(rec[name]))
    nams=['F', 'R', 'eta', 'a']
    for name in nams:
        if np.any(rec[name]>1):
            Rec[counter-1]=rec[0]
            ls.append(1e10*(np.amax(rec[name])))
            np.savez('Rec', Rec=Rec, ls=ls)
            return 1e10*(np.amax(rec[name]))
    if rec['Ts'][0]>100:
        Rec[counter-1]=rec[0]
        ls.append(1e10*rec['Ts'][0])
        np.savez('Rec', Rec=Rec, ls=ls)
        return 1e10*rec['Ts'][0]
    if np.any(rec['St'][0]<0.5):
        Rec[counter-1]=rec[0]
        ls.append(1e10*(1+np.abs(np.amin(rec['St'][0]))))
        np.savez('Rec', Rec=Rec, ls=ls)
        return 1e10*(1+np.abs(np.amin(rec['St'][0])))
    if np.any(rec['T'][0]<10):
        Rec[counter-1]=rec[0]
        ls.append(1e10*(10-np.amin(rec['T'][0])))
        np.savez('Rec', Rec=Rec, ls=ls)
        return 1e10*(10-np.amin(rec['T'][0]))


    l=0
    m=make_3D(t, N, rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0], rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0])
    model=np.sum(H[:,0,0])*np.ravel(m)
    data=np.ravel(H[:,:100,:])
    if np.any(model<0):
        Rec[counter-1]=rec[0]
        ls.append(1e10*(1-np.amin(model)))
        np.savez('Rec', Rec=Rec, ls=ls)
        return 1e10*(1-np.amin(model))
    l+=np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

    data=np.ravel(spectra)
    model=np.ravel(np.sum(spectra[:,0])*make_spectra(m, PEs))
    l+=np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

    for i in range(len(pmts)-1):
        for j in range(i+1, len(pmts)):
            x=delays[names=='{}_{}'.format(pmts[i], pmts[j])]
            data=delay_hs[names=='{}_{}'.format(pmts[i], pmts[j])]
            rng=np.nonzero(np.logical_and(x>x[np.argmax(data)]-3, x<x[np.argmax(data)]+3))[0]
            model=(x[1]-x[0])*np.exp(-0.5*(x[rng]-rec['T'][0,j]+rec['T'][0,i])**2/(rec['St'][0,i]**2+rec['St'][0,j]**2))/np.sqrt(2*np.pi*(rec['St'][0,i]**2+rec['St'][0,j]**2))
            model=model/np.amax(model)*np.amax(data)
            data=data[rng]
            l+=np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

    if -l<l_min:
        l_min=-l
        np.savez('best_p', p=p, l_min=l_min)
        print('$$$$$$$$$$$ NEW best p $$$$$$$$$$$$$$$$$$$$')
    if True:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('iteration=', int(counter/(len(p)+1)), 'fanc=',-l)
        print('--------------------------------')
        print(rec)
    # params=np.vstack((params, np.append(p[-5:], -l)))
    Rec[counter-1]=rec[0]
    ls.append(-l)
    np.savez('Rec', Rec=Rec, ls=ls)
    return -l


rec[0]=([0.19657476,  0.13754033,  0.12743771,  0.18797336,  0.17835696,  0.3510241],
 [42.00312603, 42.07819445, 42.03561186, 42.05469875, 42.03181254 ,42.13596326],
  [0.94580769,  0.61208912,  0.84663691,  1.25148529,  0.78060014,  0.59422144],
  0.09440092,  2.06581567, 37.59474049,  0.68417731,  0.454177,    0.19242887)


# p=minimize(L, make_ps(rec_to_p(rec)))
# rec=p_to_rec(p.x)

m=make_3D(t, N, rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0], rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0])
s, GS, GS_spectrum, Sspectra, Gtrp, Gsng, GRtrp, GRsng=Sim2(N, rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0], rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0], PEs)

fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(t, np.sum(H[:,:,i].T*np.arange(np.shape(H)[0]), axis=1)/np.sum(H[:,0,i]), 'ko', label='Data - PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].plot(t[:100], np.sum(m[:,:,i].T*np.arange(np.shape(m)[0]), axis=1), 'r.-', label='2 exp model, {}'.format(np.sum(np.sum(m[:,:,i].T*np.arange(np.shape(m)[0]), axis=1))), linewidth=3)
    np.ravel(ax)[i].plot(t, np.sum(s[:,:,i].T*np.arange(np.shape(s)[0]), axis=1), 'g.-', label='2 exp sim, {}'.format(np.sum(np.sum(s[:,:,i].T*np.arange(np.shape(s)[0]), axis=1))), linewidth=3)
    np.ravel(ax)[i].legend(fontsize=15)
    np.ravel(ax)[i].set_xlabel('Time [ns]', fontsize='15')
fig.text(0.04, 0.5, r'$N_{events}\sum_n nH_{ni}$', va='center', rotation='vertical', fontsize=15)

fig, ax=plt.subplots(2,3)
model=np.sum(spectra[:,0])*make_spectra(m, PEs)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(PEs, spectra[:,i], 'ko', label='spectrum - PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].plot(PEs, model[:,i], 'r-.')
    np.ravel(ax)[i].plot(PEs, np.sum(spectra[:,0])*Sspectra[:,i]/np.sum(Sspectra[:,0]), 'g.-', label='sim')
    np.ravel(ax)[i].legend()

fig, ax=plt.subplots(3,5)
k=0
for i in range(len(pmts)-1):
    for j in range(i+1, len(pmts)):
        x=delays[names=='{}_{}'.format(pmts[i], pmts[j])]
        data=delay_hs[names=='{}_{}'.format(pmts[i], pmts[j])]
        rng=np.nonzero(np.logical_and(x>x[np.argmax(data)]-7, x<x[np.argmax(data)]+7))
        model=(x[1]-x[0])*np.exp(-0.5*(x-rec['T'][0,j]+rec['T'][0,i])**2/(rec['St'][0,i]**2+rec['St'][0,j]**2))/np.sqrt(2*np.pi*(rec['St'][0,i]**2+rec['St'][0,j]**2))
        model=model/np.amax(model)*np.amax(data)
        np.ravel(ax)[k].step(x, data, label='Delays {}_{}'.format(pmts[i], pmts[j]))
        np.ravel(ax)[k].plot(x[rng], model[rng], 'r-.')
        np.ravel(ax)[k].set_xlabel('Delay [ns]', fontsize='15')
        np.ravel(ax)[k].legend(fontsize=15)
        k+=1

x=np.arange(200)
fig, ax1=plt.subplots(1,1)
ax1.plot(x, np.sum(G.T*np.arange(np.shape(G)[0]), axis=1), 'ko', label='Global Data')
ax1.plot(x, np.sum(G[:,0])*np.sum(GS.T*np.arange(np.shape(GS)[0]), axis=1), 'r-.', label='Global 2 exp sim')
ax1.plot(x, np.sum(G[:,0])*np.sum(Gtrp.T*np.arange(np.shape(GS)[0]), axis=1), 'g-.', label='trp')
ax1.plot(x, np.sum(G[:,0])*np.sum(Gsng.T*np.arange(np.shape(GS)[0]), axis=1), 'b-.', label='sng')
ax1.plot(x, np.sum(G[:,0])*np.sum(GRtrp.T*np.arange(np.shape(GS)[0]), axis=1), 'y-.', label='Rtrp')
ax1.plot(x, np.sum(G[:,0])*np.sum(GRsng.T*np.arange(np.shape(GS)[0]), axis=1), 'c-.', label='Rsng')
ax1.legend()

plt.show()
