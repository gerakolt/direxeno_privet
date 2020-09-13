import numpy as np
import matplotlib.pyplot as plt
import time
import os
from fun import Sim, q0_model, make_P, model_area, make_3D
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf
import warnings
from minimize import minimize, make_ps


pmts=[0,1,4,7,8,14]
Datas=['Cs137', 'Cs137B', 'Co57', 'Co57B']


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

H=[]
G=[]
spectrum=[]
spectra=[]

for Data in Datas:
    path='/home/gerak/Desktop/DireXeno/190803/'+Data+'/EventRecon/'
    data=np.load(path+'H.npz')
    H.append(data['H'][:30,:,:])
    G.append(data['G'])
    spectrum.append(data['spectrum'])
    spectra.append(data['spectra'])


t=np.arange(200)
dt=t[1]-t[0]



rec=np.recarray(1, dtype=[
    ('Q', 'f8', len(pmts)),
    ('T', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('N', 'f8', 2),
    ('F', 'f8', 2),
    ('a', 'f8', 2),
    ('eta', 'f8', 2),
    ('Tf', 'f8', 1),
    ('Ts', 'f8', 1),
    ('R', 'f8', 1),
    ])

Rec=np.recarray(5000, dtype=[
    ('Q', 'f8', len(pmts)),
    ('T', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('N', 'f8', 2),
    ('F', 'f8', 2),
    ('a', 'f8', 2),
    ('eta', 'f8', 2),
    ('Tf', 'f8', 1),
    ('Ts', 'f8', 1),
    ('R', 'f8', 1),
    ])


def rec_to_p(rec):
    p=np.array([])
    for name in rec.dtype.names:
        p=np.append(p, np.array(rec[0][name]))
    return p


def p_to_rec(p):
    rec['Q']=p[:len(pmts)]
    rec['T']=p[len(pmts):2*len(pmts)]
    rec['St']=p[2*len(pmts):3*len(pmts)]
    rec['N']=p[3*len(pmts):3*len(pmts)+2]
    rec['F']=p[3*len(pmts)+2:3*len(pmts)+4]
    rec['a']=p[3*len(pmts)+4:3*len(pmts)+6]
    rec['eta']=p[3*len(pmts)+6: 3*len(pmts)+8]
    rec['Tf']=p[3*len(pmts)+8]
    rec['Ts']=p[3*len(pmts)+9]
    rec['R']=p[3*len(pmts)+10]
    return rec

counter=0
l_min=1e10
ls=[]
def L(p):
    rec=p_to_rec(p)
    global counter, l_min, ls
    counter+=1
    print(counter)
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
    if np.any(rec['St'][0]<0.3):
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
    for i in range(len(Datas)):
        m=make_3D(t, rec['N'][0,i//2], rec['F'][0,i//2], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0,i//2], rec['eta'][0,i//2], rec['Q'][0], rec['T'][0], rec['St'][0])
        model=np.sum(H[i][:,0,0])*np.ravel(m)
        data=np.ravel(H[i][:,:100,:])
        if np.any(model<0):
            Rec[counter-1]=rec[0]
            ls.append(1e10*(1-np.amin(model)))
            np.savez('Rec', Rec=Rec, ls=ls)
            return 1e10*(1-np.amin(model))
        l+=np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

        PEs=np.arange(len(spectra[i][:,0]))
        data=np.ravel(spectra[i])
        lmda=np.sum(np.matmul(np.transpose(m, (2,1,0)), np.arange(np.shape(m)[0]).reshape(np.shape(m)[0], 1))[:,:,0], axis=1)
        I=np.arange(len(PEs)*len(lmda))
        model=poisson.pmf(PEs[I//len(lmda)], lmda[I%len(lmda)]).reshape(len(PEs), len(lmda))
        model=np.ravel(model/np.amax(model, axis=0)*np.amax(spectra[i], axis=0))
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


rec[0]=([2.06436156e-01, 1.53739608e-01, 1.35673763e-01, 2.07045270e-01,
 2.03289610e-01, 3.78172789e-01], [4.28603707e+01, 4.27728962e+01,
 4.37043201e+01, 4.36866233e+01, 4.31535963e+01, 4.26430885e+01],
 [7.08836162e-01, 1.10146273e+00, 1.58123658e+00, 1.92190596e+00,
 7.89546120e-01, 8.25660856e-01], [3.89209041e+04, 8.50216818e+03],
 [8.02020474e-02, 1.15605912e-01], [4.25063975e-01, 4.19293224e-01],
 [3.74194703e-01, 2.53987575e-01], 1.87200329e+00, 3.58456460e+01,
 5.29547734e-01)


# p=minimize(L, make_ps(rec_to_p(rec)))
# rec=p_to_rec(p.x)

for j in range(len(Datas)):
    PEs=np.arange(len(spectra[j][:,0]))
    m=make_3D(t, rec['N'][0,j//2], rec['F'][0,j//2], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0,j//2], rec['eta'][0,j//2], rec['Q'][0], rec['T'][0], rec['St'][0])
    s, GS, GS_spectrum, S_spectra, Gtrp, Gsng, GRtrp, GRsng=Sim(rec['N'][0,j//2], rec['F'][0,j//2], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0,j//2], rec['eta'][0,j//2], rec['Q'][0], rec['T'][0], rec['St'][0], PEs)

    fig, ax=plt.subplots(2,3)
    fig.suptitle(Datas[j])
    for i in range(len(pmts)):
        np.ravel(ax)[i].plot(t, np.sum(H[j][:,:,i].T*np.arange(np.shape(H[j])[0]), axis=1)/np.sum(H[j][:,0,i]), 'ko', label='Data - PMT{}'.format(pmts[i]))
        np.ravel(ax)[i].plot(t[:100], np.sum(m[:,:,i].T*np.arange(np.shape(m)[0]), axis=1), 'r.-', label='2 exp model, {}'.format(np.sum(np.sum(m[:,:,i].T*np.arange(np.shape(m)[0]), axis=1))), linewidth=3)
        np.ravel(ax)[i].plot(t, np.sum(s[:,:,i].T*np.arange(np.shape(s)[0]), axis=1), 'g.-', label='2 exp sim, {}'.format(np.sum(np.sum(s[:,:,i].T*np.arange(np.shape(s)[0]), axis=1))), linewidth=3)
        np.ravel(ax)[i].legend(fontsize=15)
        np.ravel(ax)[i].set_xlabel('Time [ns]', fontsize='15')
    fig.text(0.04, 0.5, r'$N_{events}\sum_n nH_{ni}$', va='center', rotation='vertical', fontsize=15)

    fig, ax=plt.subplots(2,3)
    fig.suptitle(Datas[j])
    lmda=np.sum(np.matmul(np.transpose(m, (2,1,0)), np.arange(np.shape(m)[0]).reshape(np.shape(m)[0], 1))[:,:,0], axis=1)
    I=np.arange(len(PEs)*len(lmda))
    model=poisson.pmf(PEs[I//len(lmda)], lmda[I%len(lmda)]).reshape(len(PEs), len(lmda))
    model=model/np.amax(model, axis=0)*np.amax(spectra[j], axis=0)
    S_spectra=S_spectra/np.amax(S_spectra, axis=0)*np.amax(spectra[j], axis=0)
    for i in range(len(pmts)):
        np.ravel(ax)[i].plot(PEs, spectra[j][:,i], 'ko', label='spectrum - PMT{}'.format(pmts[i]))
        np.ravel(ax)[i].plot(PEs, model[:,i], 'r-.')
        np.ravel(ax)[i].plot(PEs, S_spectra[:,i], 'g-.')
        np.ravel(ax)[i].legend()


    x=np.arange(200)
    fig, ax1=plt.subplots(1,1)
    fig.suptitle(Datas[j])
    ax1.plot(x, np.sum(G[j].T*np.arange(np.shape(G[j])[0]), axis=1), 'ko', label='Global Data')
    ax1.plot(x, np.sum(G[j][:,0])*np.sum(GS.T*np.arange(np.shape(GS)[0]), axis=1), 'r-.', label='Global 2 exp sim')
    ax1.plot(x, np.sum(G[j][:,0])*np.sum(Gtrp.T*np.arange(np.shape(GS)[0]), axis=1), 'g-.', label='trp')
    ax1.plot(x, np.sum(G[j][:,0])*np.sum(Gsng.T*np.arange(np.shape(GS)[0]), axis=1), 'b-.', label='sng')
    ax1.plot(x, np.sum(G[j][:,0])*np.sum(GRtrp.T*np.arange(np.shape(GS)[0]), axis=1), 'y-.', label='Rtrp')
    ax1.plot(x, np.sum(G[j][:,0])*np.sum(GRsng.T*np.arange(np.shape(GS)[0]), axis=1), 'c-.', label='Rsng')
    ax1.legend()

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
plt.show()
