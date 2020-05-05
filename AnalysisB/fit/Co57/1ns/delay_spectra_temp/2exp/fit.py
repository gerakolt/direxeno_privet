import numpy as np
import matplotlib.pyplot as plt
import time
import os
from fun import Model, Sim, q0_model, make_P, model_area, Global_Sim, subGlobal_Sim
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf
import warnings

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


path='/home/gerak/Desktop/DireXeno/190803/Co57/EventRecon/'
data=np.load(path+'H1ns.npz')
H=data['H']
G=data['G']


spectra=data['spectra']
PEs=np.arange(len(spectra[0]))
rec=np.recarray(1, dtype=[
    ('a_delay', 'f8', len(names)),
    ('NQ', 'f8', len(pmts)),
    ('a_spectra', 'f8', len(pmts)),
    ('T', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('F', 'f8', 1),
    ('Tf', 'f8', 1),
    ('Ts', 'f8', 1),
    ])


def rec_to_p(rec):
    p=np.array([])
    for name in rec.dtype.names:
        p=np.append(p, np.array(rec[name][0]))
    return p

def p_to_rec(p):
    for i, name in enumerate(rec.dtype.names):
        if np.shape(rec[name][0])==(len(pmts),):
            rec[name][0]=p[len(names)+(i-1)*len(pmts):len(names)+i*len(pmts)]
        elif np.shape(rec[name][0])==(len(names),):
            rec[name][0]=p[:len(names)]
        else:
            if name=='F':
                rec[name][0]=p[-3]
            elif name=='Tf':
                rec[name][0]=p[-2]
            elif name=='Ts':
                rec[name][0]=p[-1]
            else:
                print('fuck')
                sys.exit()
    return rec

counter=0
def L(p):
    rec=p_to_rec(p)
    global counter
    counter+=1

    nams=['NQ', 'F', 'Tf', 'Ts', 'T', 'a_spectra', 'a_delay']
    for name in nams:
        if np.any(rec[name]<0):
            return 1e10*(1-np.amin(rec[name]))
    nams=['F']
    for name in nams:
        if np.any(rec[name]>1):
            return 1e10*(np.amax(rec[name]))
    if rec['Ts'][0]>100:
        return 1e10*rec['Ts'][0]
    if rec['Ts'][0]<rec['Tf'][0]:
        return 1e10*(rec['Tf'][0]-rec['Ts'][0])
    if np.any(rec['St'][0]<0.2):
        return 1e10*(1+np.abs(np.amin(rec['St'][0])))
    if np.any(2*rec['Tf'][0]<rec['St'][0]):
        return 1e10*(1+np.amax(rec['St'][0])-2*rec['Tf'][0])
    # if np.any(rec['Tf'][0]<1):
    #     return 1e10*(1+np.abs(np.amin(rec['Tf'][0])))

    l=0
    m=Model(rec['NQ'][0], rec['T'][0], np.zeros(len(pmts)), rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0])
    for i in range(len(pmts)):
        model=np.sum(H[:,0,i])*np.ravel(m[:,:200,i])
        if np.any(np.isnan(model)) or np.any(np.isinf(model)):
            print('model is nan or inf')
            print('NQ=', rec['NQ'][0,i], 'T=', rec['T'][0,i], 'F=', rec['F'], 'Tf=', rec['Tf'], 'Ts=', rec['Ts'], 'St=', rec['St'][0, i])
        data=np.ravel(H[:,:200,i])
        L=len(model)
        for j in range(L):
            if model[j]>0 and data[j]<=0:
                l-=model[j]-data[j]
            elif model[j]<=0 and data[j]>0:
                print('NQ=', rec['NQ'][0,i], 'T=', rec['T'][0,i], 'F=', rec['F'], 'Tf=', rec['Tf'], 'Ts=', rec['Ts'], 'St=', rec['St'][0, i])
                return 1e10*(data[j]-model[j])
            elif model[j]==0 and data[j]==0:
                l+=1
            else:
                l+=data[j]*np.log(model[j])-data[j]*np.log(data[j])+data[j]-model[j]

        spectra_rng=np.nonzero(np.logical_and(PEs>PEs[np.argmax(spectra[i])]-10, PEs<PEs[np.argmax(spectra[i])]+10))[0]
        model=rec['a_spectra'][0,i]*poisson.pmf(PEs, np.sum(m[:,:,i].T*np.arange(np.shape(H)[0])))[spectra_rng]
        data=spectra[i][spectra_rng]
        L=len(model)
        for j in range(L):
            if model[j]>0 and data[j]<=0:
                l-=model[j]-data[j]
            elif model[j]<=0 and data[j]>0:
                return 1e10*(data[j]-model[j])
            elif model[j]==0 and data[j]==0:
                l+=1
            else:
                l+=data[j]*np.log(model[j])-data[j]*np.log(data[j])+data[j]-model[j]
    for i in range(len(pmts)-1):
        for j in range(i+1, len(pmts)):
            k=np.nonzero(np.array(names)=='{}_{}'.format(pmts[i], pmts[j]))[0][0]
            x=delays[k]
            delay_h=delay_hs[k]
            rng=np.nonzero(np.logical_and(x>x[np.argmax(delay_h)]-3, x<x[np.argmax(delay_h)]+3))[0]
            data=delay_h[rng]
            model=(x[1]-x[0])*rec['a_delay'][0,k]*np.exp(-0.5*(x-rec['T'][0,j]+rec['T'][0,i])**2/(rec['St'][0,i]**2+rec['St'][0,j]**2))[rng]/np.sqrt(2*np.pi*(rec['St'][0,i]**2+rec['St'][0,j]**2))
            for j in range(len(data)):
                if model[j]>0 and data[j]<=0:
                    l-=model[j]-data[j]
                elif model[j]<=0 and data[j]>0:
                    return 1e10*(data[j]-model[j])
                elif model[j]==0 and data[j]==0:
                    l+=1
                else:
                    l+=data[j]*np.log(model[j])-data[j]*np.log(data[j])+data[j]-model[j]

    if counter%(len(p)+1)==0:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('iteration=', int(counter/(len(p)+1)), 'fanc=',-l)
        print('--------------------------------')
        print(rec)
    return -l

rec[0]=([2080.21356586, 2501.81594547, 1889.82534701, 1847.95314879, 2574.33120707, 1790.25752064, 2263.97366442, 1575.56527258, 1921.97402859,  675.94204246, 2244.94497804, 2077.02296211, 2238.32536362, 2225.42073384, 1875.28136122],
 [36.25080606, 26.91902805, 21.14997576, 34.91074107, 35.06067081, 54.93729789], [13727.82272613, 14677.75142414, 17792.69565042, 16818.45467068, 16171.46398967, 10809.11245454],
  [20.3130921 , 20.34259746, 20.26870905, 20.03458108, 20.13676493, 20.42784198], [0.82365565, 0.97906695, 1.03098748, 0.23454212, 0.66751749, 0.9632058 ], 0.06911841, 1.50760787, 35.01487077)
print(L(rec_to_p(rec)))
# p=minimize(L, rec_to_p(rec), method='Nelder-Mead', options={'disp':True, 'maxfev':100000})
# rec=p_to_rec(p.x)
m=Model(rec['NQ'][0], rec['T'][0], np.zeros(len(pmts)), rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0])
s=Sim(rec['NQ'][0], rec['T'][0], 0.1, np.zeros(len(pmts)), rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0])

x=np.arange(200)
fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(x, np.sum(H[:,:,i].T*np.arange(np.shape(H)[0]), axis=1), 'ko', label='Data - PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].plot(x, np.sum(H[:,0,i])*np.sum(m[:,:,i].T*np.arange(np.shape(H)[0]), axis=1), 'r.-', label='2 exp model', linewidth=3)
    np.ravel(ax)[i].plot(x, np.sum(H[:,0,i])*np.sum(s[:,:,i].T*np.arange(np.shape(H)[0]), axis=1), 'g.-', label='2 exp sim', linewidth=3)
    np.ravel(ax)[i].legend(fontsize=15)
    np.ravel(ax)[i].set_xlabel('Time [ns]', fontsize='15')
fig.text(0.04, 0.5, r'$N_{events}\sum_n nH_{ni}$', va='center', rotation='vertical', fontsize=15)
#
fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    spectra_rng=np.nonzero(np.logical_and(PEs>PEs[np.argmax(spectra[i])]-10, PEs<PEs[np.argmax(spectra[i])]+10))[0]
    model=rec['a_spectra'][0,i]*poisson.pmf(PEs, np.sum(m[:,:,i].T*np.arange(np.shape(H)[0])))[spectra_rng]
    np.ravel(ax)[i].plot(PEs, spectra[i], 'ko', label='spectrum - PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].plot(PEs[spectra_rng], model, 'r-.')


fig, ax=plt.subplots(3,5)
c=0
for i in range(len(pmts)-1):
    for j in range(i+1, len(pmts)):
        k=np.nonzero(np.array(names)=='{}_{}'.format(pmts[i], pmts[j]))[0][0]
        x=delays[k]
        delay_h=delay_hs[k]
        rng=np.nonzero(np.logical_and(x>x[np.argmax(delay_h)]-3, x<x[np.argmax(delay_h)]+3))[0]
        data=delay_h
        model=(x[1]-x[0])*rec['a_delay'][0,k]*np.exp(-0.5*(x-rec['T'][0,j]+rec['T'][0,i])**2/(rec['St'][0,i]**2+rec['St'][0,j]**2))[rng]/np.sqrt(2*np.pi*(rec['St'][0,i]**2+rec['St'][0,j]**2))

        np.ravel(ax)[c].step(x, data, label='Delays {}_{}'.format(pmts[i], pmts[j]))
        np.ravel(ax)[c].plot(x[rng], model, 'r-.')
        np.ravel(ax)[c].set_xlabel('Delay [ns]', fontsize='15')
        np.ravel(ax)[c].legend(fontsize=15)
        c+=1


x=np.arange(200)
plt.figure()
plt.plot(x, np.sum(G.T*np.arange(np.shape(G)[0]), axis=1), 'ko', label='Global Data')
GS=Global_Sim(rec['NQ'][0], rec['T'][0], 5, np.zeros(len(pmts)), rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0])
f=subGlobal_Sim(rec['NQ'][0], rec['T'][0], 5, np.zeros(len(pmts)), rec['F'][0], 0, rec['Tf'][0], rec['Ts'][0], rec['St'][0])
s=subGlobal_Sim(rec['NQ'][0], rec['T'][0], 5, np.zeros(len(pmts)), 0, 1-rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0])

plt.plot(x, np.sum(G[:,0])*np.sum(GS.T*np.arange(np.shape(G)[0]), axis=1), 'r-.', label='Global 2 exp sim')
plt.plot(x, np.sum(G[:,0])*np.sum(f.T*np.arange(np.shape(G)[0]), axis=1), 'g-.', label='Fast')
plt.plot(x, np.sum(G[:,0])*np.sum(s.T*np.arange(np.shape(G)[0]), axis=1), 'b-.', label='Slow')
plt.show()
