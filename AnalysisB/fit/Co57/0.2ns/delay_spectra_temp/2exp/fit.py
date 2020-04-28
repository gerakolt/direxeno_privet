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
data=np.load(path+'H.npz')
H=data['H']
G=data['G']

spectra=data['spectra']
PEs=np.arange(len(spectra[0]))
rec=np.recarray(1, dtype=[
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
            rec[name][0]=p[i*len(pmts):(i+1)*len(pmts)]
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

    names=['NQ', 'F', 'Tf', 'Ts', 'T', 'a_spectra']
    for name in names:
        if np.any(rec[name]<0):
            return 1e10*(1-np.amin(rec[name]))
    names=['F']
    for name in names:
        if np.any(rec[name]>1):
            return 1e10*(np.amax(rec[name]))
    if rec['Ts'][0]>100:
        return 1e10*rec['Ts'][0]
    if rec['Ts'][0]<rec['Tf'][0]:
        return 1e10*(rec['Tf'][0]-rec['Ts'][0])
    if np.any(rec['St'][0]<0.5):
        return 1e10*(1+np.abs(np.amin(rec['St'][0])))
    # if np.any(rec['Tf'][0]<1):
    #     return 1e10*(1+np.abs(np.amin(rec['Tf'][0])))

    l=0
    m=Model(rec['NQ'][0], rec['T'][0], np.zeros(len(pmts)), rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0])
    for i in range(len(pmts)):
        model=np.sum(H[:,0,i])*np.ravel(m[:,:500,i])
        if np.any(np.isnan(model)) or np.any(np.isinf(model)):
            print('model is nan or inf')
            print('NQ=', rec['NQ'][0,i], 'T=', rec['T'][0,i], 'F=', rec['F'][0], 'Tf=', rec['Tf'][0], 'Ts=', rec['Ts'][0], 'St=', rec['St'][0, i])
            plt.figure()
            plt.plot(np.mean(t.T*np.arange(np.shape(t)[0])), 'k.')
            plt.show()
            sys.exit()
        data=np.ravel(H[:,:500,i])
        L=len(model)
        for j in range(L):
            if model[j]>0 and data[j]<=0:
                l-=model[j]-data[j]
            elif model[j]<=0 and data[j]>0:
                return 1e10*(data[j]-model[j])
            elif model[j]==0 and data[j]==0:
                l+=1
            else:
                l+=(data[j]*np.log(model[j])-data[j]*np.log(data[j])+data[j]-model[j])

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
                l+=(data[j]*np.log(model[j])-data[j]*np.log(data[j])+data[j]-model[j])

    for i in range(len(pmts)-1):
        for j in range(i+1, len(pmts)):
            x=delays[names=='{}_{}'.format(pmts[i], pmts[j])]
            rng=np.nonzero(np.logical_and(x>x[np.argmax(data)]-5, x<x[np.argmax(data)]+5))[0]
            data=delay_hs[names=='{}_{}'.format(pmts[i], pmts[j])]
            model=(x[1]-x[0])*np.sum(data)*np.exp(-0.5*(x[rng]-rec['T'][0,j]+rec['T'][0,i])**2/(rec['St'][0,i]**2+rec['St'][0,j]**2))/np.sqrt(2*np.pi*(rec['St'][0,i]**2+rec['St'][0,j]**2))
            data=data[rng]
            L=len(model)
            for j in range(L):
                if model[j]>0 and data[j]<=0:
                    l-=model[j]-data[j]
                elif model[j]<=0 and data[j]>0:
                    return 1e10*(data[j]-model[j])
                elif model[j]==0 and data[j]==0:
                    l+=1
                else:
                    l+=(data[j]*np.log(model[j])-data[j]*np.log(data[j])+data[j]-model[j])

    if counter%(len(p)+1)==0:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('iteration=', int(counter/(len(p)+1)), 'fanc=',-l)
        print('--------------------------------')
        print(rec)
    return -l

# rec[0]=([44.10656898, 44.16785627], [2.40187361, 2.05004474], [0.88833944, 1.11088245], [0.04048149, 0.04177063], [5.64966695e-05, 2.44157508e-05], 0.66410346, 31.87824534, 367.91330633, 460.48414159)
rec[0]=([33.20158253, 25.29476026, 20.72751616, 33.59451679, 33.53686614, 54.68329379], [20260.25388504, 29150.92321062, 52534.96147846, 37853.4798459 , 15756.64552532, 26379.20016521],
 [25.56333621, 24.13766518, 23.58918014, 24.51682068, 23.94002462, 22.22005736], [1.31440546, 1.43426818, 1.26274381, 0.54548308, 0.83098988, 1.59084966], 0.10533943, 0.3040308, 33.13975893)
# p=minimize(L, rec_to_p(rec), method='Nelder-Mead', options={'disp':True, 'maxfev':100000})
# rec=p_to_rec(p.x)

m=Model(rec['NQ'][0], rec['T'][0], np.zeros(len(pmts)), rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0])
# s=Sim(rec['NQ'][0], rec['T'][0], 5, np.zeros(len(pmts)), rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0])
s=m
x=np.arange(1000)/5
fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(x, np.sum(H[:,:,i].T*np.arange(np.shape(H)[0]), axis=1), 'ko', label='Data - PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].plot(x, np.sum(H[:,0,i])*np.sum(m[:,:,i].T*np.arange(np.shape(H)[0]), axis=1), 'r.-', label='2 exp model', linewidth=3)
    np.ravel(ax)[i].plot(x, np.sum(H[:,0,i])*np.sum(s[:,:,i].T*np.arange(np.shape(H)[0]), axis=1), 'g.-', label='2 exp model', linewidth=3)
    np.ravel(ax)[i].legend(fontsize=15)
    np.ravel(ax)[i].set_xlabel('Time [ns]', fontsize='15')
fig.text(0.04, 0.5, r'$N_{events}\sum_n nH_{ni}$', va='center', rotation='vertical', fontsize=15)

fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    spectra_rng=np.nonzero(np.logical_and(PEs>PEs[np.argmax(spectra[i])]-10, PEs<PEs[np.argmax(spectra[i])]+10))[0]
    model=rec['a_spectra'][0,i]*poisson.pmf(PEs, np.sum(m[:,:,i].T*np.arange(np.shape(H)[0])))[spectra_rng]
    np.ravel(ax)[i].plot(PEs, spectra[i], 'ko', label='spectrum - PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].plot(PEs[spectra_rng], model, 'r-.')


fig, ax=plt.subplots(3,5)
k=0
for i in range(len(pmts)-1):
    for j in range(i+1, len(pmts)):
        x=delays[names=='{}_{}'.format(pmts[i], pmts[j])]
        data=delay_hs[names=='{}_{}'.format(pmts[i], pmts[j])]
        rng=np.nonzero(np.logical_and(x>x[np.argmax(data)]-5, x<x[np.argmax(data)]+5))
        model=(x[1]-x[0])*np.sum(data)*np.exp(-0.5*(x-rec['T'][0,j]+rec['T'][0,i])**2/(rec['St'][0,i]**2+rec['St'][0,j]**2))/np.sqrt(2*np.pi*(rec['St'][0,i]**2+rec['St'][0,j]**2))
        np.ravel(ax)[k].step(x, data, label='Delays {}_{}'.format(pmts[i], pmts[j]))
        np.ravel(ax)[k].plot(x[rng], model[rng], 'r-.')
        np.ravel(ax)[k].set_xlabel('Delay [ns]', fontsize='15')
        np.ravel(ax)[k].legend(fontsize=15)
        k+=1

x=np.arange(1000)/5
plt.figure()
plt.plot(x, np.sum(G.T*np.arange(np.shape(G)[0]), axis=1), 'ko', label='Global Data')
GS=Global_Sim(rec['NQ'][0], rec['T'][0], 5, np.zeros(len(pmts)), rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0])
plt.plot(x, np.sum(G[:,0])*np.sum(GS.T*np.arange(np.shape(G)[0]), axis=1), 'r-.', label='Global 2 exp sim')

plt.show()
