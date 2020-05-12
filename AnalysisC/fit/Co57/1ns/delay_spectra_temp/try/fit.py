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
spectrum=data['spectrum']
spectra=data['spectra']
left=data['left']
right=data['right']
PEs=np.arange(len(spectra[0]))
rec=np.recarray(1, dtype=[
    ('NQ', 'f8', len(pmts)),
    ('T', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('R', 'f8', len(pmts)),
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
            if name=='Ts':
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

    nams=['NQ', 'Ts', 'T', 'R']
    for name in nams:
        if np.any(rec[name]<0):
            return 1e10*(1-np.amin(rec[name]))
    nams=['R']
    for name in nams:
        if np.any(rec[name]>1):
            return 1e10*(np.amax(rec[name]))
    if rec['Ts'][0]>100:
        return 1e10*rec['Ts'][0]
    if np.any(rec['St'][0]<0.5):
        return 1e10*(1+np.abs(np.amin(rec['St'][0])))
    # if np.any(rec['Tf'][0]<1):
    #     return 1e10*(1+np.abs(np.amin(rec['Tf'][0])))

    l=0
    m=Model(rec['NQ'][0], rec['T'][0],  rec['R'][0], 0, 1000, rec['Ts'][0], rec['St'][0])
    for i in range(len(pmts)):
        model=np.sum(H[:,0,i])*np.ravel(m[:,:100,i])
        if np.any(np.isnan(model)) or np.any(np.isinf(model)):
            print('model is nan or inf')
            print('NQ=', rec['NQ'][0,i], 'T=', rec['T'][0,i], 'R=', rec['R'][0,i], 'Ts=', rec['Ts'][0], 'St=', rec['St'][0, i])
        data=np.ravel(H[:,:100,i])
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
        model=poisson.pmf(PEs, np.sum(m[:,:,i].T*np.arange(np.shape(H)[0])))[spectra_rng]
        data=spectra[i][spectra_rng]
        model=model/np.amax(model)*np.amax(data)
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
            rng=np.nonzero(np.logical_and(x>x[np.argmax(data)]-3, x<x[np.argmax(data)]+3))[0]
            data=delay_hs[names=='{}_{}'.format(pmts[i], pmts[j])]
            model=(x[1]-x[0])*np.exp(-0.5*(x[rng]-rec['T'][0,j]+rec['T'][0,i])**2/(rec['St'][0,i]**2+rec['St'][0,j]**2))/np.sqrt(2*np.pi*(rec['St'][0,i]**2+rec['St'][0,j]**2))
            model=model/np.amax(model)*np.amax(data)
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

rec[0]=([35.49302268, 26.17497766, 21.34892126, 34.11857516, 34.25089177, 55.35067421], [24.18066808, 24.14013932, 24.37615609, 24.31612566, 24.51001521, 24.57748248], [1.0884017 , 0.88510012, 0.99856744, 1.05147662, 1.09080665, 1.14670428],
 [0.06125483, 0.05278206, 0.0909597 , 0.09726535, 0.09490963, 0.08954461], 32.79469072)

# print(L(rec_to_p(rec)))
# p=minimize(L, rec_to_p(rec), method='Nelder-Mead', options={'disp':True, 'maxfev':100000})
# rec=p_to_rec(p.x)

m=Model(rec['NQ'][0], rec['T'][0],  rec['R'][0], 0, 1000, rec['Ts'][0], rec['St'][0])
s, GS=Sim(rec['NQ'][0], rec['T'][0], 1e-6, rec['R'][0], 0, 1000, rec['Ts'][0], rec['St'][0])


x=np.arange(200)
fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(x, np.sum(H[:,0,i])*np.sum(m[:,:,i].T*np.arange(np.shape(m)[0]), axis=1), 'r.-', label='2 exp model ({:3.2f} PEs)'.format(np.sum(m[:,:,i].T*np.arange(np.shape(m)[0]))), linewidth=3)
    np.ravel(ax)[i].plot(x, np.sum(H[:,0,i])*np.sum(s[:,:,i].T*np.arange(np.shape(s)[0]), axis=1), 'g.-', label='2 exp sim ({:3.2f} PEs)'.format(np.sum(s[:,:,i].T*np.arange(np.shape(s)[0]))), linewidth=3)
    np.ravel(ax)[i].legend(fontsize=15)
    np.ravel(ax)[i].set_xlabel('Time [ns]', fontsize='15')
fig.text(0.04, 0.5, r'$N_{events}\sum_n nH_{ni}$', va='center', rotation='vertical', fontsize=15)

fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(x, np.sum(H[:,0,i])*np.sum(m[:,:,i].T*np.arange(np.shape(m)[0]), axis=1)-np.sum(H[:,0,i])*np.sum(s[:,:,i].T*np.arange(np.shape(s)[0]), axis=1),
        'r.-', label='2 exp model ({:3.2f} PEs)'.format(np.sum(m[:,:,i].T*np.arange(np.shape(m)[0]))), linewidth=3)
    np.ravel(ax)[i].legend(fontsize=15)
    np.ravel(ax)[i].set_xlabel('Time [ns]', fontsize='15')

plt.show()
