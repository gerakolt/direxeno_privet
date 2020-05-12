import numpy as np
import matplotlib.pyplot as plt
import time
import os
from fun import Model, Sim, q0_model, make_P, model_area, Sim2
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


path='/home/gerak/Desktop/DireXeno/190803/Cs137/EventRecon/'
data=np.load(path+'H1ns_slow.npz')
H=data['H']
G=data['G']
spectrum=data['spectrum']
spectra=data['spectra']
left=data['left']
right=data['right']
GPEs=data['GPEs']
REC=np.load(path+'recon1ns99992.npz')['rec']

rec=np.recarray(1, dtype=[
    ('NQ', 'f8', len(pmts)),
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
            if name=='Ts':
                rec[name][0]=p[-1]
            elif name=='Tf':
                rec[name][0]=p[-2]
            elif name=='F':
                rec[name][0]=p[-3]
            else:
                print('fuck')
                sys.exit()
    return rec

counter=0
PEs=np.arange(len(spectra[0]))
def L(p):
    rec=p_to_rec(p)
    global counter
    counter+=1

    nams=['NQ', 'Ts', 'T', 'F', 'Tf', 'Ts']
    for name in nams:
        if np.any(rec[name]<0):
            return 1e10*(1-np.amin(rec[name]))
    nams=['F']
    for name in nams:
        if np.any(rec[name]>1):
            return 1e10*(np.amax(rec[name]))
    if rec['Ts'][0]>100:
        return 1e10*rec['Ts'][0]
    if np.any(rec['St'][0]<0.5):
        return 1e10*(1+np.abs(np.amin(rec['St'][0])))


    l=0
    l+=Model(REC['h'][0,:,:], 250, 0.3*np.ones(len(rec[0]['St'])), rec[0]['T'], np.zeros(len(rec[0]['St'])), rec[0]['F'], rec[0]['Tf'], rec[0]['Ts'], rec[0]['St'])
    sys.exit()

    #     spectra_rng=np.nonzero(np.logical_and(PEs>PEs[np.argmax(spectra[i])]-20, PEs<PEs[np.argmax(spectra[i])]+20))[0]
    #     model=poisson.pmf(PEs, np.sum(m[:,:,i].T*np.arange(np.shape(m)[0])))[spectra_rng]
    #     data=spectra[i][spectra_rng]
    #     model=model/np.amax(model)*np.amax(data)
    #     L=len(model)
    #     for j in range(L):
    #         if model[j]>0 and data[j]<=0:
    #             l-=model[j]-data[j]
    #         elif model[j]<=0 and data[j]>0:
    #             return 1e10*(data[j]-model[j])
    #         elif model[j]==0 and data[j]==0:
    #             l+=1
    #         else:
    #             l+=(data[j]*np.log(model[j])-data[j]*np.log(data[j])+data[j]-model[j])
    #
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

rec[0]=([155.23219353, 111.54624717,  99.43339147, 147.80737649, 139.84950503, 277.26826484],
 [25.34443756, 25.59962705, 25.41200686, 25.40401068, 25.60161608, 25.70069353],
 [1.25566649, 0.88644238, 0.98605188, 0.99167018, 1.01187586, 0.99076703], 0.0440187, 0.82417004, 39.00998345)

print(L(rec_to_p(rec)))
# p=minimize(L, rec_to_p(rec), method='Nelder-Mead', options={'disp':True, 'maxfev':100000})
# rec=p_to_rec(p.x)

m=Model(rec['NQ'][0], rec['T'][0],  np.zeros(len(pmts)), rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0])
s, GS, GS_spectrum=Sim(rec['NQ'][0], rec['T'][0], 5,  np.zeros(len(pmts)), rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0])

x=np.arange(200)
fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(x, np.sum(H[:,:,i].T*np.arange(np.shape(H)[0]), axis=1), 'ko', label='Data - PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].plot(x, np.sum(H[:,0,i])*np.sum(m[:,:,i].T*np.arange(np.shape(m)[0]), axis=1), 'r.-', label='2 exp model ({:3.2f} PEs)'.format(np.sum(m[:,:,i].T*np.arange(np.shape(m)[0]))), linewidth=3)
    np.ravel(ax)[i].plot(x, np.sum(H[:,0,i])*np.sum(s[:,:,i].T*np.arange(np.shape(s)[0]), axis=1), 'g.-', label='2 exp sim ({:3.2f} PEs)'.format(np.sum(s[:,:,i].T*np.arange(np.shape(s)[0]))), linewidth=3)
    np.ravel(ax)[i].legend(fontsize=15)
    np.ravel(ax)[i].set_xlabel('Time [ns]', fontsize='15')
fig.text(0.04, 0.5, r'$N_{events}\sum_n nH_{ni}$', va='center', rotation='vertical', fontsize=15)


fig, ax=plt.subplots(2,3)
PEs=np.arange(len(spectra[0]))
for i in range(len(pmts)):
    spectra_rng=np.nonzero(np.logical_and(PEs>PEs[np.argmax(spectra[i])]-20, PEs<PEs[np.argmax(spectra[i])]+20))[0]
    # model=poisson.pmf(PEs, np.sum(m[:,:,i].T*np.arange(np.shape(m)[0])))[spectra_rng]
    # model=model/np.amax(model)*np.amax(spectra[i])
    np.ravel(ax)[i].plot(PEs, spectra[i], 'ko', label='spectrum - PMT{}'.format(pmts[i]))
    # np.ravel(ax)[i].plot(PEs[spectra_rng], model, 'r-.')


fig, ax=plt.subplots(3,5)
k=0
for i in range(len(pmts)-1):
    for j in range(i+1, len(pmts)):
        x=delays[names=='{}_{}'.format(pmts[i], pmts[j])]
        data=delay_hs[names=='{}_{}'.format(pmts[i], pmts[j])]
        rng=np.nonzero(np.logical_and(x>x[np.argmax(data)]-3, x<x[np.argmax(data)]+3))
        model=(x[1]-x[0])*np.exp(-0.5*(x-rec['T'][0,j]+rec['T'][0,i])**2/(rec['St'][0,i]**2+rec['St'][0,j]**2))/np.sqrt(2*np.pi*(rec['St'][0,i]**2+rec['St'][0,j]**2))
        model=model/np.amax(model)*np.amax(data)
        np.ravel(ax)[k].step(x, data, label='Delays {}_{}'.format(pmts[i], pmts[j]))
        np.ravel(ax)[k].plot(x[rng], model[rng], 'r-.')
        np.ravel(ax)[k].set_xlabel('Delay [ns]', fontsize='15')
        np.ravel(ax)[k].legend(fontsize=15)
        k+=1

x=np.arange(200)
fig, (ax1, ax2)=plt.subplots(1,2)
ax1.plot(x, np.sum(G.T*np.arange(np.shape(G)[0]), axis=1), 'ko', label='Global Data')
ax1.plot(x, np.sum(G[:,0])*np.sum(GS.T*np.arange(np.shape(GS)[0]), axis=1), 'r-.', label='Global 2 exp sim')

ax2.step(GPEs, spectrum)
GS_spectrum=GS_spectrum/np.amax(GS_spectrum)*np.amax(spectrum)
ax2.step(GPEs, GS_spectrum)
# model=poisson.pmf(GPEs, np.sum(GS.T*np.arange(np.shape(GS)[0])))
# model=model/np.amax(model)*np.amax(spectrum)
# ax2.step(GPEs, model, 'r-.')

plt.show()
