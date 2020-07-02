import numpy as np
import matplotlib.pyplot as plt
import time
import os
from fun import Sim, q0_model, make_P, model_area, Sim2, make_3D
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

T_CsB=1564825612162-1564824285761
T_BG=1564874707904-1564826183355
path='/home/gerak/Desktop/DireXeno/190803/Cs137B/EventRecon/'
data=np.load(path+'H.npz')
H=data['H']
G=data['G']
spectrum=data['spectrum']
spectra=data['spectra']
left=data['left']
right=data['right']
t=np.arange(200)
dt=t[1]-t[0]

path='/home/gerak/Desktop/DireXeno/190803/BG/EventRecon/'
data=np.load(path+'H.npz')
H_BG=data['H']*T_CsB/T_BG
G_BG=data['G']*T_CsB/T_BG
spectrum_BG=data['spectrum']*T_CsB/T_BG
spectra_BG=data['spectra']*T_CsB/T_BG


N=60*662
rec=np.recarray(1, dtype=[
    ('Q', 'f8', len(pmts)),
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
PEs=np.arange(len(spectra[:,0]))
GPEs=np.arange(len(spectrum))
l_min=1e10
def L(p):
    rec=p_to_rec(p)
    global counter, l_min
    counter+=1

    nams=['Q', 'Ts', 'T', 'F', 'Tf', 'Ts']
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
    if np.any(rec['T'][0]<10):
        return 1e10*(10-np.amin(rec['T'][0]))


    l=0
    m=make_3D(t, dt, N, np.zeros(len(pmts)), rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['Q'][0], rec['T'][0], rec['St'][0])
    model=np.sum(H[:,0,0])*np.ravel(m)
    data=np.ravel(H[:,:100,:])
    if np.any(model<0):
        print('Model<0')
        sys.exit()
    l+=np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

    data=np.ravel(spectra)
    lmda=np.sum(np.matmul(np.transpose(m, (2,1,0)), np.arange(np.shape(m)[0]).reshape(np.shape(m)[0], 1))[:,:,0], axis=1)
    I=np.arange(len(PEs)*len(lmda))
    model=poisson.pmf(PEs[I//len(lmda)], lmda[I%len(lmda)]).reshape(len(PEs), len(lmda))
    model=np.ravel(model/np.amax(model, axis=0)*np.amax(spectra, axis=0))
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
    return -l


rec[0]=([1.52955643e-01, 1.21685760e-01, 1.12773659e-01, 1.57756046e-01,
 1.49101133e-01, 3.07357999e-01], [4.21813089e+01, 4.23137955e+01,
 4.23327547e+01, 4.21586313e+01, 4.23591370e+01, 4.22545259e+01],
 [9.79864208e-01, 8.01769269e-01, 8.52056472e-01, 9.65873999e-01,
 5.71299405e-01, 8.97126437e-01], 1.70502766e-02, 5.39283739e+00, 4.70872930e+01)
# print(L(rec_to_p(rec)))
# p=minimize(L, rec_to_p(rec), method='Nelder-Mead', options={'disp':True, 'maxfev':100000})
# rec=p_to_rec(p.x)

m=make_3D(t, dt, N, np.zeros(len(pmts)), rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['Q'][0], rec['T'][0], rec['St'][0])
# s, GS, GS_spectrum=Sim2(rec['N'][0], rec['Q'][0], rec['T'][0], 5,  rec['R'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0])
s=m


fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(t, np.sum(H[:,:,i].T*np.arange(np.shape(H)[0]), axis=1), 'ko', label='Data - PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].plot(t[:100], np.sum(H[:,0,i])*np.sum(m[:,:,i].T*np.arange(np.shape(m)[0]), axis=1), 'r.-', label='2 exp model', linewidth=3)
    # np.ravel(ax)[i].plot(t, np.sum(H[:,0,i])*np.sum(s[:,:,i].T*np.arange(np.shape(s)[0]), axis=1), 'g.-', label='2 exp sim', linewidth=3)
    np.ravel(ax)[i].legend(fontsize=15)
    np.ravel(ax)[i].set_xlabel('Time [ns]', fontsize='15')
fig.text(0.04, 0.5, r'$N_{events}\sum_n nH_{ni}$', va='center', rotation='vertical', fontsize=15)


fig, ax=plt.subplots(2,3)
lmda=np.sum(np.matmul(np.transpose(m, (2,1,0)), np.arange(np.shape(m)[0]).reshape(np.shape(m)[0], 1))[:,:,0], axis=1)
I=np.arange(len(PEs)*len(lmda))
model=poisson.pmf(PEs[I//len(lmda)], lmda[I%len(lmda)]).reshape(len(PEs), len(lmda))
model=model/np.amax(model, axis=0)*np.amax(spectra, axis=0)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(PEs, spectra[:,i], 'ko', label='spectrum - PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].step(PEs, spectra_BG[:,i], label='BG'.format(pmts[i]))
    np.ravel(ax)[i].plot(PEs, model[:,i], 'r-.')
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
plt.show()
x=np.arange(200)
fig, (ax1, ax2)=plt.subplots(1,2)
ax1.plot(x, np.sum(G.T*np.arange(np.shape(G)[0]), axis=1), 'ko', label='Global Data')
ax1.plot(x, np.sum(G[:,0])*np.sum(GS.T*np.arange(np.shape(GS)[0]), axis=1), 'r-.', label='Global 2 exp sim')

ax2.step(GPEs, spectrum)
GS_spectrum=GS_spectrum/np.amax(GS_spectrum)*np.amax(spectrum)
ax2.plot(GPEs, GS_spectrum, 'r-.')
# model=model/np.amax(model)*np.amax(spectrum)
# ax2.step(GPEs, model, 'r-.')

plt.show()
