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


path='/home/gerak/Desktop/DireXeno/190803/Co57/EventRecon/'
data=np.load(path+'H.npz')
H=data['H']
G=data['G']
spectrum=data['spectrum']
spectra=data['spectra']
left=data['left']
right=data['right']
t=np.arange(200)
dt=t[1]-t[0]


rec=np.recarray(1, dtype=[
    ('Q', 'f8', len(pmts)),
    ('T', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('N', 'f8', 1),
    ('F1', 'f8', 1),
    ('F2', 'f8', 1),
    ('Tf', 'f8', 1),
    ('Tm', 'f8', 1),
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
            elif name=='Tm':
                rec[name][0]=p[-2]
            elif name=='Tf':
                rec[name][0]=p[-3]
            elif name=='F2':
                rec[name][0]=p[-4]
            elif name=='F1':
                rec[name][0]=p[-5]
            elif name=='N':
                rec[name][0]=p[-6]
            else:
                print('fuck')
                sys.exit()
    return rec

counter=0
PEs=np.arange(len(spectra[:,0]))
GPEs=np.arange(len(spectrum))
def L(p):
    rec=p_to_rec(p)
    global counter
    counter+=1

    nams=['Q', 'Ts', 'T', 'F1', 'F2', 'Tf', 'Ts']
    for name in nams:
        if np.any(rec[name]<0):
            return 1e10*(1-np.amin(rec[name]))
    nams=['F1', 'F2']
    for name in nams:
        if np.any(rec[name]>1):
            return 1e10*(np.amax(rec[name]))
    if rec['Ts'][0]>100:
        return 1e10*rec['Ts'][0]
    if np.any(rec['St'][0]<0.5):
        return 1e10*(1+np.abs(np.amin(rec['St'][0])))


    l=0
    m=make_3D(t, dt, rec['N'][0], np.zeros(len(pmts)), rec['F1'][0], rec['F2'][0], rec['Tf'][0], rec['Tm'][0], rec['Ts'][0], rec['Q'][0], rec['T'][0], rec['St'][0])
    model=np.ravel(m*np.sum(H[:,0,:], axis=0))
    data=np.ravel(H)
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

    if counter%(len(p)+1)==0:
    # if True:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('iteration=', int(counter/(len(p)+1)), 'fanc=',-l)
        print('--------------------------------')
        print(rec)
    return -l

rec[0]=([0.30878609, 0.22992061, 0.18489446, 0.30308087, 0.30266814, 0.48582283],
 [5.08976768e-01, 2.24425730e-09, 4.35594578e-01, 2.15423030e-01, 2.35416538e-01, 4.77517045e-01],
  [0.78569348, 0.9, 0.97304069, 0.936820594, 1.16424185, 1.02137693], 115.72704788, 0.15770965, 13.82264744, 37.57653787)

print(L(rec_to_p(rec)))
# p=minimize(L, rec_to_p(rec), method='Nelder-Mead', options={'disp':True, 'maxfev':100000})
# rec=p_to_rec(p.x)

m=make_3D(t, dt, rec['N'][0], np.zeros(len(pmts)), rec['F1'][0], rec['F2'][0], rec['Tf'][0], rec['Tm'][0], rec['Ts'][0], rec['Q'][0], rec['T'][0], rec['St'][0])
s, GS, GS_spectrum=Sim(rec['N'][0], rec['Q'][0], rec['T'][0], 5,  np.zeros(len(pmts)), rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0])


fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(t, np.sum(H[:,:,i].T*np.arange(np.shape(H)[0]), axis=1), 'ko', label='Data - PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].plot(t, np.sum(H[:,0,i])*np.sum(m[:,:,i].T*np.arange(np.shape(m)[0]), axis=1), 'r.-', label='2 exp model', linewidth=3)
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
