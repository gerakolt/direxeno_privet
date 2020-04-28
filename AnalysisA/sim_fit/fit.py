import numpy as np
import matplotlib.pyplot as plt
import time
import os
from fun import Model, Sim
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf
import warnings

pmts=[7,8]
path='/home/gerak/Desktop/DireXeno/190803/Co57/EventRecon/'
rec=np.load(path+'recon.npz')['rec']
data=np.load(path+'H.npz')
H=data['H']
ns=data['ns']
blw_cut=4.7
init_cut=20
chi2_cut=500

rec=rec[np.all(rec['init_wf']>init_cut, axis=1)]
rec=rec[np.all(rec['blw']<blw_cut, axis=1)]
rec=rec[np.all(rec['chi2']<chi2_cut, axis=1)]

rec=np.recarray(1, dtype=[
    ('NQ', 'f8', len(pmts)),
    ('T', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('F', 'f8', len(pmts)),
    ('Tf', 'f8', len(pmts)),
    ('Ts', 'f8', len(pmts)),
    ('Strig', 'f8', 1)
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
            if name=='Strig':
                rec[name][0]=p[-1]
            else:
                print('fuck --- problem with p_to_rec')
                sys.exit()
    return rec

counter=0
Ls=[]
def L(p):
    rec=p_to_rec(p)
    global counter
    counter+=1

    names=['NQ', 'St', 'F', 'Tf', 'Ts', 'T']
    for name in names:
        if np.any(rec[name]<0):
            return 1e10*(1-np.amin(rec[name]))
    names=['F']
    for name in names:
        if np.any(rec[name]>1):
            return 1e10*(np.amax(rec[name]))

    l=0
    t=Sim(rec['NQ'][0], rec['T'][0], rec['Strig'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0])

    for i in range(len(pmts)):
        model=np.sum(H[:,0,i])*np.ravel(t[:,:500,i])
        if np.any(np.isnan(model)) or np.any(np.isinf(model)):
            print('model is nan or inf')
            print('NQ=', rec['NQ'][0,i], 'T=', rec['T'][0,i], 'F=', rec['F'][0,i], 'Tf=', rec['Tf'][0,i], 'Ts=', rec['Ts'][0,i], 'St=', rec['St'][0, i])
            plt.figure()
            plt.plot(np.mean(t.T*np.arange(np.shape(t)[0])), 'k.')
            plt.show()
            sys.exit()
        data=np.ravel(H[:,:500,i])
        l+=np.mean((model-data)**2)

        # for j in range(L):
        #     if model[j]>0 and data[j]<=0:
        #         l-=model[j]-data[j]
        #     elif model[j]<=0 and data[j]>0:
        #         return 1e10*(data[j]-model[j])
        #     elif model[j]==0 and data[j]==0:
        #         l+=1
        #     else:
        #         l+=data[j]*np.log(model[j])-data[j]*np.log(data[j])+data[j]-model[j]
    Ls.append(l)
    if True:
        np.savez('L', Ls, rec)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('iteration=', int(counter/(len(p)+1)), 'fanc=',l)
        print('--------------------------------')
        print(rec)
    return l

rec[0]=([31.80178041, 35.76559763], [39.93532598, 40.53743245], [0.44510939, 1.02078997], [0.22746828, 0.04994591], [14.40456186,  0.14758156], [46.53134732, 35.93295095], 1.98999684)
# p=minimize(L, rec_to_p(rec), method='Nelder-Mead', options={'disp':True, 'maxfev':100000})
# rec=p_to_rec(p.x)

t=Sim(rec['NQ'][0], rec['T'][0], rec['Strig'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0])
m=Model(rec['NQ'][0], rec['T'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0])
x=np.arange(1000)/5
fig, (ax1, ax2)=plt.subplots(2,1, sharex=True)

ax1.plot(x[:250], np.mean(H[:,:,0].T*np.arange(np.shape(H)[0]), axis=1)[:250], 'ko', label='Data')
ax1.plot(x[:250], np.sum(H[:,0,0])*np.mean(t[:,:,0].T*np.arange(np.shape(H)[0]), axis=1)[:250], '.-', label='2 exp model', linewidth=3)
ax1.plot(x[:250], np.sum(H[:,0,0])*np.mean(m[:,:,0].T*np.arange(np.shape(H)[0]), axis=1)[:250], '.-', label='2 exp model', linewidth=3)


ax2.plot(x[:250], np.mean(H[:,:,1].T*np.arange(np.shape(H)[0]), axis=1)[:250], 'ko', label='Data')
ax2.plot(x[:250], np.sum(H[:,0,1])*np.mean(t[:,:,1].T*np.arange(np.shape(H)[0]), axis=1)[:250], '.-', label='2 exp model', linewidth=3)
ax2.plot(x[:250], np.sum(H[:,0,1])*np.mean(m[:,:,1].T*np.arange(np.shape(H)[0]), axis=1)[:250], '.-', label='2 exp model', linewidth=3)

plt.subplots_adjust(hspace=0)
plt.show()
