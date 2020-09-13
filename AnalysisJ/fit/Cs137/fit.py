import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.stats import poisson, binom
from scipy.special import erf as erf
from fun import make_3D, Sim, Sim_fit
from minimize import minimize, make_ps
from PMTgiom import make_mash
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

type=''
path='/home/gerak/Desktop/DireXeno/190803/Cs137'+type+'/EventRecon/'
data=np.load(path+'H.npz')
H=data['H'][:50,:,:]
G=data['G']
spectrum=data['spectrum']
spectra=data['spectra']
left=data['left']
right=data['right']
cov=data['cov']
Xcov=data['Xcov']
# t=data['t']
# dt=data['dt']
t=np.arange(200)
dt=np.ones(len(t))

if type=='B':
    x1=1
    x2=0
elif type=='':
    x1=0
    x2=1

rec=np.recarray(1, dtype=[
    ('Q', 'f8', len(pmts)),
    ('T', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('mu', 'f8', 1),
    ('N', 'f8', 1),
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
    ('mu', 'f8', 1),
    ('N', 'f8', 1),
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
            elif name=='N':
                rec[name][0]=p[-7]
            elif name=='mu':
                rec[name][0]=p[-8]
            else:
                print('fuck')
                sys.exit()
    return rec

PEs=np.arange(len(spectra[:,0]))
# PEs=np.arange(100)
r_mash, V_mash, dS=make_mash(pmts)
l_min=1e10
counter=0
ls=[]
def L(p):
    start_time = time.time()
    global dS, PEs, r_mash, counter, l_min, ls
    rec=p_to_rec(p)
    Names=['Q', 'Ts', 'T', 'F', 'Tf', 'Ts', 'R', 'a', 'eta']
    for name in Names:
        if np.any(rec[name]<0):
            return 1e10*(1-np.amin(rec[name]))

    Names=['F', 'R', 'eta', 'a']
    for name in Names:
        if np.any(rec[name]>1):
            return 1e10*(np.amax(rec[name]))
    l=0
    S, Sspectra, Scov=Sim_fit(rec['N'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0],
                                                                        rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0], PEs, rec[0]['mu'], x1, x2, r_mash, left, right)
    model=np.sum(H[:,0,0])*np.ravel(S)
    data=np.ravel(H[:,:,:])
    if np.any(model<0):
        print(counter, time.time()-start_time, 1e10*(1-np.amin(model)))
        return 1e10*(1-np.amin(model))
    l+=np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

    data=np.ravel(spectra[PEs,:])
    model=np.sum(H[:,0,0])*np.ravel(Sspectra)
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

    model=np.sum(H[:,0,0])*np.ravel(Scov)
    data=np.ravel(cov)
    if np.any(model<0):
        print('model<0', np.amin(model))
        sys.exit()
        return 1e10*(1-np.amin(model))
    l+=np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

    print(counter, time.time()-start_time, l)

    if -l<l_min:
        l_min=-l
        np.savez('best_p', p=p, l_min=l_min)
    counter+=1
    Rec[counter-1]=rec[0]
    ls.append(-l)
    np.savez('Rec', Rec=Rec, ls=ls, time=time.time()-start_time)
    return -l



#Co
# rec[0]=([2.51289925e-01, 1.82086891e-01, 1.45211638e-01, 2.26251668e-01,
#  2.28801541e-01, 3.76389026e-01], [4.27081315e+01, 4.24972396e+01,
#  4.26176429e+01, 4.25952143e+01, 4.26876810e+01, 4.28385368e+01],
#  [1.09602380e+00, 6.09583279e-01, 5.28437439e-01, 1.17965295e+00,
#  9.30813312e-01, 8.63177126e-01], 5.92083629e-01, 7.95430192e+03,
#  9.21861715e-02, 1.05468747e+00, 3.20438254e+01, 5.53313508e-01,
#  3.74126949e-01, 2.68691223e-01)


#CoB
# rec[0]=([2.47449081e-01, 1.73455094e-01, 1.40915994e-01, 2.23938188e-01,
# 2.27196744e-01, 4.03724823e-01], [4.30007358e+01, 4.27364088e+01,
# 4.26244721e+01, 4.27050188e+01, 4.27595651e+01, 4.27768675e+01],
# [1.19129338e+00, 7.06773710e-01, 6.42329986e-01, 1.02327019e+00,
# 8.84223879e-01, 9.07674808e-01], 5.53213718e-01, 7.83374741e+03,
# 9.86594244e-02, 8.09728899e-01, 3.18503273e+01, 5.54707755e-01,
# 3.85974402e-01, 2.39150776e-01)

 #Cs
rec[0]=([3.04296339e-01, 2.08690740e-01, 1.77672303e-01, 2.66309963e-01,
2.41427960e-01, 4.85648231e-01], [4.34374353e+01, 4.31961978e+01,
4.31231821e+01, 4.34677352e+01, 4.35263087e+01, 4.32596742e+01],
[1.23586665e+00, 9.79583658e-01, 9.07633523e-01, 1.50414341e+00,
9.83470468e-01, 6.78459024e-01], 4.46079733e+00, 622*60,
9.43331431e-02, 4.01582004e+00, 3.52528630e+01, 5.32416831e-01,
1.81172211e-01, 3.51986897e-01)

# CsB
# rec[0]=([2.45320036e-01, 1.66503338e-01, 1.60613707e-01, 2.35057522e-01,
#  2.26676081e-01, 4.19947869e-01], [4.28190829e+01, 4.28455151e+01,
#  4.27058445e+01, 4.23081522e+01, 4.30042587e+01, 4.26596060e+01],
#  [1.34047633e+00, 6.54632148e-01, 6.41646582e-01, 1.02866161e+00,
#  1.04377416e+00, 7.57914588e-01], 6.15129303e+00, 3.30747763e+04,
#  9.06991646e-02, 1.36137465e+00, 3.58038043e+01, 5.71828126e-01,
#  3.66017987e-01, 3.05918824e-01)

p=minimize(L, make_ps(rec_to_p(rec)))
# rec=p_to_rec(p.x)
start_time = time.time()
# m, s_model, Mcov=make_3D(t, dt, rec['N'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0], rec['eta'][0], rec['Q'][0], rec['T'][0]-35, rec['St'][0], dS, PEs, r_mash, V/np.sum(V), Xcov)
S, Sspectra, Scov=Sim_fit(rec['N'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0],
                                                                    rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0], PEs, rec[0]['mu'], x1, x2, r_mash, left, right)
dT=time.time()-start_time
print(dT)

fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].step(t, np.sum(H[:,:,i].T*np.arange(np.shape(H)[0]), axis=1)/(np.sum(H[:,0,i])*dt), label='Data - PMT{}'.format(pmts[i]), linewidth=3, where='post')
    np.ravel(ax)[i].plot(t[t<100], np.sum(S[:,:100,i].T*np.arange(np.shape(S)[0]), axis=1)/dt[t<100], 'ro', label='model', linewidth=3)
    # np.ravel(ax)[i].plot(np.arange(np.shape(s)[1]), np.sum(s[:,:,i].T*np.arange(np.shape(s)[0]), axis=1), 'g.', label='sim', linewidth=3)
    # np.ravel(ax)[i].errorbar(np.arange(np.shape(s)[1]), np.sum(s[:,:,i].T*np.arange(np.shape(s)[0]), axis=1), np.sum(np.sqrt(s[:,:,i].T)*np.arange(np.shape(s)[0]), axis=1)/np.sqrt(np.sum(Sspectra[:,0])), fmt='k.')
    # np.ravel(ax)[i].legend(fontsize=15)
    np.ravel(ax)[i].set_xlabel('Time [ns]', fontsize='15')
fig.text(0.04, 0.5, r'$N_{events}\sum_n nH_{ni}$', va='center', rotation='vertical', fontsize=15)


fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(PEs, spectra[PEs,i], 'ko', label='spectrum - PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].plot(PEs, np.sum(spectra[:,0])*Sspectra[:,i], 'g.-', label='sim')
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


fig, bx=plt.subplots(3,5)
k=0
for k in range(15):
    np.ravel(bx)[k].step(Xcov, cov[:,k], where='mid')
    np.ravel(bx)[k].plot(Xcov, Scov[:,k]*np.sum(spectra[:,0]), 'go', label='sim')
    np.ravel(bx)[k].legend()

plt.show()
