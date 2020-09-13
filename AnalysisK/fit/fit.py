import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.stats import poisson, binom
from scipy.special import erf as erf
from fun import Sim_fit, Sim, make_3D
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

type='B'
source='Co57'
path='/home/gerak/Desktop/DireXeno/190803/'+source+type+'/EventRecon/'
data=np.load(path+'H.npz')
H=data['H'][:50,:,:]
G=data['G']
spectrum=data['spectrum']
spectra=data['spectra']
left=data['left']
right=data['right']
cov=data['cov']
Xcov=data['Xcov']
t=np.arange(200)
dt=1

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
l_min=1e10
counter=0
ls=[]

def L(p):
    global counter, l_min, ls
    rec=p_to_rec(p)
    Names=['Q', 'Ts', 'T', 'F', 'Tf', 'St', 'R', 'a', 'eta', 'N', 'mu']
    for name in Names:
        if np.any(rec[name]<0):
            return 1e10*(1-np.amin(rec[name]))

    Names=['F', 'R', 'eta', 'a', 'Q']
    for name in Names:
        if np.any(rec[name]>1):
            return 1e10*(np.amax(rec[name]))
    l=0
    S, Sspectra, Scov=Sim_fit(rec['N'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0],
                                                                        rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0], PEs, rec[0]['mu'], x1, x2, left, right)
    model=np.sum(H[:,0,0])*np.ravel(S[:,:100,:])
    data=np.ravel(H[:,:100,:])
    if np.any(model<0):
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

    print(counter, l)
    counter+=1
    Rec[counter-1]=rec[0]
    ls.append(-l)
    np.savez('Rec_'+source+type, Rec=Rec, ls=ls, time=time.time()-start_time)
    return -l

start_time=time.time()

rec[0]=([0.28609523, 0.21198892, 0.1661045 , 0.23595573, 0.2543458 , 0.46767996], [42.43727439, 42.48680044, 42.48223214, 42.61715417, 42.97131299, 42.35603571],
 [1.14722701, 0.82496347, 0.71858647, 1.61434698, 1.48554624, 1.03053529], 2.57341188, 7920.93881033, 0.11035399, 0.94339727, 34.3602973, 0.5760872, 0.33983152, 0.36124252)

L1=L(rec_to_p(rec))



# p=minimize(L, make_ps(rec_to_p(rec), source))
# M,Mspectra, Mcov=make_3D(rec['N'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0],
#                                                                     rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0], PEs, rec[0]['mu'], x1, x2, Xcov)

S, Sspectra, Scov, G, Gtrp, Gsng, GRtrp, GRsng, SN=Sim(rec['N'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0],
                                                                    rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0], PEs, rec[0]['mu'], x1, x2, left, right)



rec[0]=([2.47449081e-01, 1.73455094e-01, 1.40915994e-01, 2.23938188e-01,
2.27196744e-01, 4.03724823e-01], [4.30007358e+01, 4.27364088e+01,
4.26244721e+01, 4.27050188e+01, 4.27595651e+01, 4.27768675e+01],
[1.19129338e+00, 7.06773710e-01, 6.42329986e-01, 1.02327019e+00,
8.84223879e-01, 9.07674808e-01], 5.53213718e-01, 7.83374741e+03,
9.86594244e-02, 8.09728899e-01, 3.18503273e+01, 5.54707755e-01,
3.85974402e-01, 2.39150776e-01)
L2=L(rec_to_p(rec))
S2, Sspectra2, Scov2, G2, Gtrp2, Gsng2, GRtrp2, GRsng2, SN2=Sim(rec['N'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0],
                                                                    rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0], PEs, rec[0]['mu'], x1, x2, left, right)


print(L1, L2)
N=np.sum(H[:,0,0])
fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].step(t, np.sum(H[:,:,i].T*np.arange(np.shape(H)[0]), axis=1)/(dt), label='Data - PMT{}'.format(pmts[i]), linewidth=3, where='post')
    # np.ravel(ax)[i].plot(t[:100]+0.5, N*np.sum(M[:,:,i].T*np.arange(np.shape(M)[0]), axis=1)/dt, 'ro', label='model', linewidth=3)
    np.ravel(ax)[i].plot(t+0.5, N*np.sum(S[:,:,i].T*np.arange(np.shape(S)[0]), axis=1)/dt, 'g.', label='sim', linewidth=3)
    np.ravel(ax)[i].plot(t+0.5, N*np.sum(S2[:,:,i].T*np.arange(np.shape(S2)[0]), axis=1)/dt, 'r.', label='sim2', linewidth=3)
    np.ravel(ax)[i].errorbar(t+0.5, N*np.sum(S[:,:,i].T*np.arange(np.shape(S)[0]), axis=1)/dt, N*np.sqrt(np.sum(S[:,:,i].T*np.arange(np.shape(S)[0]), axis=1)/(dt*SN)), fmt='g.')
    np.ravel(ax)[i].set_xlabel('Time [ns]', fontsize='15')
fig.text(0.04, 0.5, r'$N_{events}\sum_n nH_{ni}$', va='center', rotation='vertical', fontsize=15)

fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(PEs, spectra[:,i], 'ko', label='spectrum - PMT{}'.format(pmts[i]))
    # np.ravel(ax)[i].plot(PEs, N*Mspectra[:,i], 'r.', label='model')
    np.ravel(ax)[i].plot(PEs, N*Sspectra[:,i], 'g.', label='sim')
    np.ravel(ax)[i].plot(PEs, N*Sspectra2[:,i], 'r.', label='sim2')
    np.ravel(ax)[i].errorbar(PEs, N*Sspectra[:,i], N*np.sqrt(Sspectra[:,i]/SN), fmt='g.')
    np.ravel(ax)[i].legend()

ig, ax=plt.subplots(3,5)
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
    # np.ravel(bx)[k].plot(Xcov, Mcov[:,k]*N, 'r.', label='model')
    np.ravel(bx)[k].plot(Xcov, Scov[:,k]*N, 'g.', label='sim')
    np.ravel(bx)[k].plot(Xcov, Scov2[:,k]*N, 'r.', label='sim')
    np.ravel(bx)[k].errorbar(Xcov, Scov[:,k]*N, N*np.sqrt(Scov[:,k]/SN), fmt='g.')
    np.ravel(bx)[k].legend()


plt.show()
