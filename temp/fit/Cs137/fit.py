import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.stats import poisson, binom
from scipy.special import erf as erf
from fun import make_3D, Sim
from minimize import minimize, make_ps
from PMTgiom import make_mash
start_time = time.time()
pmts=[0,1,4,7,8,14]

# path='/home/gerak/Desktop/DireXeno/190803/pulser/DelayRecon/'
# delay_hs=[]
# names=[]
# delays=[]
# for i in range(len(pmts)-1):
#     for j in range(i+1, len(pmts)):
#         data=np.load(path+'delay_hist{}-{}.npz'.format(pmts[i], pmts[j]))
#         delays.append(data['x']-data['m'])
#         delay_hs.append(data['h'])
#         names.append('{}_{}'.format(pmts[i], pmts[j]))
#
type='B'
path='/home/gerak/Desktop/DireXeno/190803/Co57'+type+'/EventRecon/'
data=np.load(path+'H.npz')
# H=data['H'][:30,:,:]
# G=data['G']
# spectrum=data['spectrum']
# spectra=data['spectra']
left=data['left']
right=data['right']
# cov=data['cov']
Xcov=data['Xcov']
t=np.arange(200)
dt=t[1]-t[0]
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

# PEs=np.arange(len(spectra[:,0]))
PEs=np.arange(100)
r_mash, V_mash, dS=make_mash(pmts)
l_min=1e10
counter=0
ls=[]
def L(p):
    global dS, PEs, r_mash, V_mash, counter, l_min, ls
    print('Counter', counter)
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
    V=V_mash*np.exp(-(r_mash[:,x1]+np.sqrt((10/40)**2-r_mash[:,x2]**2-r_mash[:,-1]**2))/rec[0]['mu'])
    if np.sum(V)==0:
        print('V=0, mu=', rec[0]['mu'])
        print(np.exp(-(r_mash[:,x1]+np.sqrt((10/40)**2-r_mash[:,x2]**2-r_mash[:,-1]**2))/rec[0]['mu']))
        sys.exit()
    m, s_model, Mcov=make_3D(t, rec['N'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0], rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0], dS, PEs, r_mash, V/np.sum(V), Xcov)
    model=np.sum(H[:,0,0])*np.ravel(m)
    data=np.ravel(H[:,:100,:])
    if np.any(model<0):
        print('model<0', np.amin(model))
        sys.exit()
        return 1e10*(1-np.amin(model))
    l+=np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

    data=np.ravel(spectra[PEs,:])
    model=np.sum(H[:,0,0])*np.ravel(s_model)
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

    model=np.sum(H[:,0,0])*np.ravel(Mcov)
    data=np.ravel(cov)
    if np.any(model<0):
        print('model<0', np.amin(model))
        sys.exit()
        return 1e10*(1-np.amin(model))
    l+=np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

    if -l<l_min:
        l_min=-l
        np.savez('best_p', p=p, l_min=l_min)
    counter+=1
    Rec[counter-1]=rec[0]
    ls.append(-l)
    np.savez('Rec', Rec=Rec, ls=ls, time=time.time()-start_time)
    return -l




rec[0]=([2.47449081e-01, 1.73455094e-01, 1.40915994e-01, 2.23938188e-01,
2.27196744e-01, 4.03724823e-01], [4.30007358e+01, 4.27364088e+01,
4.26244721e+01, 4.27050188e+01, 4.27595651e+01, 4.27768675e+01],
[1.19129338e+00, 7.06773710e-01, 6.42329986e-01, 1.02327019e+00,
8.84223879e-01, 9.07674808e-01], 5.53213718e-01, 7.83374741e+03,
9.86594244e-02, 8.09728899e-01, 3.18503273e+01, 5.54707755e-01,
3.85974402e-01, 2.39150776e-01)

rec[0]['mu']=10
V=V_mash
# V=np.ones(len(V_mash))/len(V_mash)
# r_mash=np.zeros(np.shape(r_mash))
# dS=np.ones(np.shape(dS))*dS[0]
# V=V_mash*np.exp(-(r_mash[:,x1]+np.sqrt((10/40)**2-r_mash[:,x2]**2-r_mash[:,-1]**2))/rec[0]['mu'])
s, GS, GS_spectrum, Sspectra, Gtrp, Gsng, GRtrp, GRsng, Scov, Xcov=Sim(rec['N'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0],
                                                                    rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0], PEs, rec[0]['mu'], x1, x2, r_mash, V/np.sum(V), left, right)
s_model, Mcov=make_3D(t, rec['N'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['R'][0], rec['a'][0], rec['eta'][0], rec['Q'][0], rec['T'][0], rec['St'][0], dS, PEs, r_mash, V/np.sum(V), Xcov)




fig, ax=plt.subplots(2,3)
for i in range(len(pmts)):
    np.ravel(ax)[i].plot(PEs, s_model[:,i], 'r.-')
    np.ravel(ax)[i].plot(PEs, Sspectra[:,i]/np.sum(Sspectra[:,0]), 'g.-', label='sim')
    np.ravel(ax)[i].legend()


fig, bx=plt.subplots(3,5)
k=0
for k in range(15):
    np.ravel(bx)[k].plot(Xcov, Mcov[:,k], 'ro', label='model')
    np.ravel(bx)[k].plot(Xcov, Scov[:,k]/np.sum(Sspectra[:,0]), 'go', label='sim')
    np.ravel(bx)[k].errorbar(Xcov, Scov[:,k]/np.sum(Sspectra[:,0]), yerr=np.sqrt(Scov[:,k])/np.sum(Sspectra[:,0]))
    np.ravel(bx)[k].legend()


plt.show()