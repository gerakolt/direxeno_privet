import numpy as np
import matplotlib.pyplot as plt
import time
import os
from fun import Model, Sim, q0_model, make_P, model_area, smd5, comb
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf
import warnings

pmts=[7,8]
H_areas=[]
areas=[]
rng_areas=[]
for i, pmt in enumerate(pmts):
    path='/home/gerak/Desktop/DireXeno/190803/pulser/NEWPMT{}/'.format(pmt)
    data=np.load(path+'areas.npz')
    H_areas.append(data['H_areas'])
    areas.append(data['areas'])
    rng_areas.append(data['rng_area'])

path='/home/gerak/Desktop/DireXeno/190803/pulser/delays/'
data=np.load(path+'delays_7_8.npz')
delays=data['delays']
delay_h=data['h_delays']
rng_delay=np.nonzero(np.logical_and(delays>delays[np.argmax(delay_h)]-2, delays<delays[np.argmax(delay_h)]+2))[0]

path='/home/gerak/Desktop/DireXeno/190803/pulser/EventRecon/'
rec=np.load(path+'recon.npz')['rec']
h0, bins=np.histogram(np.ravel(rec['h'][:,400:,0]), bins=np.arange(5)-0.5)
h1, bins=np.histogram(np.ravel(rec['h'][:,400:,1]), bins=np.arange(5)-0.5)
n_q0=0.5*(bins[1:]+bins[:-1])
h_q0=[h0, h1]


path='/home/gerak/Desktop/DireXeno/190803/Co57/EventRecon/'
rec=np.load(path+'recon.npz')['rec']
h=rec['h']
spectrum=np.zeros((100, len(pmts)))
spec_PEs=np.arange(30,40)
for i in range(len(pmts)):
    spectrum[:,i]=np.histogram(np.sum(h[:,:,i], axis=1), bins=np.arange(101)-0.5)[0]

data=np.load(path+'H.npz')
H=data['H']
G=data['G']
spectra=data['spectra']
PEs=np.arange(len(spectra[0]))


rec1=np.recarray(1, dtype=[
    ('NQ', 'f8', len(pmts)),
    ('T', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('R', 'f8', len(pmts)),
    ('F', 'f8', 1),
    ('Tf', 'f8', 1),
    ('Ts', 'f8', 1),
    ('a_delay', 'f8', 1),
    ])


rec2=np.recarray(1, dtype=[
    ('NQ', 'f8', len(pmts)),
    ('T', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('R', 'f8', len(pmts)),
    ('F', 'f8', 1),
    ('Tf', 'f8', 1),
    ('Ts', 'f8', 1),
    ('a_delay', 'f8', 1),
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


rec1[0]=([34.17255625, 35.09148282], [39.65578361, 39.30607808], [0.98964095, 1.08379849], [0,0], 0.07063164, 0.21911959, 36.64258893, 463.23111438)
rec2[0]=([34.1231953 , 35.08659966], [38.19114816, 37.94501817], [0.97305945, 1.11166723], [0.0443506 , 0.05271929], 0.02677325, 0.21487918, 36.61184292, 466.39540394)


P=np.dstack((np.identity(100), np.identity(100)))
m1=Model(rec1['NQ'][0], rec1['T'][0], rec1['R'][0], rec1['F'][0], rec1['Tf'][0], rec1['Ts'][0], rec1['St'][0], [0,0], P)
s1=Sim(rec1['NQ'][0], rec1['T'][0], 5, rec1['R'][0], rec1['F'][0], rec1['Tf'][0], rec1['Ts'][0], rec1['St'][0], [0,0], [0,0], [1e-6, 1e-6], [1e-6, 1e-6], [0,0])

m2=Model(rec2['NQ'][0], rec2['T'][0], rec2['R'][0], rec2['F'][0], rec2['Tf'][0], rec2['Ts'][0], rec2['St'][0], [0,0], P)
s2=Sim(rec2['NQ'][0], rec2['T'][0], 5, rec2['R'][0], rec2['F'][0], rec2['Tf'][0], rec2['Ts'][0], rec2['St'][0], [0,0], [0,0], [1e-6, 1e-6], [1e-6, 1e-6], [0,0])

x=np.arange(200)/5

fig, ((ax1, ax3), (ax2, ax4))=plt.subplots(2,2, sharex=True)


ax1.title.set_text('Double Exp')
ax1.plot(x[:100], smd5(np.sum(H[:,:,0].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'ko', label='Data - PMT7')
ax1.plot(x[:100], smd5(np.sum(H[:,0,0])*np.sum(m1[:,:,0].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'r.-', label='2 exp model', linewidth=3)
ax1.plot(x[:100], smd5(np.sum(H[:,0,0])*np.sum(s1[:,:,0].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'g.-', label='2 exp simulation', linewidth=3)

ax2.plot(x[:100], smd5(np.sum(H[:,:,1].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'ko', label='Data - PMT8')
ax2.plot(x[:100], smd5(np.sum(H[:,0,1])*np.sum(m1[:,:,1].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'r.-', label='2 exp model', linewidth=3)
ax2.plot(x[:100], smd5(np.sum(H[:,0,1])*np.sum(s1[:,:,1].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'g.-', label='2 exp simulation', linewidth=3)

ax1.legend(fontsize=15)
ax2.legend(fontsize=15)
ax2.set_xlabel('Time [ns]', fontsize='15')

ax3.title.set_text(r'$\delta(t)+$'+'Double Exp')
ax3.plot(x[:100], smd5(np.sum(H[:,:,0].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'ko', label='Data - PMT7')
ax3.plot(x[:100], smd5(np.sum(H[:,0,0])*np.sum(m2[:,:,0].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'r.-', label=r'$\delta(t)+$'+' 2 exp model', linewidth=3)
ax3.plot(x[:100], smd5(np.sum(H[:,0,0])*np.sum(s2[:,:,0].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'g.-', label=r'$\delta(t)+$'+' 2 exp simulation', linewidth=3)

ax4.plot(x[:100], smd5(np.sum(H[:,:,1].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'ko', label='Data - PMT8')
ax4.plot(x[:100], smd5(np.sum(H[:,0,1])*np.sum(m2[:,:,1].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'r.-', label=r'$\delta(t)+$'+' 2 exp model', linewidth=3)
ax4.plot(x[:100], smd5(np.sum(H[:,0,1])*np.sum(s2[:,:,1].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'g.-', label=r'$\delta(t)+$'+' 2 exp simulation', linewidth=3)

ax3.legend(fontsize=15)
ax4.legend(fontsize=15)
ax4.set_xlabel('Time [ns]', fontsize='15')
plt.subplots_adjust(hspace=0)
fig.text(0.04, 0.5, r'$N_{events}\sum_n nH_{ni}$', va='center', rotation='vertical', fontsize=15)

model1=rec1['a_delay'][0]*np.exp(-0.5*(delays[rng_delay]-rec1['T'][0,1]+rec1['T'][0,0])**2/(rec1['St'][0,0]**2+rec1['St'][0,1]**2))/np.sqrt(2*np.pi*(rec1['St'][0,0]**2+rec1['St'][0,1]**2))
model2=rec2['a_delay'][0]*np.exp(-0.5*(delays[rng_delay]-rec2['T'][0,1]+rec2['T'][0,0])**2/(rec2['St'][0,0]**2+rec2['St'][0,1]**2))/np.sqrt(2*np.pi*(rec2['St'][0,0]**2+rec2['St'][0,1]**2))

plt.figure()
plt.plot(delays, delay_h, 'ko', label='Delay PMT8-PMT7')
plt.plot(delays[rng_delay], model1, 'r.-', linewidth=5, label='2 exp model')
plt.plot(delays[rng_delay], model2, 'g.-', linewidth=5, label=r'$\delta(t)+$'+'2 exp model')

plt.xlabel('Delay [ns]', fontsize=25)
plt.legend(fontsize=25)
plt.xticks(fontsize=15)

fig, (ax1, ax2)=plt.subplots(1,2, sharey=True)
spectra_rng=np.nonzero(np.logical_and(PEs>PEs[np.argmax(spectra[0])-5], PEs<PEs[np.argmax(spectra[0])+5]))[0]
model1=np.sum(H[:,0,0])*poisson.pmf(PEs, np.sum(m1[:,:,0].T*np.arange(np.shape(H)[0])))[spectra_rng]
model2=np.sum(H[:,0,0])*poisson.pmf(PEs, np.sum(m2[:,:,0].T*np.arange(np.shape(H)[0])))[spectra_rng]

ax1.plot(PEs, spectra[0], 'ko', label='Total number of PEs\nin event - PMT7')
ax1.plot(PEs[spectra_rng], model1, 'r.-', label='2 exp model', linewidth=5)
ax1.plot(PEs[spectra_rng], model2, 'g.-', label=r'$\delta(t)+$'+'2 exp model', linewidth=5)



spectra_rng=np.nonzero(np.logical_and(PEs>PEs[np.argmax(spectra[1])-5], PEs<PEs[np.argmax(spectra[1])+5]))[0]
model1=np.sum(H[:,0,1])*poisson.pmf(PEs, np.sum(m1[:,:,1].T*np.arange(np.shape(H)[0])))[spectra_rng]
model2=np.sum(H[:,0,1])*poisson.pmf(PEs, np.sum(m2[:,:,1].T*np.arange(np.shape(H)[0])))[spectra_rng]

ax2.plot(PEs, spectra[1], 'ko', label='Total number of PEs\nin event - PMT8')
ax2.plot(PEs[spectra_rng], model1, 'r.-', label='2 exp model', linewidth=5)
ax2.plot(PEs[spectra_rng], model2, 'g.-', label=r'$\delta(t)+$'+'2 exp model', linewidth=5)

plt.subplots_adjust(wspace=0)
ax1.set_xlabel('PE', fontsize=25)
ax2.set_xlabel('PE', fontsize=25)
ax1.legend(fontsize=25)
ax2.legend(fontsize=25)

plt.show()
