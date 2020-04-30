import numpy as np
import matplotlib.pyplot as plt
import time
import os
from fun import Model, Sim, q0_model
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf
import warnings

pmts=[7,8]
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
# rec=np.load(path+'recon.npz')['rec']
data=np.load(path+'H.npz')
H=data['H']
ns=data['ns']
blw_cut=4.7
init_cut=20
chi2_cut=500

# rec=rec[np.all(rec['init_wf']>init_cut, axis=1)]
# rec=rec[np.all(rec['blw']<blw_cut, axis=1)]
# rec=rec[np.all(rec['chi2']<chi2_cut, axis=1)]

rec=np.recarray(1, dtype=[
    ('NQ', 'f8', len(pmts)),
    ('T', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('R', 'f8', len(pmts)),
    ('q0', 'f8', len(pmts)),
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
                rec[name][0]=p[-4]
            elif name=='Tf':
                rec[name][0]=p[-3]
            elif name=='Ts':
                rec[name][0]=p[-2]
            elif name=='a_delay':
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

    names=['NQ', 'St', 'F', 'Tf', 'Ts', 'T', 'R', 'q0']
    for name in names:
        if np.any(rec[name]<0):
            return 1e10*(1-np.amin(rec[name]))
    names=['F', 'R', 'q0']
    for name in names:
        if np.any(rec[name]>1):
            return 1e10*(np.amax(rec[name]))
    if rec['Ts'][0]>100:
        return 1e10*rec['Ts'][0]

    l=0
    t=Model(rec['NQ'][0], rec['T'][0], rec[0]['R'], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0], rec['q0'][0])

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
        L=len(model)
        for j in range(L):
            if model[j]>0 and data[j]<=0:
                l-=model[j]-data[j]
            elif model[j]<=0 and data[j]>0:
                return 1e10*(data[j]-model[j])
            elif model[j]==0 and data[j]==0:
                l+=1
            else:
                l+=data[j]*np.log(model[j])-data[j]*np.log(data[j])+data[j]-model[j]

        model=np.sum(h_q0[i])*q0_model(n_q0, rec['q0'][0,i])
        data=h_q0[i]

        L=len(model)
        for j in range(L):
            if model[j]>0 and data[j]<=0:
                l-=model[j]-data[j]
            elif model[j]<=0 and data[j]>0:
                return 1e10*(data[j]-model[j])
            elif model[j]==0 and data[j]==0:
                l+=1
            else:
                l+=data[j]*np.log(model[j])-data[j]*np.log(data[j])+data[j]-model[j]

    model=rec['a_delay'][0]*np.exp(-0.5*(delays[rng_delay]-rec['T'][0,1]+rec['T'][0,0])**2/(rec['St'][0,0]**2+rec['St'][0,1]**2))/np.sqrt(2*np.pi*(rec['St'][0,0]**2+rec['St'][0,1]**2))
    data=delay_h[rng_delay]

    L=len(model)
    for j in range(L):
        if model[j]>0 and data[j]<=0:
            l-=model[j]-data[j]
        elif model[j]<=0 and data[j]>0:
            return 1e10*(data[j]-model[j])
        elif model[j]==0 and data[j]==0:
            l+=1
        else:
            l+=data[j]*np.log(model[j])-data[j]*np.log(data[j])+data[j]-model[j]

    if counter%(len(p)+1)==0:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('iteration=', int(counter/(len(p)+1)), 'fanc=',-l)
        print('--------------------------------')
        print(rec)
    return -l

# rec[0]=([44.10656898, 44.16785627], [2.40187361, 2.05004474], [0.88833944, 1.11088245], [0.04048149, 0.04177063], [5.64966695e-05, 2.44157508e-05], 0.66410346, 31.87824534, 367.91330633, 460.48414159)
rec[0]=([36.64488212, 36.70417636], [40, 40], [0.88574388, 1.10794293], [0.04836568, 0.04973108], [5.63314209e-05, 2.45113089e-05], 0.71182004, 30.10985681, 99.99999927, 460.70331834)

# p=minimize(L, rec_to_p(rec), method='Nelder-Mead', options={'disp':True, 'maxfev':100000})
# rec=p_to_rec(p.x)

t=Model(rec['NQ'][0], rec['T'][0], rec[0]['R'], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0], rec['q0'][0])
s=Sim(rec['NQ'][0], rec['T'][0], 1, rec['R'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0], rec['q0'][0])


x=np.arange(1000)/5

fig, ((ax1, ax3), (ax2, ax4))=plt.subplots(2,2)

ax1.plot(x[:30*5], np.mean(H[:,:,0].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], 'ko', label='Data - PMT7')
ax1.plot(x[:30*5], np.sum(H[:,0,0])*np.mean(t[:,:,0].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], 'r.-', label=r'$\delta+$'+' 2 exp model', linewidth=3)
ax1.plot(x[:30*5], np.sum(H[:,0,0])*np.mean(s[:,:,0].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], 'g.-', label=r'$\delta+$'+' 2 exp simulation', linewidth=3)

ax2.plot(x[:30*5], np.mean(H[:,:,1].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], 'ko', label='Data - PMT8')
ax2.plot(x[:30*5], np.sum(H[:,0,1])*np.mean(t[:,:,1].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], 'r.-', label=r'$\delta+$'+' 2 exp model', linewidth=3)
ax2.plot(x[:30*5], np.sum(H[:,0,1])*np.mean(s[:,:,1].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], 'g.-', label=r'$\delta+$'+' 2 exp simulation', linewidth=3)

ax1.legend(fontsize=15)
ax2.legend(fontsize=15)
ax2.set_xlabel('Time [ns]', fontsize='15')
fig.text(0.04, 0.5, r'$N_{events}\sum_n nH_{ni}$', va='center', rotation='vertical', fontsize=15)

# fig,  (ax3, ax4)=plt.subplots(2,1)
ax3.plot(delays, delay_h, 'ko')
ax3.plot(delays[rng_delay], rec['a_delay'][0]*np.exp(-0.5*(delays[rng_delay]-rec['T'][0,1]+rec['T'][0,0])**2/(rec['St'][0,0]**2+rec['St'][0,1]**2))/np.sqrt(2*np.pi*(rec['St'][0,0]**2+rec['St'][0,1]**2)), 'r.-', linewidth=5)
ax3.set_xlabel('Delay [ns]', fontsize='15')

ax4.plot(n_q0, h_q0[0], 'o', label='PMT{} - data'.format(pmts[0]))
ax4.plot(n_q0, h_q0[1], 'o', label='PMT{} - data'.format(pmts[1]))
ax4.plot(n_q0, np.sum(h_q0[0])*q0_model(n_q0, rec['q0'][0,0]), '+', label='PMT{} - model'.format(pmts[0]))
ax4.plot(n_q0, np.sum(h_q0[1])*q0_model(n_q0, rec['q0'][0,1]), '+', label='PMT{} - model'.format(pmts[1]))
ax4.set_xlabel('Dark PEs', fontsize='15')

ax3.legend(fontsize=15)
ax4.legend(fontsize=15)
ax4.set_yscale('log')
# plt.subplots_adjust(hspace=0)
plt.show()