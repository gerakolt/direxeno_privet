import numpy as np
import matplotlib.pyplot as plt
import time
import os
from fun import Model, Sim, q0_model, make_P, model_area, SubModel
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

data=np.load(path+'H2.npz')
H=np.zeros((15, 500, 2))
H=data['H2']


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
    ('a0', 'f8', len(pmts)),
    ('Spad', 'f8', len(pmts)),
    ('Spe', 'f8', len(pmts)),
    ('a_pad', 'f8', len(pmts)),
    ('a_spe', 'f8', len(pmts)),
    ('a_dpe', 'f8', len(pmts)),
    ('a_trpe', 'f8', len(pmts)),
    ('m_pad', 'f8', len(pmts)),
    ('a_spec', 'f8', len(pmts)),
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

    names=['NQ', 'F', 'Tf', 'Ts', 'T', 'R', 'q0', 'a0', 'Spad', 'Spe', 'a_pad', 'a_spe', 'a_dpe', 'a_trpe', 'a_spec']
    for name in names:
        if np.any(rec[name]<0):
            return 1e10*(1-np.amin(rec[name]))
    names=['F', 'R', 'q0', 'a0', 'Spad']
    for name in names:
        if np.any(rec[name]>1):
            return 1e10*(np.amax(rec[name]))
    if rec['Ts'][0]>100:
        return 1e10*rec['Ts'][0]
    if rec['Ts'][0]<rec['Tf'][0]:
        return 1e10*(rec['Tf'][0]-rec['Ts'][0])
    if np.any(rec['St'][0]<0.2):
        return 1e10*(1+np.abs(np.amin(rec['St'][0])))

    l=0
    m=Model(rec['NQ'][0], rec['T'][0], rec['R'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0], rec['q0'][0], make_P(rec['a0'][0], rec['Spad'][0], rec['Spe'][0], rec['m_pad'][0]))
    m_area=model_area(areas, rec['m_pad'][0], rec['a_pad'][0], rec['a_spe'][0], rec['a_dpe'][0], rec['a_trpe'][0], rec['Spad'][0], rec['Spe'][0])
    for i in range(len(pmts)):
        model=np.sum(H[:,0,i])*np.ravel(m[:,:250,i])
        if np.any(np.isnan(model)) or np.any(np.isinf(model)):
            print('model is nan or inf')
            print('NQ=', rec['NQ'][0,i], 'T=', rec['T'][0,i], 'F=', rec['F'][0,i], 'Tf=', rec['Tf'][0,i], 'Ts=', rec['Ts'][0,i], 'St=', rec['St'][0, i])
            plt.figure()
            plt.plot(np.mean(t.T*np.arange(np.shape(t)[0])), 'k.')
            plt.show()
            sys.exit()
        data=np.ravel(H[:,:250,i])
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

        model=rec['a_spec'][0,i]*poisson.pmf(spec_PEs, np.sum(m[:,:,i].T*np.arange(np.shape(H)[0])))
        data=spectrum[spec_PEs, i]
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

        model=m_area[i]
        data=H_areas[i]
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
rec[0]=([34.35316862, 34.31004615], [37.61478037, 37.60316171], [0.57704455, 0.74961068], [0.00018943, 0.01444127], [5.13313549e-05, 2.31112352e-05], [7.49295572e-05, 3.02382910e-02],
 [0.27288189, 0.24191372], [8.15874656e-03, 1.06040545e-10], [75564.47105848, 79148.06342345], [15008.51974368, 12231.14053128],
  [1660.140292  , 1171.29503664], [21.62900683,  1.74987813], [-0.07395446, -0.04018459], [34168.63357467, 36202.64114333],
  0.03662392, 1.82148065, 37.02975368, 387.83802331)
# p=minimize(L, rec_to_p(rec), method='Nelder-Mead', options={'disp':True, 'maxfev':100000})
# rec=p_to_rec(p.x)

m=Model(rec['NQ'][0], rec['T'][0], rec['R'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0], rec['q0'][0], make_P(rec['a0'][0], rec['Spad'][0], rec['Spe'][0], rec['m_pad'][0]))
dl=SubModel(rec['NQ'][0], rec['T'][0], rec['R'][0], 0, 0, rec['Tf'][0], rec['Ts'][0], rec['St'][0], rec['q0'][0], make_P(rec['a0'][0], rec['Spad'][0], rec['Spe'][0], rec['m_pad'][0]))
fst=SubModel(rec['NQ'][0], rec['T'][0], [0,0], rec['F'][0], 0, rec['Tf'][0], rec['Ts'][0], rec['St'][0], rec['q0'][0], make_P(rec['a0'][0], rec['Spad'][0], rec['Spe'][0], rec['m_pad'][0]))
slw=SubModel(rec['NQ'][0], rec['T'][0], [0,0],0 ,1-rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0], rec['q0'][0], make_P(rec['a0'][0], rec['Spad'][0], rec['Spe'][0], rec['m_pad'][0]))
# s=Sim(rec['NQ'][0], rec['T'][0], 1, rec['R'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0], rec['q0'][0], rec['a0'][0], rec['Spad'][0], rec['Spe'][0], rec['m_pad'][0])

x=np.arange(500)/5*2

fig, ((ax1, ax3), (ax2, ax4))=plt.subplots(2,2)

ax1.plot(x[:30*5], np.sum(H[:,:,0].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], 'ko', label='Data - PMT7')
ax1.plot(x[:30*5], np.sum(H[:,0,0])*np.sum(m[:,:,0].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], 'r.-', label=r'$\delta+$'+' 2 exp model', linewidth=3)
ax1.plot(x[:30*5], np.sum(H[:,0,0])*np.sum(dl[:,:,0].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], '.-', label=r'$\delta$', linewidth=3)
ax1.plot(x[:30*5], np.sum(H[:,0,0])*np.sum(fst[:,:,0].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], '.-', label='fast', linewidth=3)
ax1.plot(x[:30*5], np.sum(H[:,0,0])*np.sum(slw[:,:,0].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], '.-', label='slw', linewidth=3)

# ax1.plot(x[:30*5], np.sum(H[:,0,0])*np.sum(s[:,:,0].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], 'g.-', label=r'$\delta+$'+' 2 exp simulation', linewidth=3)

ax2.plot(x[:30*5], np.sum(H[:,:,1].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], 'ko', label='Data - PMT8')
ax2.plot(x[:30*5], np.sum(H[:,0,1])*np.sum(m[:,:,1].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], 'r.-', label=r'$\delta+$'+' 2 exp model', linewidth=3)
ax2.plot(x[:30*5], np.sum(H[:,0,1])*np.sum(dl[:,:,1].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], '.-', label=r'$\delta$', linewidth=3)
ax2.plot(x[:30*5], np.sum(H[:,0,1])*np.sum(fst[:,:,1].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], '.-', label='fast', linewidth=3)
ax2.plot(x[:30*5], np.sum(H[:,0,1])*np.sum(slw[:,:,1].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], '.-', label='slw', linewidth=3)

# ax2.plot(x[:30*5], np.sum(H[:,0,1])*np.sum(s[:,:,1].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], 'g.-', label=r'$\delta+$'+' 2 exp simulation', linewidth=3)

ax1.legend(fontsize=15)
ax2.legend(fontsize=15)
ax2.set_xlabel('Time [ns]', fontsize='15')
fig.text(0.04, 0.5, r'$N_{events}\sum_n nH_{ni}$', va='center', rotation='vertical', fontsize=15)

ax3.plot(spectrum[:,0], 'ko', label='spectrum - PMT7')
ax3.plot(spec_PEs, rec['a_spec'][0,0]*poisson.pmf(spec_PEs, np.sum(m[:,:,0].T*np.arange(np.shape(H)[0]))), 'r-.')


ax4.plot(spectrum[:,1], 'ko', label='spectrum - PMT8')
ax4.plot(spec_PEs, rec['a_spec'][0,1]*poisson.pmf(spec_PEs, np.sum(m[:,:,1].T*np.arange(np.shape(H)[0]))), 'r-.')





fig, ((ax1, ax3), (ax2, ax4))=plt.subplots(2,2)
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

m_area=model_area(areas, rec['m_pad'][0], rec['a_pad'][0], rec['a_spe'][0], rec['a_dpe'][0], rec['a_trpe'][0], rec['Spad'][0], rec['Spe'][0])
m_pad=model_area(areas, rec['m_pad'][0], rec['a_pad'][0], [0,0], [0,0], [0,0], rec['Spad'][0], rec['Spe'][0])
m_spe=model_area(areas, rec['m_pad'][0], [0,0], rec['a_spe'][0], [0,0], [0,0], rec['Spad'][0], rec['Spe'][0])
m_dpe=model_area(areas, [0,0], [0,0], rec['a_spe'][0], [0,0], rec['a_trpe'][0], rec['Spad'][0], rec['Spe'][0])
m_trpe=model_area(areas, [0,0], [0,0], [0,0], rec['a_dpe'][0], rec['a_trpe'][0], rec['Spad'][0], rec['Spe'][0])




ax1.plot(areas[0], H_areas[0], 'ko')
ax2.plot(areas[1], H_areas[1], 'ko')

ax1.plot(areas[0], m_area[0], 'r.-')
ax2.plot(areas[1], m_area[1], 'r.-')

ax1.plot(areas[0], m_pad[0], '.-')
ax2.plot(areas[1], m_pad[1], '.-')

ax1.plot(areas[0], m_spe[0], '.-')
ax2.plot(areas[1], m_spe[1], '.-')

ax1.plot(areas[0], m_dpe[0], '.-')
ax2.plot(areas[1], m_dpe[1], '.-')

ax1.plot(areas[0], m_trpe[0], '.-')
ax2.plot(areas[1], m_trpe[1], '.-')

ax1.set_yscale('log')
ax2.set_yscale('log')

ax1.set_ylim(1,1e4)
ax2.set_ylim(1,1e4)


plt.show()
