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


rec1=np.recarray(1, dtype=[
    ('NQ', 'f8', len(pmts)),
    ('T', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('R', 'f8', len(pmts)),
    ('F', 'f8', 1),
    ('Tf', 'f8', 1),
    ('Ts', 'f8', 1),
    ])


rec2=np.recarray(1, dtype=[
    ('NQ', 'f8', len(pmts)),
    ('T', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('R', 'f8', len(pmts)),
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


rec1[0]=([34.02756583, 35.01526499], [39.90433706, 39.52340932], [1.01172871, 1.08035508], [0,0], 0.07143648, 0.19729812, 36.56672071)
rec2[0]=([36.31467141, 37.28609595], [45.38143879, 45.07967551], [0.97639089, 1.0867154 ], [0.06456251, 0.07011971], 0.7323385, 30.55506336, 99.99999548)


P=np.dstack((np.identity(100), np.identity(100)))
m1=Model(rec1['NQ'][0], rec1['T'][0], rec1['R'][0], rec1['F'][0], rec1['Tf'][0], rec1['Ts'][0], rec1['St'][0], [0,0], P)
# s1=Sim(rec1['NQ'][0], rec1['T'][0], 5, rec1['R'][0], rec1['F'][0], rec1['Tf'][0], rec1['Ts'][0], rec1['St'][0], [0,0], [0,0], [1e-6, 1e-6], [1e-6, 1e-6], [0,0])

m2=Model(rec2['NQ'][0], rec2['T'][0], rec2['R'][0], rec2['F'][0], rec2['Tf'][0], rec2['Ts'][0], rec2['St'][0], [0,0], P)
# s2=Sim(rec2['NQ'][0], rec2['T'][0], 5, rec2['R'][0], rec2['F'][0], rec2['Tf'][0], rec2['Ts'][0], rec2['St'][0], [0,0], [0,0], [1e-6, 1e-6], [1e-6, 1e-6], [0,0])

# x=np.arange(200)/5
#
# fig, ((ax1, ax3), (ax2, ax4))=plt.subplots(2,2, sharex=True)
#
#
# ax1.title.set_text('Double Exp')
# ax1.plot(x[:100], smd5(np.sum(H[:,:,0].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'ko', label='Data - PMT7')
# ax1.plot(x[:100], smd5(np.sum(H[:,0,0])*np.sum(m1[:,:,0].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'r.-', label='2 exp model', linewidth=3)
# ax1.plot(x[:100], smd5(np.sum(H[:,0,0])*np.sum(s1[:,:,0].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'g.-', label='2 exp simulation', linewidth=3)
#
# ax2.plot(x[:100], smd5(np.sum(H[:,:,1].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'ko', label='Data - PMT8')
# ax2.plot(x[:100], smd5(np.sum(H[:,0,1])*np.sum(m1[:,:,1].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'r.-', label='2 exp model', linewidth=3)
# ax2.plot(x[:100], smd5(np.sum(H[:,0,1])*np.sum(s1[:,:,1].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'g.-', label='2 exp simulation', linewidth=3)
#
# ax1.legend(fontsize=15)
# ax2.legend(fontsize=15)
# ax2.set_xlabel('Time [ns]', fontsize='15')
#
# ax3.title.set_text(r'$\delta(t)+$'+'Double Exp')
# ax3.plot(x[:100], smd5(np.sum(H[:,:,0].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'ko', label='Data - PMT7')
# ax3.plot(x[:100], smd5(np.sum(H[:,0,0])*np.sum(m2[:,:,0].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'r.-', label=r'$\delta(t)+$'+' 2 exp model', linewidth=3)
# ax3.plot(x[:100], smd5(np.sum(H[:,0,0])*np.sum(s2[:,:,0].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'g.-', label=r'$\delta(t)+$'+' 2 exp simulation', linewidth=3)
#
# ax4.plot(x[:100], smd5(np.sum(H[:,:,1].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'ko', label='Data - PMT8')
# ax4.plot(x[:100], smd5(np.sum(H[:,0,1])*np.sum(m2[:,:,1].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'r.-', label=r'$\delta(t)+$'+' 2 exp model', linewidth=3)
# ax4.plot(x[:100], smd5(np.sum(H[:,0,1])*np.sum(s2[:,:,1].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'g.-', label=r'$\delta(t)+$'+' 2 exp simulation', linewidth=3)
#
# ax3.legend(fontsize=15)
# ax4.legend(fontsize=15)
# ax4.set_xlabel('Time [ns]', fontsize='15')
# plt.subplots_adjust(hspace=0)
# fig.text(0.04, 0.5, r'$N_{events}\sum_n nH_{ni}$', va='center', rotation='vertical', fontsize=15)


# plt.plot(delays, delay_h, 'ko', label='Delay PMT8-PMT7')
# plt.xlabel('Delay [ns]', fontsize=25)
# plt.legend(fontsize=25)
# plt.xticks(fontsize=15)

# fig, (ax1, ax2)=plt.subplots(1,2, sharey=True)
# ax1.plot(np.arange(len(spectra[0])), spectra[0], 'ko', label='Total number of PEs\nin event - PMT7')
# ax2.plot(np.arange(len(spectra[1])), spectra[1], 'ko', label='Total number of PEs\nin event - PMT8')
# plt.subplots_adjust(wspace=0)
# ax1.set_xlabel('PE', fontsize=25)
# ax2.set_xlabel('PE', fontsize=25)
# ax1.legend(fontsize=25)
# ax2.legend(fontsize=25)


fig, (ax1, ax2)=plt.subplots(1,2, sharey=True)
fig.suptitle('SPE area distribution', fontsize=20)
ax1.plot(areas[0], H_areas[0], 'ko', label='PMT7')
ax2.plot(areas[1], H_areas[1], 'ko', label='PMT8')

ax1.fill_between(areas[0][areas[0]<=0.6], y1=0, y2=H_areas[0][areas[0]<=0.6], label='PE will not be\nresolved (area<a0)', alpha=0.3)
ax2.fill_between(areas[1][areas[1]<=0.6], y1=0, y2=H_areas[1][areas[1]<=0.6], label='PE will not be\nresolved (area<a0)', alpha=0.3)

ax1.fill_between(areas[0][np.logical_and(areas[0]>=0.6, areas[0]<=1.5)], y1=0, y2=H_areas[0][np.logical_and(areas[0]>=0.6, areas[0]<=1.5)], label='PE will be resolved\nas 1PE (a0<area<1.5)', alpha=0.3)
ax2.fill_between(areas[1][np.logical_and(areas[0]>=0.6, areas[0]<=1.5)], y1=0, y2=H_areas[1][np.logical_and(areas[0]>=0.6, areas[0]<=1.5)], label='PE will be resolved\n as 1PE (a0<area<1.5)', alpha=0.3)

ax1.fill_between(areas[0][np.logical_and(areas[0]>=1.5, areas[0]<=2.5)], y1=0, y2=H_areas[0][np.logical_and(areas[0]>=1.5, areas[0]<=2.5)], label='PE will be resolved\nas 2PE (1.5<area<2.5)', alpha=0.3)
ax2.fill_between(areas[1][np.logical_and(areas[0]>=1.5, areas[0]<=2.5)], y1=0, y2=H_areas[1][np.logical_and(areas[0]>=1.5, areas[0]<=2.5)], label='PE will be resolved\nas 2PE (1.5<area<2.5)', alpha=0.3)


fig.text(0.5, 0.04, 'SPE area [mean SPE area]', ha='center', fontsize=20)
ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.legend(fontsize=20)
ax2.legend(fontsize=20)
plt.show()
