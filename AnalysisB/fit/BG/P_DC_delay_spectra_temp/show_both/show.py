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


path='/home/gerak/Desktop/DireXeno/190803/BG/EventRecon/'
rec=np.load(path+'recon.npz')['rec']
h=rec['h']
spectrum=np.zeros((100, len(pmts)))
spec_PEs=np.arange(30,40)
for i in range(len(pmts)):
    spectrum[:,i]=np.histogram(np.sum(h[:,:,i], axis=1), bins=np.arange(101)-0.5)[0]

data=np.load(path+'H0.npz')
H=data['H']
G=data['G']
spectra=data['spectra']
PEs=np.arange(len(spectra[0]))


rec1=np.recarray(1, dtype=[
    ('NQ', 'f8', len(pmts)),
    ('T', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('q0', 'f8', len(pmts)),
    ('a0', 'f8', len(pmts)),
    ('a_pad', 'f8', len(pmts)),
    ('a_spe', 'f8', len(pmts)),
    ('a_dpe', 'f8', len(pmts)),
    ('a_trpe', 'f8', len(pmts)),
    ('m_pad', 'f8', len(pmts)),
    ('Spad', 'f8', len(pmts)),
    ('Spe', 'f8', len(pmts)),
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
    ('q0', 'f8', len(pmts)),
    ('a0', 'f8', len(pmts)),
    ('a_pad', 'f8', len(pmts)),
    ('a_spe', 'f8', len(pmts)),
    ('a_dpe', 'f8', len(pmts)),
    ('a_trpe', 'f8', len(pmts)),
    ('m_pad', 'f8', len(pmts)),
    ('Spad', 'f8', len(pmts)),
    ('Spe', 'f8', len(pmts)),
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


rec1[0]=([34.26568683, 34.7510686 ], [33.33421224, 32.97278574], [1.04143771, 1.07316965], [2.38170934e-05, 1.80266093e-05], [8.69568295e-04, 1.04088444e-06], [76444.32752178, 80305.18709964], [13406.82106563, 12309.65767153],
 [1340.15339863,  839.08110928], [ 8.8474879 , 14.79308662], [-1.16495364e-03, -6.95042298e-05], [0.24693002, 0.22431859], [3.01472501e-03, 1.57832917e-10], 0.07043034, 0.18587841, 36.85577814, 408.93094263)
rec2[0]=([34.03822962, 34.92790393], [36.85313527, 36.68593916], [0.70746969, 0.95376529], [2.79037134e-12, 7.43713468e-03], [4.98611299e-05, 2.39955353e-05], [2.51283888e-03, 6.70268148e-07], [74940.7599723 , 79719.34352622],
 [14586.17209451, 11292.02216426], [1696.51693551,  848.71451248], [85.16514545, 59.93128527], [-0.07018456, -0.00026282], [0.25880411, 0.22653098], [4.41810781e-03, 8.09144120e-26], 0.05258429, 1.09214088, 36.2361104, 408.93094263)

x=np.arange(1000)/5
fig, (ax1, ax2)=plt.subplots(1,2, sharex=True)
fig.suptitle('Temporal structure of BG (few PE events)', fontsize=25)
ax1.plot(x[:100], smd5(np.sum(H[:,:,0].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'k-.', label='Data - PMT7')
ax2.plot(x[:100], smd5(np.sum(H[:,:,1].T*np.arange(np.shape(H)[0]), axis=1))[:100], 'k-.', label='Data - PMT8')
ax1.legend(fontsize=25)
ax2.legend(fontsize=25)

fig.text(0.5, 0.04, 'Time [ns]', ha='center', fontsize=20)

# P=make_P(rec1['a0'][0], rec1['Spad'][0], rec1['Spe'][0], rec1['m_pad'][0])
# m1=Model(rec1['NQ'][0], rec1['T'][0], [0,0], rec1['F'][0], rec1['Tf'][0], rec1['Ts'][0], rec1['St'][0], rec1['q0'][0], P)
# m_area1=model_area(areas, rec1['m_pad'][0], rec1['a_pad'][0], rec1['a_spe'][0], rec1['a_dpe'][0], rec1['a_trpe'][0], rec1['Spad'][0], rec1['Spe'][0])
# s1=Sim(rec1['NQ'][0], rec1['T'][0], 5, [0,0], rec1['F'][0], rec1['Tf'][0], rec1['Ts'][0], rec1['St'][0], rec1['q0'][0], rec1['a0'][0], rec1['Spad'][0], rec1['Spe'][0], rec1['m_pad'][0])
#
# P=make_P(rec2['a0'][0], rec2['Spad'][0], rec2['Spe'][0], rec2['m_pad'][0])
# m2=Model(rec2['NQ'][0], rec2['T'][0], rec2['R'][0], rec2['F'][0], rec2['Tf'][0], rec2['Ts'][0], rec2['St'][0], rec2['q0'][0], P)
# m_area2=model_area(areas, rec2['m_pad'][0], rec2['a_pad'][0], rec2['a_spe'][0], rec2['a_dpe'][0], rec2['a_trpe'][0], rec2['Spad'][0], rec2['Spe'][0])
# s2=Sim(rec2['NQ'][0], rec2['T'][0], 5, rec2['R'][0], rec2['F'][0], rec2['Tf'][0], rec2['Ts'][0], rec2['St'][0], rec2['q0'][0], rec2['a0'][0], rec2['Spad'][0], rec2['Spe'][0], rec2['m_pad'][0])
#
# x=np.arange(1000)/5
#
# fig, ((ax1, ax3), (ax2, ax4))=plt.subplots(2,2, sharex=True)
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
#
# plt.figure()
# plt.plot(delays, delay_h, 'ko', label='Delay PMT8-PMT7')
# plt.plot(delays[rng_delay], rec1['a_delay'][0]*np.exp(-0.5*(delays[rng_delay]-rec1['T'][0,1]+rec1['T'][0,0])**2/(rec1['St'][0,0]**2+rec1['St'][0,1]**2))/np.sqrt(2*np.pi*(rec1['St'][0,0]**2+rec1['St'][0,1]**2)), '.-',
#     linewidth=5, label='2 exp model')
# plt.plot(delays[rng_delay], rec2['a_delay'][0]*np.exp(-0.5*(delays[rng_delay]-rec2['T'][0,1]+rec2['T'][0,0])**2/(rec2['St'][0,0]**2+rec2['St'][0,1]**2))/np.sqrt(2*np.pi*(rec2['St'][0,0]**2+rec2['St'][0,1]**2)), '.-',
#     linewidth=5, label=r'$\delta(t)+$'+'2 exp model')
# plt.xlabel('Delay [ns]', fontsize=25)
# plt.legend(fontsize=25)
# plt.xticks(fontsize=15)
#
# fig, (ax1, ax2)=plt.subplots(1,2, sharey=True)
# ax1.plot(np.arange(len(spectra[0])), spectra[0], 'ko', label='Total number of PEs\nin event - PMT7')
# ax2.plot(np.arange(len(spectra[1])), spectra[1], 'ko', label='Total number of PEs\nin event - PMT8')
#
# spectra_rng=np.nonzero(np.logical_and(PEs>PEs[np.argmax(spectra[0])-5], PEs<PEs[np.argmax(spectra[0])+5]))[0]
# model1=np.sum(H[:,0,0])*poisson.pmf(PEs, np.sum(m1[:,:,0].T*np.arange(np.shape(H)[0])))[spectra_rng]
# model2=np.sum(H[:,0,0])*poisson.pmf(PEs, np.sum(m2[:,:,0].T*np.arange(np.shape(H)[0])))[spectra_rng]
# ax1.plot(PEs[spectra_rng], model1, '.-', label='2 exp model')
# ax1.plot(PEs[spectra_rng], model2, '.-', label=r'$\delta(t)+$'+'2 exp model')
#
# spectra_rng=np.nonzero(np.logical_and(PEs>PEs[np.argmax(spectra[1])-5], PEs<PEs[np.argmax(spectra[1])+5]))[0]
# model1=np.sum(H[:,0,1])*poisson.pmf(PEs, np.sum(m1[:,:,1].T*np.arange(np.shape(H)[0])))[spectra_rng]
# model2=np.sum(H[:,0,1])*poisson.pmf(PEs, np.sum(m2[:,:,1].T*np.arange(np.shape(H)[0])))[spectra_rng]
# ax2.plot(PEs[spectra_rng], model1, '.-', label='2 exp model')
# ax2.plot(PEs[spectra_rng], model2, '.-', label=r'$\delta(t)+$'+'2 exp model')
#
# fig.subplots_adjust(wspace=0)
# ax1.set_xlabel('PE', fontsize=25)
# ax2.set_xlabel('PE', fontsize=25)
# ax1.legend(fontsize=25)
# ax2.legend(fontsize=25)
#
#
# fig, (ax1, ax2)=plt.subplots(1,2)
# fig.suptitle('SPE area distribution', fontsize=20)
# ax1.set_ylim(1,1e4)
# ax2.set_ylim(1,1e4)
#
# ax1.plot(areas[0], H_areas[0], 'ko', label='PMT7')
# ax2.plot(areas[1], H_areas[1], 'ko', label='PMT8')
#
#
# m_area1=model_area(areas, rec1['m_pad'][0], rec1['a_pad'][0], rec1['a_spe'][0], rec1['a_dpe'][0], rec1['a_trpe'][0], rec1['Spad'][0], rec1['Spe'][0])
# m_pad1=model_area(areas, rec1['m_pad'][0], rec1['a_pad'][0], [0,0], [0,0], [0,0], rec1['Spad'][0], rec1['Spe'][0])
# m_spe1=model_area(areas, rec1['m_pad'][0], [0,0], rec1['a_spe'][0], [0,0], [0,0], rec1['Spad'][0], rec1['Spe'][0])
# m_dpe1=model_area(areas, [0,0], [0,0], rec1['a_spe'][0], [0,0], rec1['a_trpe'][0], rec1['Spad'][0], rec1['Spe'][0])
# m_trpe1=model_area(areas, [0,0], [0,0], [0,0], rec1['a_dpe'][0], rec1['a_trpe'][0], rec1['Spad'][0], rec1['Spe'][0])
#
# m_area2=model_area(areas, rec2['m_pad'][0], rec2['a_pad'][0], rec2['a_spe'][0], rec2['a_dpe'][0], rec2['a_trpe'][0], rec2['Spad'][0], rec2['Spe'][0])
# m_pad2=model_area(areas, rec2['m_pad'][0], rec2['a_pad'][0], [0,0], [0,0], [0,0], rec2['Spad'][0], rec2['Spe'][0])
# m_spe2=model_area(areas, rec2['m_pad'][0], [0,0], rec2['a_spe'][0], [0,0], [0,0], rec2['Spad'][0], rec2['Spe'][0])
# m_dpe2=model_area(areas, [0,0], [0,0], rec2['a_spe'][0], [0,0], rec2['a_trpe'][0], rec2['Spad'][0], rec2['Spe'][0])
# m_trpe2=model_area(areas, [0,0], [0,0], [0,0], rec2['a_dpe'][0], rec2['a_trpe'][0], rec2['Spad'][0], rec2['Spe'][0])
#
# ax1.plot(areas[0], m_area1[0], 'r.-', label='2 exp model')
# ax2.plot(areas[1], m_area1[1], 'r.-', label='2 exp model')
# #
# ax1.plot(areas[0], m_area2[0], 'g.-', label=r'$\delta(t)+$'+'2 exp model')
# ax2.plot(areas[1], m_area2[1], 'g.-', label=r'$\delta(t)+$'+'2 exp model')
#
# # ax1.plot(areas[0], m_pad1[0], '.-', lable='padestial')
# # ax2.plot(areas[1], m_pad1[1], '.-', lable='padestial')
# #
# # ax1.plot(areas[0], m_spe1[0], '.-', lable='SPE')
# # ax2.plot(areas[1], m_spe1[1], '.-', lable='SPE')
# #
# # ax1.plot(areas[0], m_dpe1[0], '.-', lable='2PE')
# # ax2.plot(areas[1], m_dpe1[1], '.-', lable='2PE')
# #
# # ax1.plot(areas[0], m_trpe1[0], '.-', lable='3PE')
# # ax2.plot(areas[1], m_trpe1[1], '.-', lable='3PE')
#
# #
# fig.text(0.5, 0.04, 'SPE area [mean SPE area]', ha='center', fontsize=20)
# ax1.set_yscale('log')
# ax2.set_yscale('log')
# ax1.legend(fontsize=20)
# ax2.legend(fontsize=20)
#
# fig, (ax4, ax5)=plt.subplots(1,2, sharey=True)
# ax4.bar(n_q0, h_q0[0], label='PMT{} - data'.format(pmts[0]))
# ax5.bar(n_q0, h_q0[1], label='PMT{} - data'.format(pmts[1]))
# ax4.plot(n_q0, np.sum(h_q0[0])*q0_model(n_q0, rec1['q0'][0,0]), 'o', label='2 exp model')
# ax5.plot(n_q0, np.sum(h_q0[1])*q0_model(n_q0, rec1['q0'][0,1]), 'o', label=r'$\delta(t)+$'+'2 exp model')
# ax4.plot(n_q0, np.sum(h_q0[0])*q0_model(n_q0, rec2['q0'][0,0]), 'o', label='2 exp model')
# ax5.plot(n_q0, np.sum(h_q0[1])*q0_model(n_q0, rec2['q0'][0,1]), 'o', label=r'$\delta(t)+$'+'2 exp model')
# fig.text(0.5, 0.04, 'Dark PEs', ha='center', fontsize=20)
#
# ax4.legend(fontsize=15)
# ax5.legend(fontsize=15)
# ax4.set_yscale('log')
# ax5.set_yscale('log')



plt.show()
