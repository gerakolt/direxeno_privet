import numpy as np
import matplotlib.pyplot as plt
import time
import os
from fun import make_P, model_spec, model_area, rec_to_p, p_to_rec4, model_h, Model2, Model3, Model4
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf
import warnings
# warnings.filterwarnings('error')




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

H_specs=[]
for i, pmt in enumerate(pmts):
    H_specs.append(np.histogram(np.sum(rec['h'][:,:,i], axis=1), bins=np.arange(150)-0.5)[0])

ns=np.arange(30,40)
H_areas=[]
areas=[]
rng_areas=[]
NAC=[]
NH=[]
NHC=[]
for i, pmt in enumerate(pmts):
    path='/home/gerak/Desktop/DireXeno/190803/pulser/NEWPMT{}/'.format(pmt)
    data=np.load(path+'areas.npz')
    H_areas.append(data['H_areas'])
    areas.append(data['areas'])
    rng_areas.append(data['rng_area'])
    NAC.append(data['NAC'])
    NH.append(data['NH'])
    NHC.append(data['NHC'])


counter=0
def L(p):
    rec=p_to_rec4(p, pmts)
    global counter
    counter+=1
    names=['NQ', 'St', 'F', 'Tf', 'Ts', 'R']
    for name in names:
        if np.any(rec[name]<0):
            return 1e10*(1-np.amin(rec[name]))
    names=['F', 'R']
    for name in names:
        if np.any(rec[name]>1):
            return 1e10*(np.amax(rec[name]))
    if rec['Ts']<30:
        return 1e10*(30-rec['Ts'])
    if rec['Tf']>30:
        return 1e10*rec['Tf']

    l5=0
    temporal=Model3(rec['NQ'][0], rec['R'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0], [np.identity(100), np.identity(100)])

    for i in range(len(pmts)):
        model=rec['N'][0,i]*np.ravel(temporal[:,:,i])
        if np.any(np.isnan(model)) or np.any(np.isinf(model)):
            print('model is nan or inf')
            print(rec['NQ'][0,i], rec['R'][0,i], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0, i])
            sys.exit()
        data=np.ravel(H[:,:,i])
        L5=len(model)
        for j in range(L5):
            if model[j]>0 and data[j]<=0:
                l5-=model[j]-data[j]
            elif model[j]<=0 and data[j]>0:
                return 1e10*(data[j]-model[j])
            elif model[j]==0 and data[j]==0:
                l5+=1
            else:
                try:
                    l5+=data[j]*np.log(model[j])-data[j]*np.log(data[j])+data[j]-model[j]
                except:
                    print(data[j], model[j])
                    print(rec['NQ'][0,i], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0, i])
                    sys.exit()


    if counter%(1)==0:
        print('!!!!!!!!!!! counter=', counter, 'params=', len(p), 'iteration=', int(counter/(len(p)+1)), 'fanc=',-l5)
        print(rec)

    return -l5



rec=np.recarray(1, dtype=[
    ('N', 'f8', len(pmts)),
    ('NQ', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('R', 'f8', len(pmts)),
    ('F', 'f8', 1),
    ('Tf', 'f8', 1),
    ('Ts', 'f8', 1),
])



# rec['NQ'][0]=(14.40319786,  8.02606961)
# rec['Spe'][0]=(0.22535762, 0.25)
# rec['N_events'][0]=(36372.65966822, 20250.85668553)
# rec['a_pad'][0]=(91266.47300635, 106350.70453041)
# rec['a_spe'][0]=(11500, 12000)
# rec['a_dpe'][0]=(1185.51040432, 572.80248514)
# rec['a_trpe'][0]=(85.51040432, 85.80248514)
# rec['s_pad'][0]=(0.293555923, 0.29589045)
# rec['m_pad'][0]=(0.00025055, -0.00298575)
# rec['q'][0]=(0.01, 0.01)
# rec['a0'][0]=(0.01, 0.01)

rec[0]=([np.sum(H[:,0,0]), np.sum(H[:,0,1])], [34.70533116, 34.83540734], [0.53103084, 1.04593437], [0.01, 0.01], 0.05171045, 2.14752407, 39.4244497)
p=minimize(L, rec_to_p(rec), method='Nelder-Mead', options={'disp':True, 'maxfev':100000})
rec=p_to_rec4(p.x, pmts)



t_d=Model3(rec['NQ'][0], rec['R'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0], [np.identity(100), np.identity(100)])




x=np.arange(1000)/5
fig, (ax1, ax4)=plt.subplots(2,1, sharex=True)

ax1.plot(x[:30*5], np.mean(H[:,:,0].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], 'ko', label='Data')
ax1.plot(x[:30*5], rec['N'][0,0]*np.mean(t_d[:,:,0].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], '.-', label='2 exp model', linewidth=3)
# ax1.plot(x[:30*5], np.sum(H[:,0,0])*np.mean(t[:,:,0].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], '.-', label='', linewidth=3)
ax1.legend(fontsize=15)

ax4.set_xlabel('Time [ns]', fontsize=25)
ax4.plot(x[:30*5], np.mean(H[:,:,1].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], 'ko', label='Data')
ax4.plot(x[:30*5], rec['N'][0,1]*np.mean(t_d[:,:,1].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], '.-', label='2 exp model', linewidth=3)
# ax4.plot(x[:30*5], np.sum(H[:,0,1])*np.mean(t[:,:,1].T*np.arange(np.shape(H)[0]), axis=1)[:30*5], '.-', label='', linewidth=3)
ax4.legend(fontsize=15)

plt.subplots_adjust(hspace=0)
plt.show()
