import numpy as np
import matplotlib.pyplot as plt
import time
import os
from fun import make_P, model_spec, model_area, rec_to_p, p_to_rec3, model_h, Model2, Model3, Model4
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

La=[]
Lb=[]
Lc=[]
Ld=[]
Le=[]
A1=20
A2=4
A3=0.5
A4=800
A5=8000
C1=1
C2=1
C3=1
C4=1
C5=1
counter=0
def L(p):
    rec=p_to_rec3(p, pmts)
    global counter
    counter+=1
    names=['NQ', 'Spe', 'q', 'St', 'F', 'Tf', 'Ts', 'R', 's_pad']
    for name in names:
        if np.any(rec[name]<0):
            return 1e10*(1-np.amin(rec[name]))
    names=['a0', 'q', 'F', 'R', 's_pad']
    for name in names:
        if np.any(rec[name]>1):
            return 1e10*(np.amax(rec[name]))
    if rec['Ts']<30:
        return 1e10*(30-rec['Ts'])
    if rec['Tf']>30:
        return 1e10*rec['Tf']

    l5=0
    P=[]
    for i in range(len(pmts)):
        P.append(make_P(rec['Spe'][0,i], rec['s_pad'][0,i], rec['a0'][0,i], rec['q'][0,i]))

        if np.any(P[i]>1):
            return 1e10*np.amax(P[i])
        if np.any(P[i]<0):
            return 1e10*(1-np.amin(P[i]))

    temporal=Model3(rec['NQ'][0], rec['R'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0], P)

    for i in range(len(pmts)):
        model=np.sum(H[:,0,i])*np.ravel(temporal[:,:,i])
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



    np.savez('Ls', L5=l5, rec=rec)
    l=l5/C5
    if counter%(1)==0:
        print('!!!!!!!!!!! counter=', counter, 'params=', len(p), 'iteration=', int(counter/(len(p)+1)), 'fanc=',-l)
        print(rec)

    return -l



rec=np.recarray(1, dtype=[
    ('NQ', 'f8', len(pmts)),
    ('Spe', 'f8', len(pmts)),
    ('s_pad', 'f8', len(pmts)),
    ('q', 'f8', len(pmts)),
    ('a0', 'f8', len(pmts)),
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

# rec[0]=([33.7875632 , 34.80096488], [6.46976275e-10, 1.39489388e-01], [-7.88083646e-02,  5.86534391e-06], [8.55583802e-04, 2.78236912e-12], [0.08661196, 0.21998049], [0.5931011 , 1.08010902], 0.04769602, 1.45422862, 38.0569891)
rec[0]=([33.87528016, 34.77410855], [2.16617982e-09, 1.39511566e-01], [2.29951721e-06, 4.75900065e-06], [7.84190965e-04, 9.44841442e-13], [0.99825506, -0.06858682], [0.61344929, 0.89483881],
    [7.18438177e-10, 1.22125763e-02], 0.04258672, 2.01546052, 38.2516436)

rec[0]=([33.87528016, 34.77410855], [2.16617982e-09, 1.39511566e-01], [2.29951721e-06, 4.75900065e-06], [7.84190965e-04, 9.44841442e-13], [ 0.99825506, -0.06858682], [0.15, 0.15],
    [0.01, 0.01], 0.04258672, 2, 38.2516436)
# print(L(rec_to_p(rec)))
# p=minimize(L, rec_to_p(rec), method='Nelder-Mead', options={'disp':True, 'maxfev':100000})
# print(p_to_rec3(p.x, pmts))
# rec=p_to_rec3(p.x, pmts)


P=[]
temporal=np.zeros(np.shape(H))
temporal_d=np.zeros(np.shape(H))
temporal_f=np.zeros(np.shape(H))
temporal_s=np.zeros(np.shape(H))

for i in range(len(pmts)):

    P.append(make_P(rec['Spe'][0,i], rec['s_pad'][0,i], rec['a0'][0,i], rec['q'][0,i]))
    # P.append(np.identity(100))
    if np.any(P[i]>=1):
        print('P>1')
        print(rec['Spe'][0,i], rec['s_pad'][0,i], rec['a0'][0,i], rec['q'][0,i])
        print(np.amax(P[i]))
    if np.any(P[i]<0):
        print('P<0')
        print(rec['Spe'][0,i], rec['s_pad'][0,i], rec['a0'][0,i], rec['q'][0,i])
        print(np.amin(P[i]))

# temporal=Model3(rec['NQ'][0], rec['R'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0], P)
t_d=Model4(rec['NQ'][0], rec['R'][0], 0, 0, rec['Tf'][0], rec['Ts'][0], rec['St'][0], P)




x=np.arange(1000)/5
fig, (ax1, ax4)=plt.subplots(2,1)

ax1.plot(x, np.mean(H[:,:,0].T*np.arange(np.shape(H)[0]), axis=1), 'k.-')
# ax1.plot(x, np.sum(H[:,0,0])*np.mean(temporal[:,:,0].T*np.arange(np.shape(H)[0]), axis=1), '.-', label='full')
ax1.plot(x, np.sum(H[:,0,0])*np.mean(t_d[:,:,0].T*np.arange(np.shape(H)[0]), axis=1), '.-', label='d')

ax1.legend()

ax4.plot(x, np.mean(H[:,:,1].T*np.arange(np.shape(H)[0]), axis=1), 'k.-')
ax4.set_xlabel('Time [ns]', fontsize=25)
# ax4.plot(x, np.sum(H[:,0,0])*np.mean(temporal[:,:,1].T*np.arange(np.shape(H)[0]), axis=1), '.-', label='full')
ax4.plot(x, np.sum(H[:,0,0])*np.mean(t_d[:,:,1].T*np.arange(np.shape(H)[0]), axis=1), '.-', label='d')
# ax4.plot(x, np.sum(H[:,0,0])*np.mean(temporal_d[:,:,1].T*np.arange(np.shape(H)[0]), axis=1), '.-', label='d')
# ax4.plot(x, np.sum(H[:,0,0])*np.mean(temporal_f[:,:,1].T*np.arange(np.shape(H)[0]), axis=1), '.-', label='f')
# ax4.plot(x, np.sum(H[:,0,0])*np.mean(temporal_s[:,:,1].T*np.arange(np.shape(H)[0]), axis=1), '.-', label='s')
ax4.legend()

plt.show()
