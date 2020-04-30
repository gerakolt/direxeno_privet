import numpy as np
import matplotlib.pyplot as plt
import time
import os
from fun import make_P, model_spec, model_area, rec_to_p, p_to_rec2, model_h, Model2, Model3
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
    rec=p_to_rec2(p, pmts)
    global counter
    counter+=1
    names=['NQ', 'Spe', 'N_events', 'a_pad', 'a_spe', 'a_dpe', 's_pad', 'a_trpe', 'q', 'St', 'F', 'Tf', 'Ts', 'R']
    for name in names:
        if np.any(rec[name]<0):
            return 1e10*(1-np.amin(rec[name]))
    names=['m_pad', 'a0', 'q', 'F', 'R']
    for name in names:
        if np.any(rec[name]>1):
            return 1e10*(np.amax(rec[name]))

    l1=0
    l2=0
    l3=0
    l4=0
    l5=0
    P=[]
    for i in range(len(pmts)):
        a_spec=model_area(areas[i][rng_areas[i]], rec['m_pad'][0,i], rec['s_pad'][0,i], rec['Spe'][0,i], rec['a_pad'][0,i], rec['a_spe'][0,i], rec['a_dpe'][0,i], rec['a_trpe'][0, i])

        P.append(make_P(rec['Spe'][0,i], rec['s_pad'][0,i], rec['a0'][0,i], rec['q'][0,i]))

        if np.any(P[i]>1):
            return 1e10*np.amax(P[i])
            # print('P>1')
            # print(rec['Spe'][0,i], rec['s_pad'][0,i], rec['a0'][0,i], rec['q'][0,i])
            # print(np.amax(P[i]))
            # sys.exit()
        if np.any(P[i]<0):
            return 1e10*(1-np.amin(P[i]))
            # print('P<0')
            # print(rec['Spe'][0,i], rec['s_pad'][0,i], rec['a0'][0,i], rec['q'][0,i])
            # print(np.amin(P[i]))
            # sys.exit()

        spec=model_spec(ns, rec['NQ'][0,i], P[i])

        model=rec['N_events'][0,i]*spec
        data=H_specs[i][ns]
        L1=len(model)
        for j in range(L1):
            if model[j]>0 and data[j]<=0:
                l1-=model[j]-data[j]
            elif model[j]<=0 and data[j]>0:
                return 1e10*(data[j]-model[j])
            elif model[j]==0 and data[j]==0:
                l1+=1
            else:
                l1+=data[j]*np.log(model[j])-data[j]*np.log(data[j])+data[j]-model[j]

        model=a_spec
        data=H_areas[i][rng_areas[i]]
        L2=len(model)
        for j in range(L2):
            if model[j]>0 and data[j]<=0:
                l2-=model[j]-data[j]
            elif model[j]<=0 and data[j]>0:
                return 1e10*(data[j]-model[j])
            elif model[j]==0 and data[j]==0:
                l2+=1
            else:
                l2+=data[j]*np.log(model[j])-data[j]*np.log(data[j])+data[j]-model[j]

        # model=rec['q'][0,i]*NH[i]
        # data=NHC[i]
        # if model>0 and data<=0:
        #     l3-=model-data
        # elif model<=0 and data>0:
        #     return 1e10*(data-model)
        # elif model==0 and data==0:
        #     l3+=1
        # else:
        #     l3+=data*np.log(model)-data*np.log(data)+data-model
        #
        # da=areas[i][1]-areas[i][0]
        # da=1
        # model=(rec['a_pad'][0,i]*rec['q'][0,i]/da*np.sqrt(2*np.pi)*rec['s_pad'][0,i]+
        #     rec['a_spe'][0,i]/da*np.sqrt(2*np.pi*(rec['Spe'][0,i]**2+rec['s_pad'][0,i]**2))*0.5*(1+erf((1-rec['a0'][0,i])/(np.sqrt(2*(rec['s_pad'][0,i]**2+rec['Spe'][0,i]**2)))))+
        #         rec['a_dpe'][0,i]/da*np.sqrt(2*np.pi*(2*rec['Spe'][0,i]**2+rec['s_pad'][0,i]**2))*0.5*(1+erf((2-rec['a0'][0,i])/(np.sqrt(2*(rec['s_pad'][0,i]**2+2*rec['Spe'][0,i]**2)))))+
        #             rec['a_trpe'][0,i]/da*np.sqrt(2*np.pi*(3*rec['Spe'][0,i]**2+rec['s_pad'][0,i]**2))*0.5*(1+erf((3-rec['a0'][0,i])/(np.sqrt(2*(rec['s_pad'][0,i]**2+3*rec['Spe'][0,i]**2))))))
        # data=NAC[i]
        # if model>0 and data<=0:
        #     l4-=model-data
        # elif model<=0 and data>0:
        #     return 1e10*(data-model)
        # elif model==0 and data==0:
        #     l4+=1
        # else:
        #     l4+=data*np.log(model)-data*np.log(data)+data-model

    temporal=Model3(rec['NQ'][0], rec['R'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0], P)

    for i in range(len(pmts)):
        model=np.sum(H[:,0,i])*np.ravel(temporal[i])
        if np.any(np.isnan(model)) or np.any(np.isinf(model)):
            print('model is nan or inf')
            print(rec['N_events'][0,i] ,rec['NQ'][0,i], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0, i])
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


    La.append(l1/C1)
    Lb.append(l2/C2)
    Lc.append(l3/C3)
    Ld.append(l4/C4)
    Le.append(l5/C5)
    np.savez('Ls2', L1=La, L2=Lb, L3=Lc, L4=Ld, L5=Le, rec=rec)
    # l=l1/(L1*A1)+l2/(L2*A2)+l3/A3+l4/A4+l5/(L5*A5)
    l=l1/C1+l2/C2+l3/C3+l4/C4+l5/C5
    if counter%(1)==0:
        print('!!!!!!!!!!! counter=', counter, 'params=', len(p), 'iteration=', int(counter/(len(p)+1)), 'fanc=',-l)
        print(rec)

    return -l



rec=np.recarray(1, dtype=[
    ('NQ', 'f8', len(pmts)),
    ('Spe', 'f8', len(pmts)),
    ('N_events', 'f8', len(pmts)),
    ('a_pad', 'f8', len(pmts)),
    ('a_spe', 'f8', len(pmts)),
    ('a_dpe', 'f8', len(pmts)),
    ('a_trpe', 'f8', len(pmts)),
    ('s_pad', 'f8', len(pmts)),
    ('m_pad', 'f8', len(pmts)),
    ('q', 'f8', len(pmts)),
    ('a0', 'f8', len(pmts)),
    ('St', 'f8', len(pmts)),
    ('F', 'f8', 1),
    ('R', 'f8', 1),
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


rec[0]=([58.35087849, 95.66295068], [1.36412889e-01, 3.41811896e-14], [16865.10050963, 40639.07673179], [ 81851.4584917 , 131637.79773999], [11023.2700384 , 15495.73700787], [ 517.18694123, 1518.44150382],
    [ 0.0510583 , 46.89201762], [0.19648238, 0.08742664], [0.01571042, 0.00911745], [0.03121903, 0.00203122], [-2.17299086e-07, -2.42058830e-04], [1.51991036, 0.46045466], 0.004, 0.00497239, 1.70134116, 56.27437723)

rec[0]=([178.76689639, 221.51715015], [2.65931760e-03, 3.17837205e-14], [15910.2593387 , 45944.48130978], [ 66416.66906996, 254706.05202283], [21256.11585842,   312.74406905], [1122.1776688 , 1124.30928728],
 [0.0591462 , 5.22745089], [0.19602849, 0.13185204], [-0.02699918, -0.00457897], [0.11140461, 0.00184518], [8.49794737e-08, 7.46098333e-05], [1.01365252, 1.13682731], 3.94840222e-05, 0.02072487, 0.07147053, 66.79265899)
print(L(rec_to_p(rec)))
# p=minimize(L, rec_to_p(rec), method='Nelder-Mead', options={'disp':True, 'maxfev':100000})
# print(p_to_rec2(p.x, pmts))
# rec=p_to_rec2(p.x, pmts)


spec=[]
h_area=[]
h_spe=[]
h_dpe=[]
h_trpe=[]
nac=[]
nhc=[]
P=[]
temporal=np.zeros(np.shape(H))
for i in range(len(pmts)):
    nhc.append(rec['q'][0,i]*NH[i])
    da=areas[i][1]-areas[i][0]
    da=1
    nac.append(rec['a_pad'][0,i]*rec['q'][0,i]/da*np.sqrt(2*np.pi)*rec['s_pad'][0,i]+
        rec['a_spe'][0,i]/da*np.sqrt(2*np.pi*(rec['Spe'][0,i]**2+rec['s_pad'][0,i]**2))*0.5*(1+erf((1-rec['a0'][0,i])/(np.sqrt(2*(rec['s_pad'][0,i]**2+rec['Spe'][0,i]**2)))))+
            rec['a_dpe'][0,i]/da*np.sqrt(2*np.pi*(2*rec['Spe'][0,i]**2+rec['s_pad'][0,i]**2))*0.5*(1+erf((2-rec['a0'][0,i])/(np.sqrt(2*(rec['s_pad'][0,i]**2+2*rec['Spe'][0,i]**2)))))+
                rec['a_trpe'][0,i]/da*np.sqrt(2*np.pi*(3*rec['Spe'][0,i]**2+rec['s_pad'][0,i]**2))*0.5*(1+erf((3-rec['a0'][0,i])/(np.sqrt(2*(rec['s_pad'][0,i]**2+3*rec['Spe'][0,i]**2))))))
    h_area.append(model_area(areas[i][rng_areas[i]], rec['m_pad'][0,i], rec['s_pad'][0,i], rec['Spe'][0,i], rec['a_pad'][0,i], rec['a_spe'][0,i], rec['a_dpe'][0,i], rec['a_trpe'][0, i]))
    h_spe.append(model_area(areas[i][rng_areas[i]], rec['m_pad'][0,i], rec['s_pad'][0,i], rec['Spe'][0,i], 0, rec['a_spe'][0,i], 0, 0))
    h_dpe.append(model_area(areas[i][rng_areas[i]], rec['m_pad'][0,i], rec['s_pad'][0,i], rec['Spe'][0,i], 0, 0, rec['a_dpe'][0,i], 0))
    h_trpe.append(model_area(areas[i][rng_areas[i]], rec['m_pad'][0,i], rec['s_pad'][0,i], rec['Spe'][0,i], 0, 0, 0, rec['a_trpe'][0, i]))

    P.append(make_P(rec['Spe'][0,i], rec['s_pad'][0,i], rec['a0'][0,i], rec['q'][0,i]))

    if np.any(P[i]>=1):
        print('P>1')
        print(rec['Spe'][0,i], rec['s_pad'][0,i], rec['a0'][0,i], rec['q'][0,i])
        print(np.amax(P[i]))
        sys.exit()
    if np.any(P[i]<0):
        print('P<0')
        print(rec['Spe'][0,i], rec['s_pad'][0,i], rec['a0'][0,i], rec['q'][0,i])
        print(np.amin(P[i]))
        sys.exit()

    m1=model_spec(ns, rec['NQ'][0, i], P[i])
    spec.append(rec['N_events'][0, i]*m1)

temporal=Model3(rec['NQ'][0], rec['R'][0], rec['F'][0], rec['Tf'][0], rec['Ts'][0], rec['St'][0], P)

x=np.arange(1000)/5
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6))=plt.subplots(2,3)

ax1.plot(x, np.mean(H[:,:,0].T*np.arange(np.shape(H)[0]), axis=1), 'k.')
ax1.plot(x, np.sum(H[:,0,0])*np.mean(temporal[:,:,0].T*np.arange(np.shape(H)[0]), axis=1), 'r.-')

ax2.plot(H_specs[0], 'k.')
ax2.plot(ns, H_specs[0][ns], 'r.')
ax2.plot(ns, spec[0], 'g.-')
ax2.plot(ns, rec['N_events'][0,0]*(0.91*poisson.pmf(ns, rec['NQ'][0,0])+0.09*poisson.pmf(ns, 136/122*rec['NQ'][0,0])), 'b.-')


ax3.plot(areas[0], H_areas[0], 'k.', label='NAC={}, NHC={}'.format(NAC[0], NHC[0]))
ax3.plot(areas[0][rng_areas[0]], h_area[0], 'g.-', label='nac={}, nhc={}'.format(nac[0], nhc[0]))
ax3.plot(areas[0][rng_areas[0]], h_spe[0], '.-', label='SPE')
ax3.plot(areas[0][rng_areas[0]], h_dpe[0], '.-', label='DPE')
ax3.plot(areas[0][rng_areas[0]], h_trpe[0], '.-', label='TrPE')
ax3.set_yscale('log')
ax3.set_ylim(1,7000)
ax3.legend()

ax4.plot(x, np.mean(H[:,:,1].T*np.arange(np.shape(H)[0]), axis=1), 'k.')
ax4.plot(x, np.sum(H[:,0,1])*np.mean(temporal[:,:,1].T*np.arange(np.shape(H)[0]), axis=1), 'r.-')


ax5.plot(H_specs[1], 'k.')
ax5.plot(ns, H_specs[1][ns], 'r.')
ax5.plot(ns, spec[1], 'g.-')
ax5.plot(ns, rec['N_events'][0,1]*(0.91*poisson.pmf(ns, rec['NQ'][0,1])+0.09*poisson.pmf(ns, 136/122*rec['NQ'][0,1])), 'b.-')


ax6.plot(areas[1], H_areas[1], 'k.', label='NAC={}, NHC={}'.format(NAC[1], NHC[1]))
ax6.plot(areas[1][rng_areas[1]], h_area[1], 'g.-', label='nac={}, nhc={}'.format(nac[1], nhc[1]))
ax6.plot(areas[1][rng_areas[1]], h_spe[1], '.-', label='SPE')
ax6.plot(areas[1][rng_areas[1]], h_dpe[1], '.-', label='DPE')
ax6.plot(areas[1][rng_areas[1]], h_trpe[1], '.-', label='TrPE')
ax6.set_ylim(1,7000)
ax6.set_yscale('log')
ax6.legend()



plt.show()
