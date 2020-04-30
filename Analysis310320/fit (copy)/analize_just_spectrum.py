import numpy as np
import matplotlib.pyplot as plt
import time
import os
from fun import make_P, model_spec, model_area, rec_to_p, p_to_rec, model_h
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf




pmts=[7,8]
path='/home/gerak/Desktop/DireXeno/190803/Co57/EventRecon/'
rec=np.load(path+'recon.npz')['rec']
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
A1=15000
A2=20
A3=300
A4=160000
counter=0
rec=np.recarray(1, dtype=[
    ('Spe', 'f8', len(pmts)),
    ('a_pad', 'f8', len(pmts)),
    ('a_spe', 'f8', len(pmts)),
    ('a_dpe', 'f8', len(pmts)),
    ('a_trpe', 'f8', len(pmts)),
    ('s_pad', 'f8', len(pmts)),
    ('m_pad', 'f8', len(pmts)),
    ('q', 'f8', len(pmts)),
    ('a0', 'f8', len(pmts)),
    ])
rec[0]=([1.64079543e-01, 4.92318738e-12], [ 99530.90450788, 114510.34134549], [13478.28277049, 14287.50170371], [ 952.96900316, 1138.24575204],
    [43.01860748, 68.6686766 ], [0.31574359, 0.28125273], [0.02147097, 0.02173494], [0.01727227, 0.0016339 ], [-3.08054825e-06,  2.15951306e-05])
P=make_P(rec['Spe'][0,i], rec['s_pad'][0,i], rec['a0'][0,i], rec['q'][0,i])
def L(p):
    [NQ1, NQ2, N_events1, N_events2]=p
    NQ=[NQ1, NQ2]
    N_events=[N_events1, N_events2]
    global counter
    counter+=1
    names=['Spe', 'a_pad', 'a_spe', 'a_dpe', 's_pad', 'a_trpe', 'q']
    for name in names:
        if np.any(rec[name]<0):
            return 1e5*(1-np.amin(rec[name]))
    names=['m_pad', 'a0', 'q']
    for name in names:
        if np.any(rec[name]>1):
            return 1e5*(np.amax(rec[name]))

    l=0
    for i in range(len(pmts)):
        l1=0
        l2=0
        l3=0
        l4=0

        spec=0.91*model_spec(ns, NQ[i], P)+0.09*model_spec(ns, NQ[i]*136/122, P)

        model=N_events[i]*spec
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
        L2=1

        # model=a_spec
        # data=H_areas[i][rng_areas[i]]
        # L2=len(model)
        # for j in range(L2):
        #     if model[j]>0 and data[j]<=0:
        #         l2-=model[j]-data[j]
        #     elif model[j]<=0 and data[j]>0:
        #         return 1e10*(data[j]-model[j])
        #     elif model[j]==0 and data[j]==0:
        #         l2+=1
        #     else:
        #         l2+=data[j]*np.log(model[j])-data[j]*np.log(data[j])+data[j]-model[j]
        #
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

        La.append(l1/(L1*A1))
        Lb.append(l2/(L2*A2))
        Lc.append(l3/A3)
        Ld.append(l4/A4)
        # np.savez('Ls', L1=La, L2=Lb, L3=Lc, L4=Ld)
        l+=l1/(L1*A1)+l2/(L2*A2)+l3/A3+l4/A4
    if counter%(len(p)+1)==0:
        print('!!!!!!!!!!! counter=', counter, 'params=', len(p), 'iteration=', int(counter/(len(p)+1)), 'fanc=',-l)
        print(l1/L1, l2/L2, l3, l4)
        print(rec)

    return -l




p=[35.63974159    ,36.18086164 ,27232.37881741 ,29182.99416736]
p=minimize(L, p, method='Nelder-Mead', options={'disp':True, 'maxfev':100000})
p=p.x
print(p)
NQ=[p[0], p[1]]
N_events=[p[2], p[3]]

spec=[]
h_area=[]
h_spe=[]
h_dpe=[]
h_trpe=[]
nac=[]
nhc=[]
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


    m1=model_spec(ns, NQ[i], P)
    m2=model_spec(ns, NQ[i]*136/122, P)
    spec.append(N_events[i]*(0.91*m1+0.09*m2))

fig, ((ax1, ax2), (ax3, ax4))=plt.subplots(2,2)

ax1.plot(H_specs[0], 'k.')
ax1.plot(ns, H_specs[0][ns], 'r.')
ax1.plot(ns, spec[0], 'g.-')
ax1.plot(ns, N_events[0]*(0.91*poisson.pmf(ns, NQ[0])+0.09*poisson.pmf(ns, 136/122*NQ[0])), 'b.-')


ax2.plot(areas[0], H_areas[0], 'k.', label='NAC={}, NHC={}'.format(NAC[0], NHC[0]))
ax2.plot(areas[0][rng_areas[0]], h_area[0], 'g.-', label='nac={}, nhc={}'.format(nac[0], nhc[0]))
ax2.plot(areas[0][rng_areas[0]], h_spe[0], '.-', label='SPE')
ax2.plot(areas[0][rng_areas[0]], h_dpe[0], '.-', label='DPE')
ax2.plot(areas[0][rng_areas[0]], h_trpe[0], '.-', label='TrPE')

ax2.set_yscale('log')
ax2.set_ylim(1,7000)
ax2.legend()


ax3.plot(H_specs[1], 'k.')
ax3.plot(ns, H_specs[1][ns], 'r.')
ax3.plot(ns, spec[1], 'g.-')
ax3.plot(ns, N_events[1]*(0.91*poisson.pmf(ns, NQ[1])+0.09*poisson.pmf(ns, 136/122*NQ[1])), 'b.-')


ax4.plot(areas[1], H_areas[1], 'k.', label='NAC={}, NHC={}'.format(NAC[1], NHC[1]))
ax4.plot(areas[1][rng_areas[1]], h_area[1], 'g.-', label='nac={}, nhc={}'.format(nac[1], nhc[1]))
ax4.plot(areas[1][rng_areas[1]], h_spe[1], '.-', label='SPE')
ax4.plot(areas[1][rng_areas[1]], h_dpe[1], '.-', label='DPE')
ax4.plot(areas[1][rng_areas[1]], h_trpe[1], '.-', label='TrPE')
ax4.set_ylim(1,7000)
ax4.set_yscale('log')
ax4.legend()



plt.show()
