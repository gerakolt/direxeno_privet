import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import poisson, binom
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.special import erf as erf
import sys
from scipy.optimize import minimize
from fun import model_area, model_spec, make_P, model2, make_initial_simplex

dt=29710 #Co57
dt_BG=48524

pmt=8
path='/home/gerak/Desktop/DireXeno/190803/Co57/PMT{}/'.format(pmt)
data=np.load(path+'H.npz')
H=data['H']
H_BG=data['H_BG']
ns=data['ns']
spec=data['spec']
spec_BG=data['spec_BG']
areas=data['areas']
spec_spe=data['spec_spe']
rng_area=data['rng_area']
spec_spe_height_cut=data['spec_spe_height_cut']
p0=data['p0']
t_range=50*5

counter=0
best_L=1e10
def L(p):
    global counter
    global best_L
    counter+=1
    [NQ, F, Tf, Ts, St, Mpad, Spad, Mpe, Spe, a_pad, a_spe, a_dpe, p01, N_events, BG_events]=p

    if NQ<=0:
        return 1e10*(1-NQ)
    if Spad<=0:
        return 1e10*(1-Spad)
    if Spe<=0:
        return 1e10*(1-Spe)
    if a_pad<=0:
        return 1e10*(1-a_pad)
    if a_spe<=0:
        return 1e10*(1-a_spe)
    if a_dpe<=0:
        return 1e10*(1-a_dpe)
    if N_events<=0:
        return 1e10*(1-N_events)
    if p01<=0:
        return 1e10*(1-p01)
    if p01>=1:
        return 1e10*p01
    if 0.5*(1+erf(0.5/(np.sqrt(2)*Spe)))<p01:
        return 1e10*(1+p01-0.5*(1+erf(0.5/(np.sqrt(2)*Spe))))
    if BG_events>2:
        return 1e10*BG_events
    if F<0:
        return 1e10*(1-F)
    if F>1:
        return 1e10*F

    P=make_P(Spe, p01)
    if np.shape(P)==():
        return 1e10*P

    h_spec=0.91*model_spec(ns, NQ, P)+0.09*model_spec(ns, NQ*136/122, P)
    h_area=model_area(areas[rng_area], Mpad, Spad, Mpe, Spe, a_pad, a_spe, a_spe)
    h=0.91*model2(NQ, F, Tf, Ts, St, P)+0.09*model2(NQ*136/122, F, Tf, Ts, St, P)

    l1=0
    l2=0
    l3=0
    l4=0

    mod=N_events*h_spec+BG_events*spec_BG[ns]
    dat=spec[ns]
    L1=len(mod)
    for i in range(L1):
        if mod[i]>0 and dat[i]<=0:
            l1-=mod[i]-dat[i]
        elif mod[i]<=0 and dat[i]>0:
            return 1e10*(dat[i]-mod[i])
        elif mod[i]==0 and dat[i]==0:
            l1+=1
        else:
            l1+=dat[i]*np.log(mod[i])-dat[i]*np.log(dat[i])+dat[i]-mod[i]

    mod=np.ravel(N_events*h+BG_events*H_BG)
    dat=np.ravel(H)
    if len(mod)!=len(dat):
        print('in h, len(mod)!=len(dat)')
        sys.exit()
    L2=len(mod)
    for i in range(L2):
        if i%1000>t_range:
            continue
        if mod[i]>0 and dat[i]<=0:
            l2-=mod[i]-dat[i]
        elif mod[i]<=0 and dat[i]>0:
            return 1e10*(dat[i]-mod[i])
        elif mod[i]==0 and dat[i]==0:
            l2+=1
        else:
            l2+=dat[i]*np.log(mod[i])-dat[i]*np.log(dat[i])+dat[i]-mod[i]

    mod=h_area
    dat=spec_spe[rng_area]
    L3=len(mod)
    for i in range(L3):
        if mod[i]>0 and dat[i]<=0:
            l3-=mod[i]-dat[i]
        elif mod[i]<=0 and dat[i]>0:
            return 1e10*(dat[i]-mod[i])
        elif mod[i]==0 and dat[i]==0:
            l3+=1
        else:
            l3+=dat[i]*np.log(mod[i])-dat[i]*np.log(dat[i])+dat[i]-mod[i]

    all_pes=(np.sqrt(2*np.pi*(Spad**2+Mpe**2*Spe**2))*a_spe+np.sqrt(2*np.pi*(Spad**2+2*Mpe**2*Spe**2))*a_dpe)/(areas[1]-areas[0])
    mod=(1-p01)*all_pes
    dat=np.sum(spec_spe_height_cut)
    l4=dat*np.log(mod)-dat*np.log(dat)+dat-mod
    if counter%(len(p)+1)==0:
        print('!!!!!!!!!!!!!!!!!!!!!!!!')
    print(counter, -l1/L1, -l2/L2, -l3/L3, -l4, -l1/L1-l2/L2-l3/L3-l4)
    print([NQ, F, Tf, Ts, St, Mpad, Spad, Mpe, Spe, a_pad, a_spe, a_dpe, p01, N_events, BG_events])
    if -l1/L1-l2/L2-l3/L3-l4<best_L:
        best_L=-l1/L1-l2/L2-l3/L3-l4
        np.savez(path+'best_p', p=p)
    return (-l1/L1-l2/L2-l3/L3-l4)


params=['NQ', 'F', 'Tf', 'Ts', 'St', 'Mpad', 'Spad', ' Mpe', 'Spe', 'a_pad', 'a_spe', 'a_dpe', 'p01', 'a_spec', 'BG_r']
# p=[44.41581207082156, 0.008613740045507277, 3.680004008520065, 37.910446460926366, 0.66624412905815, -4.0716967385539824e-08,
#  340.20953465163393, 1903.8719632992797, 0.625670181618356, 3844.5957034329886, 273.9735269443563, 172.13657806182215, 0.0797291382499056, 7697.295177737749, 1]

p=[37.60468113356808, 0.006910880701525407, 2.1780509848779213, 86.45286698461561, 0.6463315164886841, 117.27618466252986,
 338.597078684602, 996.7342124028906, 0.6220908959026614, 1951.659733591425, 468.6190878655414, 276.2944131196665, 0.1657732723420235, 14576.078945843103, 1.2879768438650314]

# initial_simplex=make_initial_simplex(p)

# p=minimize(L, p, method='Nelder-Mead', options={'disp':True, 'maxfev':100000})
# print(p.x)
# p=p.x

[NQ, F, Tf, Ts, St, Mpad, Spad, Mpe, Spe, a_pad, a_spe, a_dpe, p01, N_events, BG_events]=p
P=make_P(Spe, p01)
h_spec=N_events*(0.91*model_spec(ns, NQ, P)+0.09*model_spec(ns, NQ*136/122, P))
h_area=model_area(areas[rng_area], Mpad, Spad, Mpe, Spe, a_pad, a_spe, a_spe)
h=N_events*(0.91*model2(NQ, F, Tf, Ts, St, P)+0.09*model2(NQ*136/122, F, Tf, Ts, St, P))

x=np.arange(1000)/5
fig, ((ax1, ax2), (ax3, ax4))=plt.subplots(2,2)

ax1.plot(x, np.mean(H.T*np.arange(len(H[:,0])), axis=1), 'g.', label='data')
ax1.plot(x, np.mean((h+BG_events*H_BG).T*np.arange(len(H[:,0])), axis=1), 'r.-', label='model')
ax1.plot(x, BG_events*np.mean(H_BG.T*np.arange(len(H[:,0])), axis=1), 'k.', label='BG')
# ax1.plot(x, np.mean((H-BG_events*H_BG).T*np.arange(len(H[:,0])), axis=1), 'g.', label='data')
# ax1.plot(x, np.mean((h).T*np.arange(len(H[:,0])), axis=1), 'r.-', label='model')
ax1.set_xlim(-2,20)
ax1.legend()

ax2.plot(spec, 'g.', label='data')
ax2.plot(BG_events*spec_BG, 'k.', label='BG')
ax2.plot(ns, h_spec+BG_events*spec_BG[ns], 'r.-', label='model')
ax2.legend()

ax3.plot(areas, spec_spe, 'g.', label='areas')
ax3.plot(areas[rng_area], h_area, 'r.-', label='model')
ax3.legend()
ax3.set_yscale('log')

plt.show()
