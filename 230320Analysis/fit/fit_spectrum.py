import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import poisson, binom
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.special import erf as erf
import sys
from scipy.optimize import minimize
from fun import model_area, model_spec, make_P

dt=29710 #Co57
dt_BG=48524

pmt=0
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
data=np.load(path+'spe.npz')
rec=data['rec']
rise_time_cut=data['rise_time_cut']
height_cut=data['height_cut']
spec_spe, bins=np.histogram(rec[rec['rise_time']>rise_time_cut]['area'], bins=100, range=[-1500, 6000])
areas=0.5*(bins[1:]+bins[:-1])
spec_spe_height_cut, bins=np.histogram(rec[np.logical_and(rec['rise_time']>rise_time_cut, rec['height']>height_cut)]['area'], bins=100, range=[-1500, 6000])
rng_area=np.nonzero(np.logical_and(areas>-1000, areas<6000))[0]

path='/home/gerak/Desktop/DireXeno/190803/BG/PMT{}/'.format(pmt)
data=np.load(path+'recon_wf.npz')
rec_BG=data['rec']
chi2_cut=1.7e6
rng_BG=np.nonzero(np.logical_and(rec_BG['chi2']<chi2_cut, rec_BG['init']>=70))
spec, bins=np.histogram(np.sum(rec_BG[rng_BG]['h'], axis=1), bins=np.arange(200)-0.5)
spec_BG=dt*spec/dt_BG

path='/home/gerak/Desktop/DireXeno/190803/Co57/PMT{}/'.format(pmt)
data=np.load(path+'recon_wf.npz')
rec=data['rec']
chi2_cut=1.7e6
rng=np.nonzero(np.logical_and(rec['chi2']<chi2_cut, rec['init']>=70))
spec, bins=np.histogram(np.sum(rec[rng]['h'], axis=1), bins=np.arange(200)-0.5)

ns=np.arange(36,150)

counter=0
def L(p):
    global counter
    counter+=1
    [NQ, Mpad, Spad, Mpe, Spe, a_pad, a_spe, a_dpe, p01, N_events, BG_events]=p

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
    if BG_events>1.2:
        return 1e10*BG_events


    l1=0
    l2=0
    l3=0
    P=make_P(Spe, p01)
    if np.shape(P)==():
        return 1e10*P
    h_spec=0.91*model_spec(ns, NQ, P)+0.09*model_spec(ns, NQ*136/122, P)
    h_area=model_area(areas[rng_area], Mpad, Spad, Mpe, Spe, a_pad, a_spe, a_spe)

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


    mod=h_area
    dat=spec_spe[rng_area]
    L2=len(mod)
    for i in range(L2):
        if mod[i]>0 and dat[i]<=0:
            l2-=mod[i]-dat[i]
        elif mod[i]<=0 and dat[i]>0:
            return 1e10*(dat[i]-mod[i])
        elif mod[i]==0 and dat[i]==0:
            l2+=1
        else:
            l2+=dat[i]*np.log(mod[i])-dat[i]*np.log(dat[i])+dat[i]-mod[i]

    all_pes=(np.sqrt(2*np.pi*(Spad**2+Mpe**2*Spe**2))*a_spe+np.sqrt(2*np.pi*(Spad**2+2*Mpe**2*Spe**2))*a_dpe)/(areas[1]-areas[0])
    mod=(1-p01)*all_pes/10
    dat=np.sum(spec_spe_height_cut)/10
    l3=dat*np.log(mod)-dat*np.log(dat)+dat-mod

    l=l1/L1+l2/L2+l3

    if counter%(len(p)+1)==0:
        print(pmt, 'counter=', counter, 'params=', len(p), 'iteration=', int(counter/(len(p)+1)))
        print(l1/len(h_spec), l2/len(h_area), l3)
        print(-l, p)

    return -l

p=[45.78367768299326, -1.7031166e-08, 334.8789513595051, 1832.410914148476,
    0.6834665201844516, 3764.1732688392367, 291.53134852468065, 166.88931963268692, 0.11281125013353435, 16098.772772626628, 0.5]

p=minimize(L, p, method='Nelder-Mead', options={'disp':True, 'maxfev':1000})
print(p.x)
p=p.x
[NQ1, Mpad, Spad, Mpe, Spe, a_pad, a_spe, a_dpe, p01, N_events, BG_events]=p
P=make_P(Spe, p01)
h_spec=N_events*model_spec(ns, NQ1, P)
h_area=model_area(areas, Mpad, Spad, Mpe, Spe, a_pad, a_spe, a_spe)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('PMT{}'.format(pmt))
ax1.plot(np.arange(len(spec)), spec, 'k.', label='number of PEs - PMT0')
ax1.plot(np.arange(len(spec)), BG_events*spec_BG, 'y.', label='number of PEs - BG')
ax1.plot(ns, h_spec+BG_events*spec_BG[ns], 'r.-', label='NQ1={},\na1={}'.format(NQ1, N_events))
ax1.legend()

ax2.plot(areas, spec_spe, 'k.', label='area')
ax2.plot(areas, spec_spe_height_cut, 'g.', label='area - height_cut')
ax2.plot(areas[rng_area], h_area[rng_area], 'r.-', label='Mpe={}, Spe={}, P01={}'.format(Mpe, Spe, p01))
ax2.set_yscale('log')
ax2.legend()

plt.show()


H=np.zeros((10,1000))
H_BG=np.zeros((10,1000))
rec=rec[rng]
rec_BG=rec_BG[rng_BG]
rng=np.nonzero(np.logical_and(np.sum(rec['h'], axis=1)<=ns[-1], np.sum(rec['h'], axis=1)>=ns[0]))[0]
rng_BG=np.nonzero(np.logical_and(np.sum(rec_BG['h'], axis=1)<=ns[-1], np.sum(rec_BG['h'], axis=1)>=ns[0]))[0]
# for i in range(1000):
#     print(i)
#     H[:,i]=np.histogram(rec[rng]['h'][:,i], bins=np.arange(len(H[:,0])+1)-0.5)[0]
#     H_BG[:,i]=np.histogram(rec_BG[rng_BG]['h'][:,i], bins=np.arange(len(H[:,0])+1)-0.5)[0]

# np.savez(path+'H', H=H, H_BG=H_BG, ns=ns, spec=spec, spec_BG=spec_BG, areas=areas, spec_spe=spec_spe, rng_area=rng_area, spec_spe_height_cut=spec_spe_height_cut, p0=p)
