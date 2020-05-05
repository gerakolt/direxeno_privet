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
# dt=3230 #Cs137
dt_BG=48524


pmt=0
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
data=np.load(path+'spe.npz')
rec=data['rec']
rise_time_cut=data['rise_time_cut']
height_cut=data['height_cut']
spectrum_spe0, bins=np.histogram(rec[rec['rise_time']>rise_time_cut]['area'], bins=100, range=[-1500, 6000])
areas0=0.5*(bins[1:]+bins[:-1])
spectrum_spe_height_cut0, bins=np.histogram(rec[np.logical_and(rec['rise_time']>rise_time_cut, rec['height']>height_cut)]['area'], bins=100, range=[-1500, 6000])
rng_area0=np.nonzero(np.logical_and(areas0>-1250, areas0<6000))

path='/home/gerak/Desktop/DireXeno/190803/BG/PMT{}/'.format(pmt)
data=np.load(path+'recon_wf.npz')
rec=data['rec']
chi2_cut=1.7e6
rng=np.nonzero(np.logical_and(rec['chi2']<chi2_cut, rec['init']>=70))
spectrum0, bins=np.histogram(np.sum(rec[rng]['h'], axis=1), bins=np.arange(200)-0.5)
spectrum_BG0=dt*spectrum0/dt_BG

path='/home/gerak/Desktop/DireXeno/190803/Co57/PMT{}/'.format(pmt)
data=np.load(path+'recon_wf.npz')
rec=data['rec']
chi2_cut=1.7e6
rng=np.nonzero(np.logical_and(rec['chi2']<chi2_cut, rec['init']>=70))
spectrum0, bins=np.histogram(np.sum(rec[rng]['h'], axis=1), bins=np.arange(200)-0.5)

path='/home/gerak/Desktop/DireXeno/190803/Co57B/PMT{}/'.format(pmt)
data=np.load(path+'recon_wf.npz')
rec=data['rec']
chi2_cut=1.7e6
rng=np.nonzero(np.logical_and(rec['chi2']<chi2_cut, rec['init']>=70))
spectrum0B, bins=np.histogram(np.sum(rec[rng]['h'], axis=1), bins=np.arange(200)-0.5)


pmt=8
path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
data=np.load(path+'spe.npz')
rec=data['rec']
rise_time_cut=data['rise_time_cut']
height_cut=data['height_cut']
spectrum_spe8, bins=np.histogram(rec[rec['rise_time']>rise_time_cut]['area'], bins=100, range=[-1500, 6000])
areas8=0.5*(bins[1:]+bins[:-1])
spectrum_spe_height_cut8, bins=np.histogram(rec[np.logical_and(rec['rise_time']>rise_time_cut, rec['height']>height_cut)]['area'], bins=100, range=[-1500, 6000])
rng_area8=np.nonzero(np.logical_and(areas8>-1250, areas8<6000))

path='/home/gerak/Desktop/DireXeno/190803/BG/PMT{}/'.format(pmt)
data=np.load(path+'recon_wf.npz')
rec=data['rec']
chi2_cut=1.7e6
rng=np.nonzero(np.logical_and(rec['chi2']<chi2_cut, rec['init']>=70))
spectrum8, bins=np.histogram(np.sum(rec[rng]['h'], axis=1), bins=np.arange(200)-0.5)
spectrum_BG8=dt*spectrum8/dt_BG

path='/home/gerak/Desktop/DireXeno/190803/Co57/PMT{}/'.format(pmt)
data=np.load(path+'recon_wf.npz')
rec=data['rec']
chi2_cut=1.7e6
rng=np.nonzero(np.logical_and(rec['chi2']<chi2_cut, rec['init']>=70))
spectrum8, bins=np.histogram(np.sum(rec[rng]['h'], axis=1), bins=np.arange(200)-0.5)

path='/home/gerak/Desktop/DireXeno/190803/Co57B/PMT{}/'.format(pmt)
data=np.load(path+'recon_wf.npz')
rec=data['rec']
chi2_cut=1.7e6
rng=np.nonzero(np.logical_and(rec['chi2']<chi2_cut, rec['init']>=70))
spectrum8B, bins=np.histogram(np.sum(rec[rng]['h'], axis=1), bins=np.arange(200)-0.5)


ns=np.arange(30,75)




counter=0
def L(p):
    global counter
    counter+=1
    [NQ1, NQ2, Mpad, Spad, Mpe, Spe, a_pad, a_spe, a_dpe, p01, a_spec1, a_spec2, BG_r]=p

    if NQ1<=0:
        return 1e10*(1-NQ1)
    if NQ2<=0:
        return 1e10*(1-NQ2)
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
    if a_spec1<=0:
        return 1e10*(1-a_spec1)
    if a_spec2<=0:
        return 1e10*(1-a_spec2)
    if p01<=0:
        return 1e10*(1-p01)
    if p01>=1:
        return 1e10*p01
    if 0.5*(1+erf(0.5/(np.sqrt(2)*Spe)))<p01:
        return 1e10*(1+p01-0.5*(1+erf(0.5/(np.sqrt(2)*Spe))))
    if BG_r>2:
        return 1e10*BG_r


    P=make_P(Spe, p01)
    if np.shape(P)==():
        return 1e10*P
    h_spec=model_spec(ns, NQ1, P, a_spec1)+model_spec(ns, NQ2, P, a_spec2)
    h_area=model_area(areas[rng_area], Mpad, Spad, Mpe, Spe, a_pad, a_spe, a_spe)

    l1=0
    for i in range(len(h_spec)):
        if spectrum[ns][i]<=BG_r*spectrum_BG[ns][i] and h_spec[i]>0:
            l1-=h_spec[i]
        elif h_spec[i]==0 and spectrum[ns][i]>BG_r*spectrum_BG[ns][i]:
            l1+=spectrum[ns][i]-BG_r*spectrum_BG[ns][i]-h_spec[i]-(spectrum[ns][i]-BG_r*spectrum_BG[ns][i])*np.log(spectrum[ns][i]-BG_r*spectrum_BG[ns][i])-(spectrum[ns][i]-BG_r*spectrum_BG[ns][i])*1e100
        elif h_spec[i]>0 and spectrum[ns][i]>BG_r*spectrum_BG[ns][i]:
            l1+=(spectrum[ns][i]-BG_r*spectrum_BG[ns][i])*np.log(h_spec[i])-(spectrum[ns][i]-BG_r*spectrum_BG[ns][i])*np.log(spectrum[ns][i]-BG_r*spectrum_BG[ns][i])+spectrum[ns][i]-BG_r*spectrum_BG[ns][i]-h_spec[i]

    l2=0
    for i in range(len(h_area)):
        if spectrum_spe[rng_area][i]==0 and h_area[i]>0:
            l2-=h_area[i]
        elif h_area[i]==0 and spectrum_spe[rng_area][i]>0:
            l2+=spectrum_spe[rng_area][i]-h_area[i]-spectrum_spe[rng_area][i]*np.log(spectrum_spe[rng_area][i])-spectrum_spe[rng_area][i]*1e100
        elif h_area[i]>0 and spectrum_spe[rng_area][i]>0:
            l2+=spectrum_spe[rng_area][i]*np.log(h_area[i])-spectrum_spe[rng_area][i]*np.log(spectrum_spe[rng_area][i])+spectrum_spe[rng_area][i]-h_area[i]

    all_pes=(np.sqrt(2*np.pi*(Spad**2+Mpe**2*Spe**2))*a_spe+np.sqrt(2*np.pi*(Spad**2+2*Mpe**2*Spe**2))*a_dpe)/(areas[1]-areas[0])
    all_pes_height_cut=(1-p01)*all_pes
    l3=0
    l3=np.sum(spectrum_spe_height_cut)*np.log(all_pes_height_cut)-np.sum(spectrum_spe_height_cut)*np.log(np.sum(spectrum_spe_height_cut))+np.sum(spectrum_spe_height_cut)-all_pes_height_cut

    l=l1/len(h_spec)+l2/len(h_area)+l3
    print('0 counter=', counter, 'params=', len(p), 'iteration=', int(counter/(len(p)+1)))
    print(l1/len(h_spec), l2/len(h_area), l3)
    print(-l, p)
    # print(all_pes, all_pes_height_cut, np.sum(spectrum_spe_height_cut))

    return -l

spectrum=spectrum0
spectrum_BG=spectrum_BG0
spectrum_spe=spectrum_spe0
rng_area=rng_area0
areas=areas0
spectrum_spe_height_cut=spectrum_spe_height_cut0

p=[7.12389036e+01, 4.72595750e+01, 5.91145951e-03, 3.38139403e+02,
 1.82144319e+03, 7.12755616e-01, 5.96755877e+03, 4.39838361e+02,
 5.79087452e+02, 4.07767931e-01, 1.19055839e+04, 6.35010427e+03,
 1.20255185e+00]

p=minimize(L, p, method='Nelder-Mead', options={'disp':True, 'maxfev':1000})
print(p.x)
[NQ1, NQ2, Mpad, Spad, Mpe, Spe, a_pad, a_spe, a_dpe, p01, a_spec1, a_spec2, BG_r]=p.x
# print(L(p))
P=make_P(Spe, p01)
h_spec=model_spec(ns, NQ1, P, a_spec1)+model_spec(ns, NQ2, P, a_spec2)
h_area=model_area(areas, Mpad, Spad, Mpe, Spe, a_pad, a_spe, a_spe)


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

ax1.plot(np.arange(len(spectrum0)), spectrum0, 'k.', label='number of PEs - PMT0')
# ax1.plot(np.arange(len(spectrum0)), spectrum0B, 'r.', label='number of PEs - PMT0 B')
ax1.plot(np.arange(len(spectrum0)), BG_r*spectrum_BG0, 'y.', label='number of PEs - BG')
ax1.legend()

ax2.plot(np.arange(len(spectrum8)), spectrum8, 'k.', label='number of PEs - PMT8')
# ax2.plot(np.arange(len(spectrum8)), spectrum8B, 'k.', label='number of PEs - PMT8 B')
ax2.plot(np.arange(len(spectrum8)), BG_r*spectrum_BG8, 'y.', label='number of PEs - BG')
ax2.legend()

ax3.plot(np.arange(len(spectrum0)), spectrum0-BG_r*spectrum_BG0, 'k.', label='number of PEs')
ax3.plot(ns, h_spec, 'r.-', label='NQ1={}, NQ2={},\na1={}, a2={}'.format(NQ1, NQ2, a_spec1, a_spec2))
ax3.legend()

ax4.plot(np.arange(len(spectrum8)), spectrum8-BG_r*spectrum_BG8, 'k.', label='number of PEs')
#ax4.plot(ns, h_spec, 'r.-', label='NQ1={}, NQ2={},\na1={}, a2={}'.format(NQ1, NQ2, a_spec1, a_spec2))
ax4.legend()



ax5.plot(areas0, spectrum_spe0, 'k.', label='area')
ax5.plot(areas0, spectrum_spe_height_cut0, 'g.', label='area - height_cut')
ax5.plot(areas0[rng_area0], h_area[rng_area0], 'r.-', label='Mpe={}, Spe={}, P01={}'.format(Mpe, Spe, p01))
ax5.set_yscale('log')
ax5.legend()

ax6.plot(areas8, spectrum_spe8, 'k.', label='area')
ax6.plot(areas8, spectrum_spe_height_cut8, 'g.', label='area - height_cut')
#ax6.plot(areas8[rng_area8], h_area[rng_area8], 'r.-', label='Mpe={}, Spe={}, P01={}'.format(Mpe, Spe, p01))
ax6.set_yscale('log')
ax6.legend()

plt.show()
