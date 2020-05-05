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
spectrum_spe, bins=np.histogram(rec[rec['rise_time']>rise_time_cut]['area'], bins=100, range=[-1500, 6000])
areas=0.5*(bins[1:]+bins[:-1])
spectrum_spe_height_cut, bins=np.histogram(rec[np.logical_and(rec['rise_time']>rise_time_cut, rec['height']>height_cut)]['area'], bins=100, range=[-1500, 6000])
rng_area=np.nonzero(np.logical_and(areas>-1250, areas<3400))

path='/home/gerak/Desktop/DireXeno/190803/BG/PMT{}/'.format(pmt)
data=np.load(path+'recon_wf.npz')
rec=data['rec']
chi2_cut=1.7e6
rng=np.nonzero(np.logical_and(rec['chi2']<chi2_cut, rec['init']>=70))
spectrum, bins=np.histogram(np.sum(rec[rng]['h'], axis=1), bins=np.arange(200)-0.5)
spectrum_BG=dt*spectrum/dt_BG

path='/home/gerak/Desktop/DireXeno/190803/Co57/PMT{}/'.format(pmt)
data=np.load(path+'recon_wf.npz')
rec=data['rec']
chi2_cut=1.7e6
rng=np.nonzero(np.logical_and(rec['chi2']<chi2_cut, rec['init']>=70))
spectrum, bins=np.histogram(np.sum(rec[rng]['h'], axis=1), bins=np.arange(200)-0.5)

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
    if a_spec2>0.2*a_spec1:
        return 500000*a_spec2/a_spec1
    if p01<=0:
        return 1e10*(1-p01)
    if p01>=1:
        return 1e10*p01
    if 0.5*(1+erf(0.5/(np.sqrt(2)*Spe)))<p01:
        return 1e10*(1+p01-0.5*(1+erf(0.5/(np.sqrt(2)*Spe))))
    if BG_r>2:
        return 1e10*BG_r
    if NQ2>1.2*NQ1:
        return 100000*NQ2/NQ1
    if NQ2<NQ1:
        return 100000*(NQ1-NQ2)


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

    if counter%(len(p)+1)==0:
        print(pmt, 'counter=', counter, 'params=', len(p), 'iteration=', int(counter/(len(p)+1)))
        print(l1/len(h_spec), l2/len(h_area), l3)
        print(-l, p)

    return -l

p=[4.43480637e+01, 4.43480637e+01, 2.57528489e-05, 3.40567722e+02,
 1.58200486e+03, 8.35731888e-01, 2.30105633e+03, 1.80048068e+02,
 7.52576401e+01, 2.30359871e-01, 5.20541055e+03, 2.66746820e+02,
 1]


p=minimize(L, p, method='Nelder-Mead', options={'disp':True, 'maxfev':10000})
print(p.x)
[NQ1, NQ2, Mpad, Spad, Mpe, Spe, a_pad, a_spe, a_dpe, p01, a_spec1, a_spec2, BG_r]=p.x
P=make_P(Spe, p01)
h1=model_spec(ns, NQ1, P, a_spec1)
h2=model_spec(ns, NQ2, P, a_spec2)
h_spec=h1+h2
h_area=model_area(areas, Mpad, Spad, Mpe, Spe, a_pad, a_spe, a_spe)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('PMT{}'.format(pmt))
ax1.plot(np.arange(len(spectrum)), spectrum, 'k.', label='number of PEs - PMT0')
ax1.plot(np.arange(len(spectrum)), BG_r*spectrum_BG, 'y.', label='number of PEs - BG')
ax1.plot(ns, h_spec+BG_r*spectrum_BG[ns], 'r.-', label='NQ1={}, NQ2={},\na1={}, a2={}'.format(NQ1, NQ2, a_spec1, a_spec2))
ax1.plot(np.arange(len(spectrum)), a_spec1*poisson.pmf(np.arange(len(spectrum)), NQ1), 'g.-')
ax1.plot(np.arange(len(spectrum)), a_spec2*poisson.pmf(np.arange(len(spectrum)), NQ2), 'b.-')
ax1.plot(ns, h1+BG_r*spectrum_BG[ns], '+-')
ax1.plot(ns, h2+BG_r*spectrum_BG[ns], '+-')

ax1.legend()

ax2.plot(areas, spectrum_spe, 'k.', label='area')
ax2.plot(areas, spectrum_spe_height_cut, 'g.', label='area - height_cut')
ax2.plot(areas[rng_area], h_area[rng_area], 'r.-', label='Mpe={}, Spe={}, P01={}'.format(Mpe, Spe, p01))
ax2.set_yscale('log')
ax2.legend()

plt.show()
