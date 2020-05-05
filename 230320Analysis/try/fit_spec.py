import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import poisson, binom
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.special import erf as erf
import sys
from scipy.optimize import minimize
from fun import model_area, model_spec

dt=29710
dt_BG=48524
pmt=0

path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
data=np.load(path+'spe.npz')
rec=data['rec']
rise_time_cut=data['rise_time_cut']
height_cut=data['height_cut']
height=rec[rec['rise_time']>rise_time_cut]['height']
spectrum_height, bins=np.histogram(rec[rec['rise_time']>rise_time_cut]['height'], bins=100, range=[0, 300])
heights=0.5*(bins[1:]+bins[:-1])
spectrum_spe, bins=np.histogram(rec[rec['rise_time']>rise_time_cut]['area'], bins=100, range=[-2000, 10000])
areas=0.5*(bins[1:]+bins[:-1])
da=areas[1]-areas[0]
spectrum_spe_height_cut, bins=np.histogram(rec[np.logical_and(rec['rise_time']>rise_time_cut, rec['height']>height_cut)]['area'], bins=100, range=[-2000, 10000])
areas_height_cut=0.5*(bins[1:]+bins[:-1])
rng_area=np.nonzero(np.logical_and(areas>-1250, areas<6000))

path='/home/gerak/Desktop/DireXeno/190803/BG/PMT{}/'.format(pmt)
data=np.load(path+'recon_wf.npz')
rec=data['rec']
chi2_cut=1.7e6
rng=np.nonzero(np.logical_and(rec['chi2']<chi2_cut, rec['init']>=70))
spectrum, bins=np.histogram(np.sum(rec[rng]['h'], axis=1), bins=np.arange(300)-0.5)
spectrum_BG=dt*spectrum/dt_BG

path='/home/gerak/Desktop/DireXeno/190803/Co57/PMT{}/'.format(pmt)
data=np.load(path+'recon_wf.npz')
rec=data['rec']
chi2_cut=1.7e6
rng=np.nonzero(np.logical_and(rec['chi2']<chi2_cut, rec['init']>=70))
spectrum, bins=np.histogram(np.sum(rec[rng]['h'], axis=1), bins=np.arange(300)-0.5)

ns=np.arange(40,70)

counter=0
def L(p):
    global counter
    counter+=1
    [NQ, Mpad, Spad, Mpe, Spe, a_pad, a_spe, a_dpe, p01, a_spec]=p

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
    if a_spec<=0:
        return 1e10*(1-a_spec)
    if p01<=0:
        return 1e10*(1-p01)
    if p01>=1:
        return 1e10*p01
    if 0.5*(1+erf(0.5/(np.sqrt(2)*Spe)))<p01:
        return 1e10*(1+p01-0.5*(1+erf(0.5/(np.sqrt(2)*Spe))))



    h_spec=model_spec(ns, NQ, Spe, p01, a_spec)
    h_area=model_area(areas[rng_area], Mpad, Spad, Mpe, Spe, a_pad, a_spe, a_spe)
    l1=0
    for i in range(len(h_spec)):
        if spectrum[ns][i]==0 and h_spec[i]>0:
            l1-=h_spec[i]
        elif h_spec[i]==0 and spectrum[ns][i]>0:
            l1+=spectrum[ns][i]-h_spec[i]-spectrum[ns][i]*np.log(spectrum[ns][i])-spectrum[ns][i]*1e100
        elif h_spec[i]>0 and spectrum[ns][i]>0:
            l1+=spectrum[ns][i]*np.log(h_spec[i])-spectrum[ns][i]*np.log(spectrum[ns][i])+spectrum[ns][i]-h_spec[i]

    l2=0
    for i in range(len(h_area)):
        if spectrum_spe[rng_area][i]==0 and h_area[i]>0:
            l2-=h_area[i]
        elif h_area[i]==0 and spectrum_spe[rng_area][i]>0:
            l2+=spectrum_spe[rng_area][i]-h_area[i]-spectrum_spe[rng_area][i]*np.log(spectrum_spe[rng_area][i])-spectrum_spe[rng_area][i]*1e100
        elif h_area[i]>0 and spectrum_spe[rng_area][i]>0:
            l2+=spectrum_spe[rng_area][i]*np.log(h_area[i])-spectrum_spe[rng_area][i]*np.log(spectrum_spe[rng_area][i])+spectrum_spe[rng_area][i]-h_area[i]

    all_pes=(np.sqrt(2*np.pi*(Spad**2+Mpe**2*Spe**2))*a_spe+np.sqrt(2*np.pi*(Spad**2+2*Mpe**2*Spe**2))*a_dpe)/da
    all_pes_height_cut=(1-p01)*all_pes
    l3=0
    l3=np.sum(spectrum_spe_height_cut)*np.log(all_pes_height_cut)-np.sum(spectrum_spe_height_cut)*np.log(np.sum(spectrum_spe_height_cut))+np.sum(spectrum_spe_height_cut)-all_pes_height_cut

    l=l1/len(h_spec)+l2/len(h_area)+l3
    print('counter=', counter, 'params=', len(p), 'iteration=', int(counter/(len(p)+1)))
    print(-l, p)
    # print(all_pes, all_pes_height_cut, np.sum(spectrum_spe_height_cut))

    return -l

p=[5.55766490e+01, -3.57420958e-04,  3.13841073e+02,  1.10527573e+03,
  1.54399782e+00,  5.85079025e+03,  4.35416711e+02,  7.79751522e+02,
  6.26830546e-01,  3.71753996e+04]
p=minimize(L, p, method='Nelder-Mead', options={'disp':True, 'maxfev':10000})
print(p.x)
[NQ, Mpad, Spad, Mpe, Spe, a_pad, a_spe, a_dpe, p01, a_spec]=p.x
h_spec=model_spec(ns, NQ, Spe, p01, a_spec)
h_area=model_area(areas, Mpad, Spad, Mpe, Spe, a_pad, a_spe, a_spe)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(np.arange(len(spectrum)), spectrum, 'k.', label='number of PEs')
ax1.plot(ns, h_spec, 'r.-', label='NQ={}'.format(NQ))
ax1.plot(np.arange(len(spectrum)), spectrum_BG, 'y.', label='number of PEs - BG')
ax1.legend()

ax2.plot(areas, spectrum_spe, 'k.', label='area')
ax2.plot(areas_height_cut, spectrum_spe_height_cut, 'g.', label='area - height_cut')
ax2.plot(areas[rng_area], h_area[rng_area], 'r.-', label='Mpe={}, Spe={}, P01={}'.format(Mpe, Spe, p01))
ax2.set_yscale('log')
ax2.legend()

ax3.plot(np.arange(len(spectrum)), spectrum-spectrum_BG, 'k.', label='number of PEs - data-BG')
ax3.legend()

plt.show()
