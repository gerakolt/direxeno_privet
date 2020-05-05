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
ns=np.arange(30,75)

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
rec_BG=data['rec']
chi2_cut=1.7e6
rng_BG=np.nonzero(np.logical_and(rec_BG['chi2']<chi2_cut, rec_BG['init']>=70))


path='/home/gerak/Desktop/DireXeno/190803/Co57/PMT{}/'.format(pmt)
data=np.load(path+'recon_wf.npz')
rec=data['rec']
chi2_cut=1.7e6
rng=np.nonzero(np.logical_and(rec['chi2']<chi2_cut, rec['init']>=70))

spec=np.histogram(np.sum(rec[rng]['h'], axis=1), bins=np.arange(200)-0.5)[0]
spec_BG=np.histogram(np.sum(rec_BG[rng_BG]['h'], axis=1), bins=np.arange(200)-0.5)[0]


H=np.zeros((10,1000))
H_BG=np.zeros((10,1000))
rec=rec[rng]
rec_BG=rec_BG[rng_BG]
rng=np.nonzero(np.logical_and(np.sum(rec['h'], axis=1)<=ns[-1], np.sum(rec['h'], axis=1)>=ns[0]))[0]
rng_BG=np.nonzero(np.logical_and(np.sum(rec_BG['h'], axis=1)<=ns[-1], np.sum(rec_BG['h'], axis=1)>=ns[0]))[0]
for i in range(1000):
    print(i)
    H[:,i]=np.histogram(rec[rng]['h'][:,i], bins=np.arange(len(H[:,0])+1)-0.5)[0]
    H_BG[:,i]=np.histogram(rec_BG[rng_BG]['h'][:,i], bins=np.arange(len(H[:,0])+1)-0.5)[0]

np.savez(path+'H', H=H, H_BG=H_BG, ns=ns, spec=spec, spec_BG=spec_BG, areas=areas)
