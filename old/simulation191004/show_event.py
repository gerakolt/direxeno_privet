from classes import WF, Event
import numpy as np
import time
import matplotlib.pyplot as plt
from fun import do_smd, do_dif, Find_Peaks, Fix_Peaks, Analize_Peaks, find_peaks, Fit_Decay, fix_peaks, analize_peaks
from hit_fun_order import find_peaks, Reconstruct_WF
from scipy.optimize import curve_fit
from scipy import special
import pickle
from fit_decay import Fit_Decay
from os import listdir
from os.path import isfile, join
import sys
from shootPEs import show_pmt_hit
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def func(x, t0, tau, T, s):
    y=s/(np.sqrt(2)*tau)+T/(np.sqrt(2)*s)
    t=x-t0
    f=np.exp(-t/tau)*(special.erf(y)-special.erf(y-t/(np.sqrt(2)*s)))/tau
    f[f<0]=0
    return f

def func2(x, a, t0, tau_s, tau_f, R_s, T, s):
    return a*((1-R_s)*func(x, *[t0, tau_f, T, s])+R_s*func(x, *[t0, tau_s, T, s]))

id=2
path='/home/gerak/Desktop/Git/simulation190809/190809sim/'
files = [path+f for f in listdir(path) if isfile(join(path, f))]
for file in files:
    if file.endswith('.pkl'):
        continue
    with open(file, 'rb') as input:
        print(file)
        dataset = pickle.load(input)
        for event in dataset.events:
            if event.id==id:
                t=event.wf[0].T
                PE=event.wf[0].PE
                PE_var=np.ones(len(PE))
                for wf in event.wf[1:]:
                    PE=PE+wf.PE
                fig = plt.figure(figsize=[15,10])
                ax0=fig.add_subplot(2,1,1)
                ax0.plot(t, PE, 'k.', label='{} PEs'.format(np.sum(PE)), markersize=20)
                ax0.errorbar(t, PE, yerr=np.sqrt(PE_var), ls='', elinewidth=6, ecolor='k')
                p, short = Fit_Decay(t, PE, PE_var)
                x=np.linspace(t[0], t[-1], 500)
                ax0.plot(x, func2(x, *p), 'r.', label='tau = {} ns'.format(p[2]), markersize=5)
                ax0.legend()
                show_pmt_hit(fig, 212, event, np.sum(PE))
                plt.show()
