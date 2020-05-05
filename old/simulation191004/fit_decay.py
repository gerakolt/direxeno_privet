from classes import WF, Event
import numpy as np
import time
import matplotlib.pyplot as plt
from fun import do_smd, do_dif, Find_Peaks, Fix_Peaks, Analize_Peaks, find_peaks, Fit_Decay, fix_peaks, analize_peaks
from hit_fun_order import find_peaks, Reconstruct_WF
from scipy.optimize import curve_fit
from scipy import special
import sys

def Fit_Decay(t, PE, PE_var):
    def func(x, t0, tau, T, s):
        y=s/(np.sqrt(2)*tau)+T/(np.sqrt(2)*s)
        t=x-t0
        f=np.exp(-t/tau)*(special.erf(y)-special.erf(y-t/(np.sqrt(2)*s)))/tau
        f[f<0]=0
        return f

    def func2(x, a, t0, tau_s, tau_f, R_s, T, s):
        return a*((1-R_s)*func(x, *[t0, tau_f, T, s])+R_s*func(x, *[t0, tau_s, T, s]))

    Maxi=t[np.argmax(PE)]
    t0_dn=Maxi-10
    t0=Maxi-3
    t0_up=Maxi

    a0_dn=0
    a0=500
    a0_up=5000

    tau_f_dn=0
    tau_f=27
    tau_f_up=200

    tau_s_dn=0
    tau_s=45
    tau_s_up=200

    R_s_dn=0
    R_s=0.5
    R_s_up=1

    T_dn=0
    T=0.5
    T_up=10

    s_dn=0
    s=1.5
    s_up=10

    p0=[a0, t0, tau_s, tau_f, R_s, T, s]
    bound_dn=[a0_dn, t0_dn, tau_s_dn, tau_f_dn, R_s_dn, T_dn, s_dn]
    bound_up=[a0_up, t0_up, tau_s_up, tau_f_up, R_s_up, T_up, s_up]
    bounds=[bound_dn, bound_up]
    short=0
    try:
        p, cov=curve_fit(func2, t, PE, p0=p0, bounds=bounds, sigma=np.sqrt(PE_var))
    except:
        a0_dn=0
        a0=500
        a0_up=3000

        tau_f_dn=0
        tau_f=5
        tau_f_up=10

        tau_s_dn=10
        tau_s=27
        tau_s_up=100

        R_s_dn=0
        R_s=0.8
        R_s_up=1

        bound_dn=[a0_dn, t0_dn, tau_s_dn, tau_f_dn, R_s_dn, T_dn, s_dn]
        bound_up=[a0_up, t0_up, tau_s_up, tau_f_up, R_s_up, T_up, s_up]
        bounds=[bound_dn, bound_up]
        p0=[a0, t0, tau_s, tau_f, R_s, T, s]
        try:
            p, cov=curve_fit(func2, t, PE, p0=p0, bounds=bounds, sigma=np.sqrt(PE_var))
            short=1
            # plt.plot(t, PE, 'k.', label='{} PEs'.format(np.sum(PE)))
            # plt.plot(t, func2(t, *p), 'r.')
            # plt.title('Fit to short event')
            # plt.errorbar(t, PE, yerr=np.sqrt(PE_var), ls='')
            # plt.show()
        except:
            p0=[a0, t0, tau_s, tau_f, 1, T, s]
            bound_dn=[a0_dn, t0_dn, tau_s_dn, tau_f_up, 0.99, T_dn, s_dn]
            bound_up=[a0_up, t0_up, tau_s_up, tau_f_dn, 1.01, T_up, s_up]
            bounds=[bound_dn, bound_up]
            try:
                p1exp=curve_fit(func2, t, PE, p0=p0, bounds=bounds, sigma=np.sqrt(PE_var))
            except:
                p=np.zeros(7)
            # plt.plot(t, PE, 'k.', label='{} PEs'.format(np.sum(PE)))
            # p0=[2200, t0, 27,5,0.8,0.5,1.2]
            # plt.plot(t, func2(t, *p0), 'r.-')
            # plt.errorbar(t, PE, yerr=np.sqrt(PE_var), ls='')
            # plt.title('Double fail')
            # plt.show()

    return p, short
