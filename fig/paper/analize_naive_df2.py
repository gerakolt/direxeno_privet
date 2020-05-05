import os, sys
import numpy as np
from scipy import signal
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
import sys
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
import matplotlib.ticker as mticker
import matplotlib.ticker as mtick

v=850
f=5
PMT_num=20
R=50
v_to_adc=38e2
elementry_charge=1.6e-19
f=5
#path='/data/Gera/coldT_splited/{}V_{}GHz/'.format(v,f)
path='/home/gerak/Desktop/gain/new/'
D=pd.read_pickle(path+'df_naive_gain_bl.pkl')
D['t']=D['maxi']-D['trig']
D['area']=(D['area_l']+D['area_r'])/2
# D['dt']=D['fin']-D['init']
Tdn=[60,32,30,45,45,30,45,35,50,30,45,50,30,55,50,50,30,30,25]
Tup=[80,53,100,75,75,60,80,70,80,100,75,75,100,85,85,80,100,100,55]
#
blup=np.array([3878,3907,3611,0,3573,3945,3674,3752,3607,0,3662,0,0,3595,3580,3637,0,0,3702])
bldn=np.array([3866,3896,3598,0,3560,3934,3660,3740,3592,0,3648,0,0,3580,3560,3623,0,0,3687])


sumdn=np.array([10,0,0,0,0,-5,15,0,15,0,0,0,0,20,0,0,0,0,-20])
sumup=np.array([150,125,130,0,125,125,120,125,140,0,140,0,0,140,150,150,0,0,120])

uper=[3500,2000,2000,4000,3000,6000,2000,2500,5000,2500,5000,2500,2500,3500,3000,5000,2500,2500,3500,3500,]

l=[0.18792113800848514,
0.10218812162546177,
0.22479512102153612,
0.009925558312655087,
0.26765996431843386,
0.6864389854204114,
0.5829552819183409,
0.6135652427043156,
0.6644100580270793,
0.07494866529774127,
0.48986043182631517,
0.5286383903052475,
0.0810522573764664,
0.33795927653281765,
0.46936656282450673,
0.30505211445263225,
0.021621621621621623,
0.002564102564102564,
0.2726446151434394
]

# def func(x, a,b,c,m_bl,s_bl, m, s):
#     return a*np.exp(-0.5*(x-m_bl)**2/s_bl**2)+b*np.exp(-0.5*(x-m)**2/s**2)+c*b*np.exp(-0.25*(x-2*m)**2/s**2)
def func(x, a,b,m_bl,s_bl, m, s):
    return a*np.exp(-0.5*(x-m_bl)**2/s_bl**2)+b*np.exp(-0.5*(x-m)**2/s**2)

for chn in range(5, PMT_num-1):
    d=D[(D['channel']==chn) & (D['deriv']<15) & (D['deriv_l']<15) & (D['deriv_r']<15)  & (D['bl']<blup[chn]) & (D['bl']>bldn[chn])]
    #d=D[(D['channel']==chn) & (D['deriv']<15)]
    dc=d[(d['t']<Tup[chn]) & (d['t']>Tdn[chn])]

    # #print(dc[dc['area']>2000])

    #if chn==4 or chn==5 or chn==6 or chn==7 or chn==8 or chn==10 or chn==13 or chn==14 or chn==15 or chn==18:
    if 1>0:
        def func(x, a,b,c,m_bl,s_bl, m, s):
            # return a*np.exp(-0.5*(x-m_bl)**2/s_bl**2)+b*np.exp(-0.5*(x-m)**2/s**2)+c*np.exp(-0.25*(x-2*m)**2/s**2)
            return a*np.exp(-0.5*(x-m_bl)**2/s_bl**2)+b*np.exp(-0.5*(x-(m_bl+m))**2/(s**2+s_bl**2))+c*np.exp(-0.5*(x-(m_bl+2*m))**2/(2*s**2+s_bl**2))

        h, bins, pathc = plt.hist(dc['area']/(R*5e9*elementry_charge*v_to_adc*1e6), bins=75, log=True, histtype='step', range=[-4,36], linewidth=5)
        x=(bins[1:]+bins[:-1])/2
        plt.show()
        if chn==14:
            rng=np.nonzero(np.logical_and(x>-200, x<uper[chn]))[0]
        else:
            print('fuck')
            rng=np.nonzero(np.logical_and(x>-3.5, x<35))[0]
        a=np.amax(h)
        b=np.amax(h[x>4])
        c=0.25*b
        m_bl=0
        m=11*1e6
        s_bl=1e6
        s=4*1e6
        p=(a,b,c,m_bl,s_bl,m,s)
        X=np.linspace(-5,40,100)*1e6
        fig=plt.figure(figsize=[15,10])
        plt.rcParams.update({'font.size': 30})
        ax=fig.add_subplot(111)
        ax.xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='x',style='sci')
        # ax.set_yticks([0,1e7,2e7,3e7])
        params, cov=curve_fit(func, 1e6*x[rng], h[rng], p0=p, method='trf', bounds=[[0,0,0,-1e6,0,0,0],[5e4,5e3,1000,2e6,2e6,20e6,5e6]])
        params2=np.array(params)
        params2[2]=1.4*params[2]
        print(np.sqrt(np.mean((h-func(1e6*x, *params))**2)))
        ax.plot(x*1e6,h,'ko')
        ax.errorbar(x*1e6,h,np.sqrt(h), fmt=' ', ecolor='k')
        # ax.set_yscale('log')
        ax.plot(X, func(X, *params), 'b', alpha=0.75, label=r'gain={:3.1f} $10^6$'.format((params[5]-params[3])), linewidth=5)
        ax.plot(X, func(X, *params2), 'g', alpha=0.75, label=r'gain={:3.1f} $10^6$'.format((params[5]-params[3])), linewidth=5)
        # ((0,0,0,-1000,0,0,0), (5e4,5e3,1000,500,1000,2500,1000))
        # try:
        #     params, cov=curve_fit(func, x[rng], h[rng], p0=p, method='trf', bounds=[[0,0,0,-1000,0,0,0],[5e4,5e3,1000,500,1000,2500,10]])
        #     print(params[-2], params[-1])
        #     plt.plot(X, func(X, *params), label=r'gain={:3.1f} $10^6$'.format((params[5]-params[3])))
        # except:
        #     continue
        # print(params[5]-params[3])
        ax.plot(X, func(X, *[0,params[1],0,params[3],params[4], params[5], params[6]]), 'r--', linewidth=5)
        ax.plot(X, func(X, *[0,0,params[2],params[3],params[4], params[5], params[6]]), 'r--', linewidth=5)
        plt.figure()
        plt.plot(x[x>20], h[x>20]-func(1e6*x[x>20], *params), 'k.')
        plt.plot(x[x>20], h[x>20]-func(1e6*x[x>20], *params2), 'r.')
        plt.axhline(0, xmin=0, xmax=1)
        err=np.sqrt(np.diag(cov))
        print('11111111111',params[-1]/params[-2], np.sqrt((err[-1]/params[-2])**2+(params[-1]*err[-2]/params[-2]**2)**2))
        print(params2)
    else:
        def func(x, a,b,m_bl,s_bl, m, s):
            return a*np.exp(-0.5*(x-m_bl)**2/s_bl**2)+b*np.exp(-0.5*(x-m)**2/s**2)
        h, bins, pathc = plt.hist(dc['area'], bins=200, log=True, histtype='step', range=[-800, uper[chn]])
        x=(bins[1:]+bins[:-1])/2
        rng=np.nonzero(np.logical_and(x>-200, x<uper[chn]))[0]
        a=np.amax(h)
        b=np.amax(h[x>1000])
        c=0.25*b
        m_bl=x[np.argmax(h)]
        m=1000
        s_bl=700
        s=1000
        p=(a,c,m_bl,s_bl,m,s)
        params, cov=curve_fit(func, x[rng], h[rng], p0=p, method='trf', bounds=((0,0,-1000,0,0,0), (5e4,5e3,500,1000,2500,1000)))
        plt.plot(x[rng], func(x[rng], *params), label=r'gain={:3.1f} $10^6$'.format((params[4]-params[2])/(R*5e9*elementry_charge*v_to_adc*1e6)))
    #

    #plt.hist(dc['area_l'], bins=300, log=True, histtype='step', range=[-1000,3000], label='l')
    #plt.hist(dc['area_r'], bins=300, log=True, histtype='step', range=[-1000,3000], label='r')
    # plt.hist(dc['area'], bins=100, log=True, histtype='step', range=[-1000,3000])
    # print(dc[dc['area']>1000].head())
    #plt.hist(dc['blw'], bins=100, log=True, histtype='step', range=[0,50])
    # print(d[(d['area']<2000) & (d['area']>1000)].head())
    # plt.axvline(x=bldn[chn], ymin=0, ymax=1)
    # plt.axvline(x=blup[chn], ymin=0, ymax=1)
    #plt.plot(d['event'], d['bl'])
    # plt.xlabel('Enent number')
    # plt.ylabel('Height of maximal peak')
    #plt.hist(d[(d['bl']<blup[chn]) & (d['bl']>bldn[chn])]['bl'], bins=500, log=True)
    #plt.hist(d['blw'], bins=500, log=True)
    #plt.hist(-d[(d['bl']<blup[chn]) & (d['bl']>bldn[chn])]['naive_charge']/1e8, bins=100, log=True, label='bl<3850', histtype='step')
    # plt.hist(-d[d['bl']>3850]['naive_charge']/1e8, bins=100, log=True, label='bl>3850', histtype='step', range=[-36,-35])
    # print(d[d['bl']<3840].head())
    # print(d[(d['bl']>3840) & (d['bl']<3870)].head())
    # print(d[(d['bl']>3880)].head())
    #print(dc[(dc['charge']/1e6>18) & (dc['charge']/1e6<19)].head())

    # plt.hist2d(dc['area'], dc['bl_m'], bins=[100,100], norm=mcolors.PowerNorm(0.3), range=[[-1000,2000], [3675,3725]])
    # print(dc[dc['area']<-100].head())
    # x=np.arange(0,100)
    # plt.plot(x,x,'r.-')
    # plt.plot(x,x+3,'r.-')
    # plt.plot(x,x-3,'r.-')
    #plt.hist(dc['blw_m'], log=True, histtype='step', bins=100)
    # plt.hist(dc['area'], log=True, histtype='step', range=[-500,2500], bins=100)
    #plt.hist(dc['blw_l']/dc['blw_r'], log=True, histtype='step', bins=100, label='blw')
    #plt.hist(dc['bl_l']/dc['bl_r'], log=True, histtype='step', bins=100, label='bl')
    # plt.axvline(x=1.25, ymin=0, ymax=1, linewidth=4, color='k')
    # plt.axvline(x=0.75, ymin=0, ymax=1, linewidth=4, color='k')
    # plt.axhline(y=sumup[chn], xmin=0, xmax=1, linewidth=4, color='k')
    #plt.title('PMT {}'.format(chn))
    #plt.legend(loc='best')
    ax.set_ylim(1,3e3)
    ax.set_xlim(-0.6e7,3.6e7)
    #plt.xlim(-0.5e7, 3.8e7)
    #plt.ylabel('counts')
    ax.set_xlabel('Number of electrons', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.grid()
    #plt.legend(loc='best')
    #plt.savefig('/home/gerak/Desktop/DireXeno/fig/gain{}'.format(chn))
    plt.show()
