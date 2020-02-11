import numpy as np
import math
from classes import Peak
from scipy.optimize import curve_fit
from classes import WF, Event
import matplotlib.pyplot as plt
from scipy import special
import scipy.special
import random
import sys

min_rise_t=5
max_rise_t=60
max_height=200
min_height=10
max_chi2=1000


Amp=np.array([0,0,90,38,50,94,34,85,118,56,51,91,83,44,48,55,58,117,0,68])
Amp_std=np.array([0,0,38,22,26,12,15,11,20,24,18,15,24,18,23,23,23,20,0,10])
Tau=np.array([0,0,16,16,17,18,16,19,16,16,16,14,15,16,13,16,16,17,0,15])
Tau_std=np.array([0,0,2,2,2,4,2,5,3,2,2,3,2,2,2,1,1,3,0,3])
Sigma=np.array([0,0,55,57,55,30,55,32,37,54,57,43,56,56,45,55,56,43,0,43])/100
Sigma_std=np.array([0,0,7,7,8,4,7,5,6,8,6,8,6,8,6,6,6,7,0,8])/100

#################### NEED To alow steps of 0.5 #########################

def Reconstruct_WF(self, wf, K):
    # self is WF
    dt=0.5
    if not int(len(wf)/(dt*5))==len(wf)/(dt*5):
        print('int(len(wf)/(dt*5))!=len(wf)/(dt*5)')
        sys.exit()
    recon_wf=np.zeros([len(wf),K])
    h=np.zeros([int(len(wf)/(dt*5)), K])
    chi2=np.ones(K)*1e6
    recon_wf_mean=np.array(wf)*0
    recon_wf_std=np.ones(len(wf))*0
    h_mean=np.zeros(int(len(wf)/(dt*5)))
    h_std=np.zeros(int(len(wf)/(dt*5)))
    for k in range(K):
        recon_wf[:,k], h[:,k], x, chi2[k]=reconstruct_wf(wf, dt, self.bl-self.blw, self.channel)
        #print('Realization number ', k, 'out of',K)
    for k in range(K):
        recon_wf_mean=recon_wf_mean+(recon_wf[:,k]/chi2[k])/np.sum(1/chi2)
        h_mean=h_mean+(h[:,k]/chi2[k])/np.sum(1/chi2)
    for k in range(K):
        recon_wf_std=recon_wf_std+((recon_wf[:,k]-recon_wf_mean)**2/chi2[k])/np.sum(1/chi2)
        h_std=h_std+((h[:,k]-h_mean)**2/chi2[k])/np.sum(1/chi2)
    h_std[h_std==0]=0.05
    return recon_wf_mean, np.sqrt(recon_wf_std), h_mean, np.sqrt(h_std), x



def reconstruct_wf(wf, dt, th, chn):
    waveform=WF(wf, 0)
    waveform.find_hits(wf)
    for hit in waveform.hits:
        hit.find_groups(wf, th)
    waveform.merge_hits()
    for hit in waveform.hits:
        hit.groups=[]
        hit.find_groups(wf, th)

    Recon_wf=np.array(wf)*0
    i=-1
    #while len(waveform.hits[lambda x: x.legit, waveform.hits])>0:
    while len(list(filter(lambda x: x.legit, waveform.hits))):
        i+=1
        if i%100==0 and i>0:
            print('hit number', i , 'out of', len(waveform.hits))
        if i>1000:
            x=np.arange(len(wf))
            plt.plot(x, wf, 'k.-')
            plt.plot(x,Recon_wf, 'r.-')
            plt.fill_between(x, y1=th, y2=0, color='y', alpha=0.3)
            for hit in waveform.hits:
                plt.fill_betweenx(np.arange(np.amin(wf),0), x1=hit.init, x2=hit.fin, color='y', alpha=0.3)
            for peak in waveform.peaks:
                plt.plot(x, func(x, *[peak.peak, peak.amp, peak.tau, peak.sigma]), 'k.-', alpha=0.3)
            plt.title('i>1000')
            plt.show()
            i=0
        # for hit in waveform.hits:
        for hit in list(filter(lambda x: x.legit, waveform.hits)):
            recon_wf=find_peaks(hit, wf[hit.init:hit.fin]-Recon_wf[hit.init:hit.fin], th, chn)
            Recon_wf[hit.init:hit.fin]=Recon_wf[hit.init:hit.fin]+recon_wf
            for peak in hit.peaks:
                waveform.peaks.append(peak)
            #if len(waveform.peaks)>5000:
                # plt.title('waveform.peaks>5000')
                # plt.plot(x, wf, 'k.-')
                # plt.plot(x, Recon_wf, 'g.-')
                # plt.show()
        waveform.hits=[]
        waveform.find_hits(wf-Recon_wf)
        for hit in waveform.hits:
            hit.find_groups(wf-Recon_wf,th)
        waveform.merge_hits()
        for hit in waveform.hits:
            hit.groups=[]
            hit.find_groups(wf-Recon_wf,th)

        # x=np.arange(len(Recon_wf))
        # plt.plot(x, wf, 'k.-')
        # plt.plot(x, Recon_wf, 'r.-')
        # for peak in waveform.peaks:
        #     plt.plot(x, func(x, *[peak.peak, peak.amp, peak.tau, peak.sigma]), 'g.-')
        # plt.show()
    t=[]
    for peak in waveform.peaks:
        t.append(peak.peak)
    h, bins=np.histogram(t, bins=int(len(wf)/(dt*5)), range=[0,len(wf)])



    return Recon_wf, h, 0.5*(bins[1:]+bins[:-1]), np.sum(Recon_wf-wf)**2

def func(x, peak, amp, tau, sigma):
    f=np.zeros(len(x))
    f[x-peak+tau>0]=-amp*np.exp(-0.5*np.log((x[x-peak+tau>0]-peak+tau)/tau)**2/sigma**2)
    return f

def fit_1PE(self, wf, i):
    not_peak=0
    amp_mean=57
    amp_std=24
    p0=[self.groups[i].maxi-self.init, self.groups[0].height, 34, 0.35]
    bounds=[[p0[0]-1, 0.9*p0[1], 0.5*p0[2], 0.5*p0[3]],[p0[0]+1, 1.1*p0[1], 1.5*p0[2], 1.5*p0[3]]]
    recon_wf=np.zeros(len(wf))
    try:
        p, cov = curve_fit(func, np.arange(len(wf))[self.groups[i].left-self.init:self.groups[i].right-self.init],
            wf[self.groups[i].left-self.init:self.groups[i].right-self.init], p0=p0, bounds=bounds)
        n=random.choices([1,2], weights=[np.exp(-0.5*(p[1]-amp_mean)**2/amp_std**2)/(np.sqrt(2*np.pi)*amp_std),
            np.exp(-0.25*(p[1]-2*amp_mean)**2/amp_std**2)/(np.sqrt(np.pi)*amp_std)*
            special.erf(p[1]/(2*amp_std))], k=1)
        x=np.arange(len(wf))
        for j in range(n[0]):
            self.peaks.append(Peak(p[0]+self.init, p[1]/n, p[2], p[3]))
            recon_wf=recon_wf+func(x, *[p[0], p[1]/n[0], p[2], p[3]])
    except:
        not_peak=1
    return not_peak, recon_wf

def find_peaks(self, wf, th, chn):
    # self is hit
    tau=np.random.normal(Tau[chn], Tau_std[chn])
    sigma=np.random.normal(Sigma[chn], Sigma_std[chn])
    amp=np.random.normal(Amp[chn], Amp_std[chn])
    while tau<1 or sigma<0.1 or amp<-th+1:
        #print(tau, sigma, amp, -th+1, chn)
        tau=np.random.normal(Tau[chn], Tau_std[chn])
        sigma=np.random.normal(Sigma[chn], Sigma_std[chn])
        amp=np.random.normal(Amp[chn], Amp_std[chn])
    # a=19.41
    # b=0.5
    # #sigma=np.random.normal(-a*np.log(1-b/tau), 0.05)
    # amp=np.random.normal(57, 24)
    # while sigma<0 or amp<=-th+1:
    #     tau=np.random.normal(34, 11)
    #     #sigma=np.random.normal(0.35, 0.12)
    #     a=19.41
    #     b=0.5
    #     sigma=np.random.normal(-a*np.log(1-b/tau), 0.05)
    #     amp=np.random.normal(57, 24)

    i=0
    while i<len(self.groups):
        #print(i, self.groups)
        if self.groups[i].maxi-self.init<min_rise_t or self.groups[i].height<np.amax([min_height, -th]):
            i+=1
        else:
            if  self.groups[i].height<max_height and self.groups[i].maxi-self.init<max_rise_t:
                not_peak, recon_wf=fit_1PE(self, wf, i)
                if not not_peak:
                    return recon_wf
                else:
                    i+=1
            else:
                break

    x=np.arange(len(wf))
    recon_wf=func(x, *[self.groups[0].maxi-self.init, amp, tau, sigma])
    if len(np.nonzero(wf[:self.groups[i].maxi-self.init+1]<th)[0])==0:
        plt.plot(x, wf, 'k.-', label='wf')
        plt.plot(x[self.groups[i].maxi-self.init], wf[self.groups[i].maxi-self.init], 'ro', label='height={}'.format(self.groups[i].height))
        plt.plot(x[self.groups[i].maxi-self.init], -self.groups[i].height, 'ro')
        plt.plot(x, recon_wf, 'g.-', label='recon_wf')
        plt.fill_between(x, th, 0, color='y', alpha=0.3)
        plt.fill_betweenx(np.arange(-self.groups[i].height,0), self.groups[i].left-self.init, self.groups[i].right-self.init, color='r',
            alpha=0.3, label='th={}'.format(-th))
        plt.title('wf no under th')
        plt.legend()
        plt.show()
        print('group height', np.amin(wf[:self.groups[i].maxi-self.init]), 'th', th)
    if len(np.nonzero(recon_wf[:self.groups[i].maxi-self.init+1]<th)[0])==0:
        plt.plot(x, wf, 'k.-', label='wf')
        plt.plot(x[self.groups[i].maxi-self.init], wf[self.groups[i].maxi-self.init], 'ro', label='height={}'.format(self.groups[i].height))
        plt.plot(x[self.groups[i].maxi-self.init], -self.groups[i].height, 'ro')
        plt.plot(x, recon_wf, 'g.-', label='recon_wf')
        plt.fill_between(x, th, 0, color='y', alpha=0.3)
        plt.fill_betweenx(np.arange(-self.groups[i].height,0), self.groups[i].left-self.init, self.groups[i].right-self.init, color='r',
            alpha=0.3, label='th={}'.format(-th))
        plt.title('recon_wf no under th a {}, tau {}, sigma {}'.format(amp, tau, sigma))
        plt.legend()
        plt.show()
        print('amp',amp, 'th', th)

    pnt_wf=np.amin(np.nonzero(wf[:self.groups[i].maxi-self.init+1]<th)[0])
    pnt_recon_wf=np.amin(np.nonzero(recon_wf[:self.groups[i].maxi-self.init+1]<th)[0])
    recon_wf=np.roll(recon_wf, -(pnt_recon_wf-pnt_wf))
    peak=self.groups[i].maxi-self.init-(pnt_recon_wf-pnt_wf)
    self.peaks.append(Peak(peak+self.init, amp, tau, sigma))
    return recon_wf
