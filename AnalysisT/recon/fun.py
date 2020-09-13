import numpy as np
import matplotlib.pyplot as plt
from classes import Hit, Group
from scipy.stats import poisson, binom
from scipy.optimize import minimize
import sys


def get_spes(pmts):
    spes=[]
    height_cuts=[]
    dh3_cut=[]
    BL=[]
    for pmt in pmts:
        path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt)
        data=np.load(path+'cuts.npz')
        dh3_cut.append(data['dh3_cut'])
        height_cuts.append(data['height_cut'])
        data=np.load(path+'areas.npz')
        spes.append(data['spe'])
        # BL.append(np.load(path+'BL.npz')['BL'])
        BL.append(np.zeros(1000))
    return spes, height_cuts, dh3_cut, BL

def get_delays(pmts):
    path='/home/gerak/Desktop/DireXeno/190803/pulser/DelayRecon/'
    delays=[]
    for pmt in pmts:
        if pmt!=14:
            data=np.load(path+'delay_hist{}-14.npz'.format(pmt))
            delays.append(data['m'])
        else:
            delays.append(0)
    return delays



def Recon_wf(WF, wf_origin, height_cut, dh3_cut, SPE, Init):
    t=[]
    recon_wf=np.zeros(1000)
    wf=np.array(wf_origin)
    dif=(wf-np.roll(wf,1))
    dif[0]=dif[1]
    dif_bl=np.median(dif[:Init])
    dif_blw=np.sqrt(np.mean((dif[:Init]-dif_bl)**2))
    SPE_dif=(SPE-np.roll(SPE,1))
    SPE_dif[0]=SPE_dif[1]
    # maxis=Init+np.nonzero(np.logical_and(wf[Init:-1]<-height_cut, np.logical_and(wf[Init:-1]<np.roll(wf[Init:-1], -1), wf[Init:-1]<np.roll(wf[Init:-1], 1))))[0]
    maxis=Init+np.nonzero(np.logical_and(wf[Init:990]<-height_cut, np.logical_and(wf[Init:990]<np.roll(wf[Init:990], -1), wf[Init:990]<np.roll(wf[Init:990], 1))))[0]
    maxis=maxis[np.logical_and((wf[maxis]-wf[maxis-3])/wf[maxis]<dh3_cut, (wf[maxis]-wf[maxis+3])/wf[maxis]<dh3_cut)]
    counter=0
    while len(maxis)>0 and counter<1000:
        maxi=maxis[0]
        if len(np.nonzero(np.logical_and(wf[:maxi]>-WF.blw, dif[:maxi]>dif_bl-dif_blw))[0])>0:
            left=np.amax(np.nonzero(np.logical_and(wf[:maxi]>-WF.blw, dif[:maxi]>dif_bl-dif_blw))[0])
        else:
            left=0
        if len(np.nonzero(np.logical_and(wf[maxi:]>-WF.blw, dif[maxi:]>dif_bl-dif_blw))[0])>0:
            right=maxi+np.amin(np.nonzero(np.logical_and(wf[maxi:]>-WF.blw, dif[maxi:]>dif_bl-dif_blw))[0])
        else:
            right=len(wf)-1

        counter+=1
    #     for maxi in maxis:
    #         stop=1
    #         # print(WF.blw, dif_bl, dif_blw)
    #         if len(np.nonzero(np.logical_and(wf[:maxi]>-WF.blw, dif[:maxi]>dif_bl-dif_blw))[0])>0:
    #             left=np.amax(np.nonzero(np.logical_and(wf[:maxi]>-WF.blw, dif[:maxi]>dif_bl-dif_blw))[0])
    #         else:
    #             left=0
    #         if len(np.nonzero(np.logical_and(wf[maxi:]>-WF.blw, dif[maxi:]>dif_bl-dif_blw))[0])>0:
    #             right=maxi+np.amin(np.nonzero(np.logical_and(wf[maxi:]>-WF.blw, dif[maxi:]>dif_bl-dif_blw))[0])
    #         else:
    #             right=len(wf)-1
    #
    #         if (wf[maxi-3]-wf[maxi])/wf[maxi]<dh3_cut:
    #             stop=0
    #             break
    #
    #     if stop:
    #         break
        Chi2=1e9
        I=np.argmin(SPE)
        for i in range(left+10, maxi+10):
            spe=np.roll(SPE, i-np.argmin(SPE))
            spe_dif=np.roll(SPE_dif, i-np.argmin(SPE_dif))
            chi2=np.sum((spe[left:i-5]-wf[left:i-5])**2)+np.sum((spe_dif[left:i-5]-dif[left:i-5])**2)
            if chi2<Chi2:
                Chi2=chi2
                I=i
        t.append(I)
        spe=np.roll(SPE, I-np.argmin(SPE))
        if left>np.argmin(spe):
            break
        if np.amin(spe)<np.amin(wf[left:np.argmin(spe)]):
            spe=np.amin(wf[left:np.argmin(spe)])*spe/(np.amin(spe))
        spe_dif=np.roll(SPE_dif, I-np.argmin(SPE_dif))

        wf[left:]-=spe[left:]
        wf[wf>0]=0
        recon_wf[left:]+=spe[left:]
        dif=(wf-np.roll(wf,1))
        dif[0]=dif[1]

        # plt.figure()
        # plt.plot(wf, 'k.')
        # plt.plot(recon_wf, 'r.')
        # plt.show()

        # maxis=Init+np.nonzero(np.logical_and(wf[Init:-1]<-height_cut, np.logical_and(wf[Init:-1]<np.roll(wf[Init:-1], -1), wf[Init:-1]<np.roll(wf[Init:-1], 1))))[0]
        maxis=Init+np.nonzero(np.logical_and(wf[Init:990]<-height_cut, np.logical_and(wf[Init:990]<np.roll(wf[Init:990], -1), wf[Init:990]<np.roll(wf[Init:990], 1))))[0]
        maxis=maxis[np.logical_and((wf[maxis]-wf[maxis-3])/wf[maxis]<dh3_cut, (wf[maxis]-wf[maxis+3])/wf[maxis]<dh3_cut)]
    return np.histogram(t, bins=np.arange(201)*5)[0], recon_wf


def find_hits(self, wf_origin, Init, height_cut, rise_time_cut):
    wf=np.array(wf_origin)
    dif=(wf-np.roll(wf,1))
    dif[0]=dif[1]
    dif_bl=np.median(dif[:Init])
    dif_blw=np.sqrt(np.mean((dif[:Init]-dif_bl)**2))

    while np.amin(wf)<-height_cut:
        maxi=np.argmin(wf)
        if len(np.nonzero(np.logical_and(wf[:maxi]>-self.blw, dif[:maxi]>dif_bl-dif_blw))[0])>0:
            left=np.amax(np.nonzero(np.logical_and(wf[:maxi]>-self.blw, dif[:maxi]>dif_bl-dif_blw))[0])
        else:
            left=0
        if len(np.nonzero(np.logical_and(wf[maxi:]>-self.blw, dif[maxi:]>dif_bl-dif_blw))[0])>0:
            right=maxi+np.amin(np.nonzero(np.logical_and(wf[maxi:]>-self.blw, dif[maxi:]>dif_bl-dif_blw))[0])
        else:
            right=len(wf)
        if maxi-left>rise_time_cut:
            hit=Hit(left, right)
            hit.area=-np.sum(wf[left:right])
            self.hits.append(hit)
        wf[left:right]=0
