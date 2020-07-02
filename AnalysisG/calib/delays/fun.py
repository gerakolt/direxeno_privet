import numpy as np
import matplotlib.pyplot as plt
from classes import Hit, Group
from scipy.stats import poisson, binom
from scipy.optimize import minimize
import sys


def get_spes(pmts):
    spes=[]
    height_cuts=[]
    dh3_cuts=[]
    spk_cuts=[]
    BL=[]
    for pmt in pmts:
        path='/home/gerak/Desktop/DireXeno/050520/pulser/PMT{}/'.format(pmt)
        data=np.load(path+'cuts.npz')
        dh3_cuts.append(data['dh3_cut'])
        spk_cuts.append(data['spk_cut'])
        height_cuts.append(data['height_cut'])
        data=np.load(path+'areas.npz')
        spes.append(data['spe'])
        data=np.load(path+'BL.npz')
        BL.append(data['BL'])
    return spes, BL, height_cuts, dh3_cuts, spk_cuts

def get_delays(pmts):
    path='/home/gerak/Desktop/DireXeno/050520/pulser/delays/'
    delays=[0]
    for pmt in pmts:
        if pmt>0:
            delays.append(np.load(path+'delays_0_{}.npz'.format(pmt))['delay'])
    return delays



def Recon_wf(WF, wf_origin, height_cut, dh3_cut, spk_cut, SPE, Init):
    t=[]
    recon_wf=np.zeros(1000)
    wf=np.array(wf_origin)
    dif=(wf-np.roll(wf,1))
    dif[0]=dif[1]
    dif_bl=np.median(dif[:100])
    dif_blw=np.sqrt(np.mean((dif[:100]-dif_bl)**2))
    SPE_dif=(SPE-np.roll(SPE,1))
    SPE_dif[0]=SPE_dif[1]
    maxis=Init+np.nonzero(np.logical_and(wf[Init:-1]<-height_cut, np.logical_and(wf[Init:-1]<np.roll(wf[Init:-1], -1), wf[Init:-1]<np.roll(wf[Init:-1], 1))))[0]

    counter=0
    while len(maxis)>0 and counter<100:
        counter+=1
        for i, maxi in enumerate(maxis):
            stop=1
            if len(np.nonzero(np.logical_and(wf[:maxi]>-WF.blw, dif[:maxi]>dif_bl-dif_blw))[0])>0:
                left=np.amax(np.nonzero(np.logical_and(wf[:maxi]>-WF.blw, dif[:maxi]>dif_bl-dif_blw))[0])
            else:
                left=0
            if len(np.nonzero(np.logical_and(wf[maxi:]>-WF.blw, dif[maxi:]>dif_bl-dif_blw))[0])>0:
                right=maxi+np.amin(np.nonzero(np.logical_and(wf[maxi:]>-WF.blw, dif[maxi:]>dif_bl-dif_blw))[0])
            else:
                right=len(wf)-1

            if counter>100:
                x=np.arange(1000)
                # plt.figure()
                # plt.plot(x, wf_origin, 'k.')

                fig, (ax1, ax2)=plt.subplots(2,1)
                fig.suptitle('counter>100')
                ax1.plot(x, wf, 'k.')
                ax1.fill_between(x, y1=-WF.blw, y2=0)
                ax1.plot(x[maxi], wf[maxi], 'ro')
                ax1.plot(x[left], wf[left], 'go')
                ax1.plot(x[right], wf[right], 'bo')
                print(wf[maxi]>-spk_cut , (wf[maxi]-wf[maxi-3])/wf[maxi]>dh3_cut)
                ax2.plot(x, dif, 'k.')
                ax2.fill_between(x, y1=dif_bl-dif_blw, y2=dif_bl)
                plt.show()

            if wf[maxi]>-spk_cut and (wf[maxi]-wf[maxi-3])/wf[maxi]>dh3_cut:
                if maxi>len(wf)-5:
                    wf[maxi-3:]=wf[maxi-3]
                else:
                    wf[maxi-3:maxi+4]=wf[maxi-3]
                maxis=np.delete(maxis, 0)
            else:
                break
        if len(maxis)==0:
            break

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

        maxis=Init+np.nonzero(np.logical_and(wf[Init:-1]<-height_cut, np.logical_and(wf[Init:-1]<np.roll(wf[Init:-1], -1), wf[Init:-1]<np.roll(wf[Init:-1], 1))))[0]
    return np.histogram(t, bins=np.arange(1001)-0.5)[0], recon_wf


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
