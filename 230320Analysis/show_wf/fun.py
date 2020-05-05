import numpy as np
import matplotlib.pyplot as plt
from classes import Hit, Group

def smd(wf):
    s=np.zeros(200)
    for i in range(200):
        s[i]=np.sum(wf[5*i:5*(i+1)])
    return s

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
        # plt.figure()
        # x=np.arange(1000)
        # plt.plot(x, wf, 'r.-')
        # plt.plot(x, wf_origin, 'k.-', alpha=0.2)
        # plt.fill_between(x, y1=-self.blw, y2=0, alpha=0.3)
        # plt.axhline(-height_cut, xmin=0, xmax=1, color='k')
        # plt.fill_between(x[left:right], y1=np.amin(wf), y2=0, alpha=0.3)
        # plt.show()
        if maxi-left>rise_time_cut:
            hit=Hit(left, right)
            hit.area=-np.sum(wf[left:right])
            self.hits.append(hit)
        wf[left:right]=0



def show_wf(WF, wf):
    x=np.arange(1000)
    plt.figure()
    plt.plot(x, wf, 'k.-', label='wf')
    plt.fill_between(x, y1=-WF.blw, y2=0, alpha=0.3)
    for hit in WF.hits:
        plt.fill_between(x[hit.init:hit.fin], y1=np.amin(wf), y2=0, alpha=0.3, label='init={}, area={}'.format(hit.init, hit.area))
    plt.legend()
    plt.show()


def Recon_wf(WF, wf_origin, Init, height_cut, rise_time_cut, SPE):
    t=[]
    recon_wf=np.zeros(1000)
    wf=np.array(wf_origin)
    dif=(wf-np.roll(wf,1))
    dif[0]=dif[1]
    dif_bl=np.median(dif[:Init])
    dif_blw=np.sqrt(np.mean((dif[:Init]-dif_bl)**2))
    SPE_dif=(SPE-np.roll(SPE,1))
    SPE_dif[0]=SPE_dif[1]
    maxis=Init+np.nonzero(np.logical_and(wf[Init:-1]<-height_cut, np.logical_and(wf[Init:-1]<np.roll(wf[Init:-1], -1), wf[Init:-1]<np.roll(wf[Init:-1], 1))))[0]
    while len(maxis)>0:
        # maxi=np.amin(maxis)
        for maxi in maxis:
            stop=1
            if len(np.nonzero(np.logical_and(wf[:maxi]>-WF.blw, dif[:maxi]>dif_bl-dif_blw))[0])>0:
                left=np.amax(np.nonzero(np.logical_and(wf[:maxi]>-WF.blw, dif[:maxi]>dif_bl-dif_blw))[0])
            else:
                left=0
            if len(np.nonzero(np.logical_and(wf[maxi:]>-WF.blw, dif[maxi:]>dif_bl-dif_blw))[0])>0:
                right=maxi+np.amin(np.nonzero(np.logical_and(wf[maxi:]>-WF.blw, dif[maxi:]>dif_bl-dif_blw))[0])
            else:
                right=len(wf)

            if maxi-left>rise_time_cut and right-left>2*rise_time_cut:
                stop=0
                break

        if stop:
            break
        Chi2=1e9
        I=np.argmin(SPE)
        for i in range(left+10, maxi+10):
            spe=np.roll(SPE, i-np.argmin(SPE))
            # spe_dif=np.roll(SPE_dif, i-np.argmin(SPE_dif))
            chi2=np.sum((spe[left:i-5]-wf[left:i-5])**2)
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


        # fig=plt.figure(figsize=(20,10))
        # ax=fig.add_subplot(211)
        # x=np.arange(1000)
        # ax.plot(x, wf, 'k.-')
        # ax.plot(x, spe, 'r.-')
        # ax.plot(x[left:I-5], wf[left:I-5], 'k+')
        # ax.plot(x[left:I-5], spe[left:I-5], 'r+')
        # ax.axhline(0, xmin=0.2, xmax=0.8, color='y')
        # ax.fill_between(x[left:right+1], y1=wf[left:right+1], y2=0, alpha=0.3)
        # ax.fill_between(x[left+10:maxi+10], y1=wf[left+10:maxi+10], y2=0, alpha=0.3)
        # ax.fill_between(x, y1=-WF.blw, y2=0, alpha=0.3)
        # ax.axhline(-height_cut, xmin=0, xmax=1)
        #
        # ax=fig.add_subplot(212)
        # ax.plot(x, dif, 'k.-')
        # ax.plot(x, spe_dif, 'r.-')
        # ax.fill_between(x[left:right+1], y1=dif[left:right+1], y2=0, alpha=0.3)
        # plt.show()

        wf[left:]-=spe[left:]
        wf[wf>0]=0
        recon_wf[left:]+=spe[left:]
        dif=(wf-np.roll(wf,1))
        dif[0]=dif[1]

        maxis=Init+np.nonzero(np.logical_and(wf[Init:-1]<-height_cut, np.logical_and(wf[Init:-1]<np.roll(wf[Init:-1], -1), wf[Init:-1]<np.roll(wf[Init:-1], 1))))[0]
    return np.histogram(t, bins=np.arange(1001)-0.5)[0], recon_wf
