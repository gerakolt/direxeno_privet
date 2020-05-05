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


def Recon_wf(WF, wf_origin, Init, height_cut, rise_time_cut, SPE, dn, up):
    t=[]
    recon_wf=np.zeros(1000)
    wf=np.array(wf_origin)
    dif=(wf-np.roll(wf,1))
    dif[0]=dif[1]
    dif_bl=np.median(dif[:Init])
    dif_blw=np.sqrt(np.mean((dif[:Init]-dif_bl)**2))

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
        I=0
        N=1
        for i in range(left+up, maxi+up):
            for n in range(1,2):
                spe=np.roll(SPE, i-np.argmin(SPE))
                chi2=np.sum((n*spe[i-n*up:i-dn]-wf[i-n*up:i-dn])**2)
                if chi2<Chi2:
                    Chi2=chi2
                    I=i
                    N=n
        t.append(I*np.ones(N))
        spe=np.roll(SPE, I-np.argmin(SPE))
        wf[left:]-=N*spe[left:]
        recon_wf[left:]+=N*spe[left:]
        dif=(wf-np.roll(wf,1))
        dif[0]=dif[1]

        plt.figure(figsize=(20,10))
        x=np.arange(1000)
        w=np.array(wf)
        w[left:]+=N*spe[left:]
        plt.plot(x, w, 'k.-')
        plt.plot(x, N*spe, 'r.-')
        plt.axhline(0, xmin=0, xmax=1, color='k')
        plt.plot(x[np.argmin(spe)-N*up: np.argmin(spe)-dn], N*spe[np.argmin(spe)-N*up: np.argmin(spe)-dn], 'g.-')
        plt.fill_between(x[left:right+1], y1=w[left:right+1], y2=0, alpha=0.3)
        plt.show()

        maxis=Init+np.nonzero(np.logical_and(wf[Init:-1]<-height_cut, np.logical_and(wf[Init:-1]<np.roll(wf[Init:-1], -1), wf[Init:-1]<np.roll(wf[Init:-1], 1))))[0]
    return np.histogram(t, bins=np.arange(1001)-0.5), recon_wf
