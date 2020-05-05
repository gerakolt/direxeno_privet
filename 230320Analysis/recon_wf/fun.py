import numpy as np
import matplotlib.pyplot as plt
from classes import Hit, Group
from scipy.stats import poisson, binom
from scipy.optimize import minimize
import sys


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

def fit_spectra(x, a, m, Spe, p0):
    print(a, m ,Spe, p0)
    q=(1-p0)
    H=np.zeros(len(x))
    dx=x[1]-x[0]
    if Spe==0:
        for i in range(len(x)):
            for n in range(int(np.ceil(x[i]-0.5*dx)), int(np.ceil(x[i]+0.5*dx))):
                H[i]+=a*poisson.pmf(n, m)
        return H
    def make_P(Spe, ns):
        P=np.zeros((500, 500))
        P[0,0]=1
        for i in range(len(P[:,0])):
            r=np.linspace(i-0.5,i+0.5,1000)
            dr=r[1]-r[0]
            P[i,1]=dr*np.sum(np.exp(-0.5*(r-1)**2/Spe**2))/(np.sqrt(2*np.pi)*Spe)
        for j in range(2, len(P[0,:])):
            for i in range(len(P[:,0])):
                P[i,j]=np.sum(P[:i+1,1]*np.flip(P[:i+1,j-1]))
        for i in range(1, len(P[:,0])):
            P[i]=np.sum(binom.pmf(np.arange(i+1), i, q)*P[:i+1].T, axis=1)

        if np.isnan(np.any(P)):
            print('P is nan')
            sys.exit()
        if np.isinf(np.any(P)):
            print('P is inf')
            sys.exit()
        if np.any(np.sum(P, axis=0)==0):
            print(Spe)
            print('P=0')
            sys.exit()
        P=P/np.sum(P, axis=0)

        return P
    P=make_P(Spe, np.arange(int(np.floor(x[-1]+0.5*dx))))
    ns=np.arange(np.shape(P)[0])
    h=np.matmul(P, poisson.pmf(ns, m))
    for i in range(len(x)):
        for n in range(int(np.ceil(x[i]-0.5*dx)), int(np.ceil(x[i]+0.5*dx))):
            H[i]+=a*h[n]
    return H
