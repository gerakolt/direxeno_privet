import numpy as np
import matplotlib.pyplot as plt
from classes import Hit, Group
from scipy.stats import poisson, binom
from scipy.optimize import minimize
import sys

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
        for i in range(left+up, maxi+up):
            spe=np.roll(SPE, i-np.argmin(SPE))
            chi2=np.sum((spe[i-up:i-dn]-wf[i-up:i-dn])**2)
            if chi2<Chi2:
                Chi2=chi2
                I=i
        t.append(I)
        spe=np.roll(SPE, I-np.argmin(SPE))
        wf[left:]-=spe[left:]
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


def model(x, H, xa, Ha, p0):

    def make_P(Spe, ns):
        P=np.zeros((ns[-1]+10, ns[-1]+10))
        P[0,0]=1
        for i in range(len(P[:,0])):
            r=np.linspace(i-0.5,i+0.5,1000)
            dr=r[1]-r[0]
            P[i,1]=dr*np.sum(np.exp(-0.5*(r-1)**2/Spe**2))/(np.sqrt(2*np.pi)*Spe)
        for j in range(2, len(P[0,:])):
            for i in range(len(P[:,0])):
                P[i,j]=np.sum(P[:i+1,1]*np.flip(P[:i+1,j-1]))
        return P

    def L(p):
        [a, m, b,c,d,m_bl, s_bl, ma, Spe]=p
        P=make_P(Spe, x)
        nns=np.arange(np.shape(P)[0])
        h=a*np.matmul(P, poisson.pmf(nns, m))[x]
        sa=Spe*m_bl
        ha=b*np.exp(-0.5*(xa-m_bl)**2/s_bl**2)+c*np.exp(-0.5*(xa-(m_bl+ma))**2/(s_bl**2+sa**2))+d*np.exp(-0.5*(xa-(m_bl+2*ma))**2/(s_bl**2+2*sa**2))

        l=0
        for i in range(len(h)):
            l+=H[i]*np.log(h[i])-H[i]*np.log(H[i])+H[i]-h[i]
        for i in range(len(ha)):
            l+=Ha[i]*np.log(ha[i])-Ha[i]*np.log(Ha[i])+Ha[i]-ha[i]
        print(-l)
        return -l



    p=minimize(L, p0, method='Nelder-Mead', options={'disp':True, 'maxfev':10000})
    [a, m, b,c,d,m_bl, s_bl, ma, Spe]=p.x
    P=make_P(Spe, x)
    nns=np.arange(np.shape(P)[0])
    sa=Spe*m_bl
    return a*np.matmul(P, poisson.pmf(nns, m))[x], b*np.exp(-0.5*(xa-m_bl)**2/s_bl**2)+c*np.exp(-0.5*(xa-(m_bl+ma))**2/(s_bl**2+sa**2))+d*np.exp(-0.5*(xa-(m_bl+2*ma))**2/(s_bl**2+2*sa**2))
