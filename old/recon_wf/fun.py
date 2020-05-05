import numpy as np
import matplotlib.pyplot as plt
from classes import Hit, Group

min_hit_height=10
min_hit_rise_time=5

def find_bl(wf):
    i=25
    j=i
    blw=np.std(wf[i-25:i+25])
    while i<np.argmin(wf):
        i+=1
        std=np.std(wf[i-25:i+25])
        if std<blw:
            blw=std
            j=i
    return np.mean(wf[j-25:j+25]), blw

def import_spe(pmts):
    spes=[]
    path='/home/gerak/Desktop/DireXeno/pulser_190803_46211/SPEs/'
    for pmt in pmts:
        if pmt==3:
            Data = np.load(path+'spe{}.npz'.format(pmt))
            spe=np.sum(Data['spe'], axis=0)/len(Data['spe'][:,0])
            bl=np.median(spe[:25*5])
            spe=np.array(spe-bl)
            spes.append(spe)
        else:
            Data = np.load(path+'spe{}.npz'.format(pmt))
            spe=np.sum(Data['spe'], axis=0)/len(Data['spe'][:,0])
            bl, blw, j=find_bl(spe)
            spe=np.array(spe-bl)
            if pmt==1:
                spe[70*5:]=0
            if pmt==4:
                spe[57*5:]=0
            if pmt==8:
                spe[58*5:]=0
            if pmt==17:
                spe[105*5:]=0
            if pmt==14:
                spe[89*5:]=0
            if pmt==15:
                spe[84*5:]=0
            spes.append(spe)
    return spes

def spe_height(pmts):
    h=[]
    path='/home/gerak/Desktop/DireXeno/pulser_190803_46211/SPEs/'
    for pmt in pmts:
        if pmt==3:
            Data = np.load(path+'spe{}.npz'.format(pmt))
            spe=np.sum(Data['spe'], axis=0)/len(Data['spe'][:,0])
            bl=np.median(spe[:25*5])
        else:
            Data = np.load(path+'spe{}.npz'.format(pmt))
            spe=np.sum(Data['spe'], axis=0)/len(Data['spe'][:,0])
            bl, blw, j=find_bl(spe)
        h.append(np.amin(Data['spe'], axis=1)-bl)
    return h


def Recon_WF(WF, wf, spe, dn, up):
    Recon_wf=np.zeros(len(wf))
    Real_Recon_wf=np.zeros(len(wf))
    t=[]
    while len(WF.hits)>0:
        if len(WF.hits[0].groups)==0:
            Recon_wf[WF.hits[0].init:WF.hits[0].fin]+=(wf-Recon_wf)[WF.hits[0].init:WF.hits[0].fin]
        else:
            i=WF.hits[0].groups[0].maxi
            if i<200 or (wf-Recon_wf)[i]>0.5*np.amin(spe):
                Recon_wf[WF.hits[0].groups[0].left:WF.hits[0].groups[0].right]+=np.array(wf[WF.hits[0].groups[0].left:WF.hits[0].groups[0].right]-
                    Recon_wf[WF.hits[0].groups[0].left:WF.hits[0].groups[0].right])
                N=0
                J=-1
            else:
                recon_wf=np.zeros(1000)
                Chi2=1e17
                J=i
                N=1
                for j in range(np.amax((200, WF.hits[0].groups[0].left)), np.amin((WF.hits[0].groups[0].maxi+5,999))):
                    if (wf-Recon_wf)[j]>0.5*np.amin(spe):
                        temp=1
                    else:
                        if j>np.argmin(spe):
                            recon_wf[j-np.argmin(spe):]=spe[:len(spe)-(j-np.argmin(spe))]
                        else:
                            recon_wf[:len(recon_wf)-(np.argmin(spe)-j)]=spe[np.argmin(spe)-j:]
                        n=1
                        chi2=np.mean((recon_wf[j-dn:j-up]-(wf-Recon_wf)[j-dn:j-up])**2)
                        if chi2<Chi2:
                            Chi2=chi2
                            J=j
                            N=n
                if J>np.argmin(spe):
                    Real_Recon_wf[J-np.argmin(spe):]+=spe[:len(spe)-(J-np.argmin(spe))]
                    Recon_wf[J-np.argmin(spe):]+=spe[:len(spe)-(J-np.argmin(spe))]
                else:
                    Real_Recon_wf[:len(recon_wf)-(np.argmin(spe)-J)]+=spe[np.argmin(spe)-J:]
                    Recon_wf[:len(recon_wf)-(np.argmin(spe)-J)]+=spe[np.argmin(spe)-J:]
                for i in range(N):
                    t.append(J)

        WF.hits=[]
        find_hits(WF, wf-Recon_wf)
    return Real_Recon_wf, np.sum(((Real_Recon_wf-wf)[200:])**2), t



def Show_Recon_WF(WF, wf, spe, dn, up, pl, ID):
    Recon_wf=np.zeros(len(wf))
    Real_Recon_wf=np.zeros(len(wf))
    t=[]
    while len(WF.hits)>0:
        if len(WF.hits[0].groups)==0:
            Recon_wf[WF.hits[0].init:WF.hits[0].fin]+=(wf-Recon_wf)[WF.hits[0].init:WF.hits[0].fin]
        else:
            i=WF.hits[0].groups[0].maxi
            if i<200 or (wf-Recon_wf)[i]>0.5*np.amin(spe):
                Recon_wf[WF.hits[0].groups[0].left:WF.hits[0].groups[0].right]+=np.array(wf[WF.hits[0].groups[0].left:WF.hits[0].groups[0].right]-Recon_wf[WF.hits[0].groups[0].left:WF.hits[0].groups[0].right])
                N=0
                J=-1
            else:
                recon_wf=np.zeros(1000)
                Chi2=1e17
                J=i
                N=1
                for j in range(np.amax((200, WF.hits[0].groups[0].left)), WF.hits[0].groups[0].maxi+5):
                    if (wf-Recon_wf)[j]>0.5*np.amin(spe):
                        temp=1
                    else:
                        if j>np.argmin(spe):
                            recon_wf[j-np.argmin(spe):]=spe[:len(spe)-(j-np.argmin(spe))]
                        else:
                            recon_wf[:len(recon_wf)-(np.argmin(spe)-j)]=spe[np.argmin(spe)-j:]
                        n=1
                        chi2=np.mean((recon_wf[j-dn:j-up]-(wf-Recon_wf)[j-dn:j-up])**2)
                        print(j, chi2, WF.hits[0].groups[0].maxi)
                        if chi2<Chi2:
                            Chi2=chi2
                            J=j
                            N=n
                if J>np.argmin(spe):
                    Real_Recon_wf[J-np.argmin(spe):]+=spe[:len(spe)-(J-np.argmin(spe))]
                    Recon_wf[J-np.argmin(spe):]+=spe[:len(spe)-(J-np.argmin(spe))]
                else:
                    Real_Recon_wf[:len(recon_wf)-(np.argmin(spe)-J)]+=spe[np.argmin(spe)-J:]
                    Recon_wf[:len(recon_wf)-(np.argmin(spe)-J)]+=spe[np.argmin(spe)-J:]
                for i in range(N):
                    t.append(J)

                if pl==2:
                    x=np.arange(1000)/5
                    fig=plt.figure(figsize=(20,10))
                    plt.title('ID={}'.format(ID))
                    ax=fig.add_subplot(211)
                    ax.plot(x, wf, 'k.')
                    ax.plot(x, Real_Recon_wf, 'r.-')
                    ax.fill_between(x[J-dn:J-up], y1=np.amin(spe), y2=0, color='y', alpha=0.2)
                    if J>np.argmin(spe):
                        ax.plot(x[J-np.argmin(spe):], spe[:len(spe)-(J-np.argmin(spe))], 'g--')
                    else:
                        ax.plot(x[:len(recon_wf)-(np.argmin(spe)-J)], spe[np.argmin(spe)-J:], 'g--')

                    ax=fig.add_subplot(212)
                    ax.plot(x[1:], wf[1:]-wf[:-1], 'k.')
                    ax.plot(x[J], 0, 'yo')
                    ax.fill_between(x[J-dn:J-up], y1=np.amin(spe), y2=0, color='y', alpha=0.2)
                    if J>np.argmin(spe):
                        ax.plot(x[J-np.argmin(spe):][1:], spe[:len(spe)-(J-np.argmin(spe))][1:]-spe[:len(spe)-(J-np.argmin(spe))][:-1], 'g.')
                    else:
                        ax.plot(x[:len(recon_wf)-(np.argmin(spe)-J)][1:], spe[np.argmin(spe)-J:][1:]-spe[np.argmin(spe)-J:][:-1], 'g.')
                    plt.show()

        WF.hits=[]
        find_hits(WF, wf-Recon_wf)

    if pl==1:
        x=np.arange(1000)/5
        fig=plt.figure(figsize=(20,10))
        plt.title('ID={}'.format(ID))
        plt.plot(x, wf, 'k.')
        plt.plot(x, Real_Recon_wf, 'r.-', label='{} PEs'.format(len(t)))
        plt.plot(x, wf-Real_Recon_wf, 'g--')
        plt.legend()
        plt.show()
    return Real_Recon_wf, np.sum(((Real_Recon_wf-wf)[200:])**2), t


def find_hits(self, wf):
    dif=(wf-np.roll(wf,1))[1:]
    dif=np.append(dif[0], dif)
    dif_bl, dif_blw=find_bl_dif(dif)
    prev_len_out_of_hit=1e6
    out_of_hit=np.arange(len(wf))
    counter=0
    while len(out_of_hit)>0 and np.amin(wf[out_of_hit])<-self.blw and np.any(dif[out_of_hit]>dif_bl+dif_blw)\
        and len(out_of_hit)<prev_len_out_of_hit:
    #while len(out_of_hit)>0 and np.amin(wf[out_of_hit])<-self.blw:
        maxi=out_of_hit[np.argmin(wf[out_of_hit])]
        left=0
        right=len(wf)-1
        for hit in self.hits:
            if hit.fin>left and hit.init<maxi:
                left=hit.fin
            if hit.init<right and hit.init>maxi:
                right=hit.init

        if len(np.nonzero(wf[left:maxi]>-self.blw)[0])>0:
            init_blw=left+np.amax(np.nonzero(wf[left:maxi]>-self.blw)[0])
            if len(np.nonzero(dif[left:maxi]>dif_bl-dif_blw)[0]):
                init_dif=left+np.amax(np.nonzero(dif[left:maxi]>dif_bl-dif_blw)[0])
                init=np.amin([init_blw, init_dif])
            else:
                init=left
        else:
            init=left

        if len(np.nonzero(wf[maxi:right]>-self.blw)[0])>0:
            fin_blw=maxi+np.amin(np.nonzero(wf[maxi:right]>-self.blw)[0])
            if len(np.nonzero(dif[maxi:right]<dif_bl+dif_blw)[0]):
                fin_dif=maxi+np.amin(np.nonzero(dif[maxi:right]<dif_bl+dif_blw)[0])
                fin=np.amax([fin_blw, fin_dif])
            else:
                fin=right
        else:
            fin=right


        if np.any(dif[init:maxi]<dif_bl-dif_blw) and wf[maxi]<-min_hit_height and maxi-init>min_hit_rise_time:
            hit=Hit(init,fin)
            hit.area=-np.sum(wf[init:fin+1])
            hit.height=-np.amin(wf[init:fin+1])
            find_groups(hit, wf, dif, self.blw)
            self.hits.append(hit)
        prev_len_out_of_hit=len(out_of_hit)
        for i in np.arange(init,fin+1):
            out_of_hit=np.delete(out_of_hit, np.nonzero(out_of_hit==i)[0])

        counter+=1
    self.hits=sorted(self.hits, key=lambda hit: hit.init)
    merge_hits(self, wf, self.blw)
    for hit in self.hits:
        hit.groups=sorted(hit.groups, key=lambda grp: grp.maxi)



def find_groups(self, wf, dif, blw):
    maxi=np.nonzero(np.logical_and(wf[:-1]<-blw, np.logical_and(dif[:-1]<0,dif[1:]>0)))[0]
    vals=np.nonzero(np.logical_and(dif[:-1]>0,dif[1:]<0))[0]

    for m in maxi[np.logical_and(maxi>self.init, maxi<self.fin)]:
        if np.any(vals<m):
            left=np.amax(vals[vals<m])
        else:
            left=0
        if np.any(vals>m):
            right=np.amin(vals[vals>m])
        else:
            right=len(wf)-1

        if left<self.init:
            self.init=left
        if right>self.fin:
            self.fin=right
        if -wf[m]>min_hit_height:
            self.groups.append(Group(m, left, right, -wf[m]))

def overlap(self, wf):
    #self is a wf
    for i in range(len(self.hits)):
        for j in range(i+1, len(self.hits)):
            if ((self.hits[i].init<self.hits[j].init and self.hits[i].fin>self.hits[j].init) or
                (self.hits[j].init<self.hits[i].init and self.hits[j].fin>self.hits[i].init)):
                x=np.arange(len(wf))
                plt.plot(x, wf, 'k.')
                plt.fill_betweenx(np.arange(np.amin(wf),0), x1=self.hits[i].init, x2=self.hits[i].fin, color='y', alpha=0.3)
                plt.fill_betweenx(np.arange(np.amin(wf),0), x1=self.hits[j].init, x2=self.hits[j].fin, color='r', alpha=0.3)
                plt.title('overlap')
                plt.show()


def find_init10(self, wf):
    hit=sorted(self.hits, key=lambda hit: hit.height)[-1]
    if len(list(filter(lambda grp: grp.height>50, hit.groups)))>0:
        grp=list(filter(lambda grp: grp.height>50, hit.groups))[0]
        if len(np.nonzero(wf[hit.init:grp.maxi]<-0.1*grp.height)[0])>0:
            init10=hit.init+np.amin(np.nonzero(wf[hit.init:grp.maxi]<-0.1*grp.height)[0])
            self.init10=init10
        else:
            self.init10=1001


def find_bl_dif(wf):
    i=25
    j=i
    blw=np.sqrt(np.mean(wf[i-25:i+25]**2))
    while i+25<len(wf):
        i+=1
        std=np.sqrt(np.mean(wf[i-25:i+25]**2))
        if std<blw:
            blw=std
            j=i
    return np.mean(wf[j-25:j+25]), blw



def merge_hits(self, wf, blw):
    i=0
    while i<len(self.hits):
        hit1=self.hits[i]
        j=i+1
        while j<len(self.hits):
            hit2=self.hits[j]
            if (hit1.fin==hit2.init or hit1.init==hit2.fin):
                hit1.init=np.amin([hit1.init, hit2.init])
                hit1.fin=np.amax([hit1.fin, hit2.fin])
                hit1.height=np.amax([hit1.height, hit2.height])
                hit1.area+=hit2.area
                for grp in hit2.groups:
                    hit1.groups.append(grp)
                self.hits.remove(hit2)
            else:
                j+=1
        i+=1
