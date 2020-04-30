import numpy as np
import matplotlib.pyplot as plt
from classes import Hit, Group, WaveForm, Peak

min_hit_height=10
min_hit_rise_time=5


def find_hits(self, wf):
    dif=(wf-np.roll(wf,1))[1:]
    dif=np.append(dif[0], dif)
    dif_bl, dif_blw, j=find_bl_dif(dif)
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
    return np.mean(wf[j-25:j+25]), blw, j



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





def Recon_WFs(file, spe, delay, dn, up, h_init, h_cuts, pmts, first_id):
    recon_wfs=np.zeros((len(pmts), 1000))
    recon_Hs=np.zeros((len(pmts), 1000))
    WFS=np.zeros((len(pmts), 1000))
    PMT_num=20
    time_samples=1024
    Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*first_id)
    id=first_id-1
    while(1):
        id+=1
        Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
        if len(Data)<(PMT_num+4)*(time_samples+2):
            break
        Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
        wfs=Data[2:1002,pmts+2]
        wfs=wfs-np.median(wfs[:40], axis=0)
        BLW=np.sqrt(np.mean(wfs[:40]**2, axis=0))
        Chi2=0
        for i, [recon_wf, chi2, recon_H, wf] in enumerate(Recon_WF(wfs.T, spe, dn, up, h_init, h_cuts, delay, BLW)):
            WFS[i]=wf
            recon_wfs[i]=recon_wf
            Chi2+=chi2
            recon_Hs[i]=recon_H
        yield [recon_wfs, Chi2, recon_Hs, WFS, BLW]

def Recon_WF(wfs, SPE, dn, up, h_init, h_cuts, delay, BLW):
    for i, wf in enumerate(wfs):
        wf=np.roll(wf, -int(5*delay[i]))
        wf_copy=np.array(wf)
        spe=SPE[i]
        h_cut=h_cuts[i]
        Recon_wf=np.zeros(len(wf))
        Real_Recon_wf=np.zeros(len(wf))
        t=[]
        blw=BLW[i]
        WF=WaveForm(100, blw)
        find_hits(WF, wf)
        # if len(sorted(filter(lambda hit: hit.height>h_init, WF.hits), key=lambda hit: hit.init))==0:
        #     plt.figure()
        #     plt.plot(wf, 'k.')
        #     plt.title('no hits')
        #     plt.show()
        # init=sorted(filter(lambda hit: hit.height>h_init, WF.hits), key=lambda hit: hit.init)[0].init
        init=0
        while len(WF.hits)>0:
            if len(WF.hits[0].groups)==0:
                Recon_wf[WF.hits[0].init:WF.hits[0].fin]+=(wf-Recon_wf)[WF.hits[0].init:WF.hits[0].fin]
            else:
                i=WF.hits[0].groups[0].maxi
                if i<init or (wf-Recon_wf)[i]>-h_cut:
                    Recon_wf[WF.hits[0].groups[0].left:WF.hits[0].groups[0].right]+=np.array(wf[WF.hits[0].groups[0].left:WF.hits[0].groups[0].right]-
                        Recon_wf[WF.hits[0].groups[0].left:WF.hits[0].groups[0].right])
                    N=0
                    J=-1
                else:
                    recon_wf=np.zeros(1000)
                    Chi2=1e17
                    J=i
                    N=1
                    for j in range(np.amax((init, WF.hits[0].groups[0].left)), np.amin((WF.hits[0].groups[0].maxi+5,999))):
                        if (wf-Recon_wf)[j]>-h_cut:
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
        yield [Real_Recon_wf, np.sum(((Real_Recon_wf-wf)[init:])**2), np.histogram(t, bins=1000,
                range=[-0.5, 999.5])[0], wf_copy]


def do_dif(smd):
    return np.roll(smd,1)/2-np.roll(smd,-1)/2
    # if len(np.shape(smd))>1:
    #     return (np.roll(smd,1,axis=1)-np.roll(smd,-1,axis=1))/2
    # else:
    #     return (np.roll(smd,1)-np.roll(smd,-1))/2
