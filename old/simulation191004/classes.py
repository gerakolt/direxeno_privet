import numpy as np
from types import MethodType

hit_min_height=10
rise_t=5

class DataSet:
    def __init__(self, label, nor):
        self.events=[]
        self.label=label
        self.number_of_realizations=nor
        self.first_event=0
        self.last_event=0
        self.posible_angles=np.array([])

class Event:
    def __init__(self,id):
        self.id=id
        self.sat=0
        self.wf=[]
        self.peaks=[]
        self.peaks_err=[]
        self.decay_const=0
        self.decay_s=0
        self.decay_T=0
        self.decay_tau=0
        self.dt=0
        self.legit=1
        self.angles=[]
        self.angles_t=[]
        self.PEs=0


class WF:
    def __init__(self,channel):
        self.hits=[]
        self.peaks=[]
        self.channel=channel
        self.pmt=0
        self.PE=[]
        self.PE_std=[]
        self.recon_chi2=0
        self.T=[]
        self.chi2=0

    def merge_hits(self):
        for hit1 in self.hits:
            for hit2 in self.hits:
                if hit1!=hit2 and (hit1.fin==hit2.init or hit1.init==hit2.fin):
                    hit1.init=np.amin([hit1.init, hit2.init])
                    hit1.fin=np.amax([hit1.fin, hit2.fin])
                    for grp in hit2.groups:
                        hit1.groups.append(grp)
                    self.hits.remove(hit2)
        for i in range(len(self.hits)):
            for j in range(i):
                if self.hits[i].init<self.hits[j].init:
                    hit=self.hits[j]
                    self.hits[j]=self.hits[i]
                    self.hits[i]=hit
                    break


    def find_hits(self, wf):

        dif=wf-np.roll(wf,1)
        dif_bl=np.median(dif[:100])
        dif_blw=np.sqrt(np.mean((dif[:100]-dif_bl)**2))

        out_of_hit=np.arange(len(wf))

        if np.amin(wf[out_of_hit])<self.bl-self.blw:
            maxi=np.argmin(wf)
            try:
                np.amax(np.nonzero(np.logical_and(wf[:maxi]>self.bl-self.blw, dif[:maxi]>dif_bl-dif_blw))[0])
            except:
                init=0
            else:
                init=np.amax(np.nonzero(np.logical_and(wf[:maxi]>self.bl-self.blw, dif[:maxi]>dif_bl-dif_blw))[0])
            try:
                np.amin(np.nonzero(np.logical_and(wf[maxi:]>self.bl-self.blw, dif[maxi:]>dif_bl-dif_blw))[0])
            except:
                fin=len(wf)-1
            else:
                fin=maxi+np.amin(np.nonzero(np.logical_and(wf[maxi:]>self.bl-self.blw, dif[maxi:]>dif_bl-dif_blw))[0])

            if np.any(dif[init:maxi]<dif_bl-dif_blw) and -np.amin(wf[init:fin])>hit_min_height:
                self.hits.append(Hit(wf,self.bl-self.blw, dif, init,fin))

            out_of_hit=np.delete(out_of_hit, np.arange(init,fin+1))

        while np.amin(wf[out_of_hit])<self.bl-self.blw:
            maxi=np.nonzero(wf==np.amin(wf[out_of_hit]))[0][0]
            left=0
            right=len(wf)-1

            for hit in self.hits:
                if hit.init>left and hit.init<maxi:
                    left=hit.init
                if hit.fin<right and hit.fin>maxi:
                    right=hit.fin

            try:
                np.amax(np.nonzero(np.logical_and(wf[left:maxi]>self.bl-self.blw, dif[left:maxi]>dif_bl-dif_blw))[0])
            except:
                init=left
            else:
                init=left+np.amax(np.nonzero(np.logical_and(wf[left:maxi]>self.bl-self.blw, dif[left:maxi]>dif_bl-dif_blw))[0])
            try:
                np.amin(np.nonzero(np.logical_and(wf[maxi:right]>self.bl-self.blw, dif[maxi:right]>dif_bl-dif_blw))[0])
            except:
                fin=right
            else:
                fin=maxi+np.amin(np.nonzero(np.logical_and(wf[maxi:right]>self.bl-self.blw, dif[maxi:right]>dif_bl-dif_blw))[0])

            if np.any(dif[init:maxi]<dif_bl-dif_blw) and -np.amin(wf[init:fin])>hit_min_height:
                self.hits.append(Hit(wf, self.bl-self.blw, dif, init,fin))


            for i in np.arange(init,fin+1):
                out_of_hit=np.delete(out_of_hit, np.nonzero(out_of_hit==i)[0])

class Peak:
    def __init__(self, peak, amp, tau, sigma):
        self.peak=peak
        self.tau=tau
        self.sigma=sigma
        self.amp=amp

class Group:
    def __init__(self, maxi, left, right, height):
        self.maxi=maxi
        self.left=left
        self.right=right
        self.height=height


class Hit:
    def __init__(self, wf, th, dif, init, fin):
        self.init=init
        self.fin=fin
        self.peaks=[]
        self.groups=[]
        self.chi2=0
        self.area=-np.sum(wf[init:fin])
        self.legit=1

    def find_groups(self,wf, th):
        dif=wf-np.roll(wf,1)
        maxi=np.nonzero(np.logical_and(wf[:-1]<th,np.logical_and(dif[:-1]<0,dif[1:]>0)))[0]
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
            if -wf[m]>hit_min_height:
                self.groups.append(Group(m, left, right, -wf[m]))
        self.groups=sorted(self.groups, key=lambda grp: grp.maxi)
        if len(self.groups)==0:
            self.legit=0
        elif self.groups[-1].maxi-self.init<rise_t:
            self.legit=0
