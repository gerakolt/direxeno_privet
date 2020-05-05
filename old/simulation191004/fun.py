import os, sys
import numpy as np
from scipy import signal
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
import sys
from scipy.optimize import curve_fit



def do_smd(data, w):
    if len(np.shape(data))>1:
        mask=np.vstack((np.zeros(w), np.blackman(w)/np.sum(np.blackman(w)), np.zeros(w))).T
        temp1=np.vstack((data[-int(np.floor(w/2)):,:], data, data[:int(np.floor(w/2)),:]))
        temp2=np.vstack((np.zeros(len(temp1[:,0])), temp1.T, np.zeros(len(temp1[:,0])))).T
        smd=signal.convolve2d(temp2, mask, mode='valid')
    else:
        mask=np.blackman(w)/np.sum(np.blackman(w))
        smd=signal.convolve(np.concatenate((data[-int(np.floor(w/2)):], data, data[:int(np.floor(w/2))])), mask, mode='valid')

    return smd



def do_dif(smd):
    return (np.roll(smd,1,axis=0)-np.roll(smd,-1,axis=0))/2




def find_peaks(smd, bl, blw, dif, dif_bl, dif_blw):
    chns=np.array([]).astype(int)
    init=np.array([]).astype(int)
    init10=np.array([]).astype(int)
    initd=np.array([]).astype(int)
    maxi=np.array([]).astype(int)
    fin=np.array([]).astype(int)
    h=np.array([])
    d=np.array([])

    for chn in range(len(smd[0,:])):
        Maxi=np.argmin(smd[:,chn])
        if smd[Maxi,chn]<bl[chn]-blw[chn]:
            maxi=np.append(maxi, Maxi)
            chns=np.append(chns, chn)
            if len(np.nonzero(np.logical_and(smd[:Maxi, chn]>bl[chn]-blw[chn], dif[:Maxi, chn]<dif_bl[chn]+dif_blw[chn]))[0])>0:
                Init=np.amax(np.nonzero(np.logical_and(smd[:Maxi, chn]>bl[chn]-blw[chn], dif[:Maxi, chn]<dif_bl[chn]+dif_blw[chn]))[0])

            else:
                Init=0
            if len(np.nonzero(np.logical_and(smd[Maxi:, chn]>bl[chn]-blw[chn], dif[Maxi:, chn]>dif_bl[chn]-dif_blw[chn]))[0])>0:
                Fin=Maxi+np.amin(np.nonzero(np.logical_and(smd[Maxi:, chn]>bl[chn]-blw[chn], dif[Maxi:, chn]>dif_bl[chn]-dif_blw[chn]))[0])
            else:
                Fin=len(smd[:,0])-1
            init=np.append(init, Init)
            fin=np.append(fin, Fin)

            if Init==0:
                wf_bl=smd[:,chn]
            else:
                wf_bl=smd[:Init,chn]

            bl[chn]=np.median(wf_bl)
            blw[chn]=np.sqrt(np.mean((wf_bl-bl[chn])**2, axis=0))

            #marker=Init

            i=Init-1
            while i>0:
                if smd[i,chn]<bl[chn]-blw[chn] and dif[i,chn]>dif_bl[chn]+dif_blw[chn]:
                    if len(np.nonzero(np.logical_and(smd[i:Init, chn]>bl[chn]-blw[chn], dif[i:Init, chn]>dif_bl[chn]-dif_blw[chn]))[0])>0:
                        Fin=i+np.amin(np.nonzero(np.logical_and(smd[i:Init, chn]>bl[chn]-blw[chn], dif[i:Init, chn]>dif_bl[chn]-dif_blw[chn]))[0])
                    else:
                        Fin=Init-1
                    if len(np.nonzero(np.logical_and(smd[:i, chn]>bl[chn]-blw[chn], dif[:i, chn]<dif_bl[chn]+dif_blw[chn]))[0])>0:
                        Init=np.amax(np.nonzero(np.logical_and(smd[:i, chn]>bl[chn]-blw[chn], dif[:i, chn]<dif_bl[chn]+dif_blw[chn]))[0])
                    else:
                        Init=0
                    i=Init-1
                    init=np.append(init, Init)
                    fin=np.append(fin, Fin)
                    maxi=np.append(maxi, Init+np.argmin(smd[Init:Fin, chn]))
                    chns=np.append(chns, chn)

                    #marker-=len(range(Init,Fin+1))
                    wf_bl=np.delete(wf_bl, range(Init,Fin+1))
                    if len(wf_bl)>50:
                        bl[chn]=np.median(wf_bl)
                        blw[chn]=np.sqrt(np.mean((wf_bl-bl[chn])**2, axis=0))

                else:
                    i-=1

            Fin=np.amax(fin[chns==chn])
            i=Fin+1
            while i<len(smd[:,0])-1:
                if smd[i,chn]<bl[chn]-blw[chn] and dif[i,chn]>dif_bl[chn]+dif_blw[chn]:
                    if len(np.nonzero(np.logical_and(smd[Fin:i, chn]>bl[chn]-blw[chn], dif[Fin:i, chn]<dif_bl[chn]+dif_blw[chn]))[0])>0:
                        Init=Fin+np.amax(np.nonzero(np.logical_and(smd[Fin:i, chn]>bl[chn]-blw[chn], dif[Fin:i, chn]<dif_bl[chn]+dif_blw[chn]))[0])
                    else:
                        Init=Fin+1
                    if len(np.nonzero(np.logical_and(smd[i:, chn]>bl[chn]-blw[chn], dif[i:, chn]>dif_bl[chn]-dif_blw[chn]))[0])>0:
                        Fin=i+np.amin(np.nonzero(np.logical_and(smd[i:, chn]>bl[chn]-blw[chn], dif[i:, chn]>dif_bl[chn]-dif_blw[chn]))[0])
                    else:
                        Fin=len(smd[:,0])-1
                    i=Fin
                    init=np.append(init, Init)
                    fin=np.append(fin, Fin)
                    maxi=np.append(maxi, Init+np.argmin(smd[Init:Fin, chn]))
                    chns=np.append(chns, chn)
                    # wf_bl=np.delete(wf_bl, np.arange(Init,Fin+1)-marker)
                    # bl[chn]=np.median(wf_bl)
                    # blw[chn]=np.sqrt(np.mean((wf_bl-bl[chn])**2, axis=0))
                    # marker-=len(range(Init,Fin+1))
                else:
                    i+=1


    return chns, init, maxi, fin


def fix_peaks(chns, init, maxi, fin, smd, bl, blw, dif, dif_bl, dif_blw):
    same=np.array([]).astype(int)
    for i in range(len(chns)):
        while smd[init[i],chns[i]]<bl[chns[i]]-blw[chns[i]]  and init[i]>0:
            init[i]-=1
            #wf_bl=np.delete(wf_bl, np.nonzero(wf_bl==smd[init[i]])[0])
            #bl=np.median(wf_bl)
            #blw=np.sqrt(np.mean((wf_bl-bl)**2))
        while smd[fin[i],chns[i]]<bl[chns[i]]-blw[chns[i]]  and fin[i]<len(smd[:,0])-1:
            fin[i]+=1
        maxi[i]=init[i]+np.argmin(smd[init[i]:fin[i],chns[i]])
        if len(np.nonzero(np.logical_and(maxi==maxi[i], chns==chns[i]))[0])>1:
            same=np.append(same, np.nonzero(np.logical_and(maxi==maxi[i], chns==chns[i]))[0][1:])
    chns=np.delete(chns, same)
    init=np.delete(init, same)
    maxi=np.delete(maxi, same)
    fin=np.delete(fin, same)

    return chns, init, maxi, fin


def analize_peaks(chns, init, maxi, fin, data, smd, bl, blw, dif):
    area=np.zeros(len(chns))
    h=np.zeros(len(chns))
    neg_peak=np.zeros(len(chns))
    d=np.zeros(len(chns))
    d_ind=np.zeros(len(chns))
    init10=np.zeros(len(chns)).astype(int)
    sp=np.array([]).astype(int)
    sp_p=np.array([]).astype(int)
    n_peaks=np.zeros(len(chns))
    n_pmts=np.ones(len(chns))*len(np.unique(chns))
    n_subpeaks=np.zeros(len(chns))
    bl_area=np.sum(data, axis=0)
    bl_len=len(data[:,0])*np.ones(len(data[0,:]))
    for chn in range(len(data[0,:])):
        Init=np.sort(init[chns==chn])
        Fin=np.sort(fin[chns==chn])
        for i in range(len(Init)):
            n_peaks[i]+=1
            bl_area[chn]-=np.sum(data[Init[i]:Fin[i],chn])
            bl_len[chn]-=Fin[i]-Init[i]

    for i in range(len(chns)):
        area[i]=(fin[i]-init[i])/(bl_len[chns[i]])*bl_area[chns[i]]-np.sum(data[init[i]:fin[i], chns[i]])
        h[i]=np.amax(bl[chns[i]]-data[init[i]:fin[i], chns[i]])
        neg_peak=np.amax(data[init[i]:fin[i], chns[i]]-bl[chns[i]])
        d_ind[i]=np.argmax(dif[init[i]:fin[i], chns[i]])
        d[i]=np.amax(dif[init[i]:fin[i], chns[i]])
        if len(np.nonzero((bl[chns[i]]-data[init[i]:maxi[i], chns[i]])<0.1*h[i])[0])>0:
            init10[i]=init[i]+np.amax(np.nonzero((bl[chns[i]]-data[init[i]:maxi[i], chns[i]])<0.1*h[i])[0])
        else:
            init10[i]=init[i]
        Sp=init[i]+np.nonzero(np.logical_and(
                    np.roll(smd[init[i]:fin[i], chns[i]],2)>np.roll(smd[init[i]:fin[i], chns[i]],1), np.logical_and(
                    np.roll(smd[init[i]:fin[i], chns[i]],1)>smd[init[i]:fin[i], chns[i]], np.logical_and(
                    np.roll(smd[init[i]:fin[i], chns[i]],-2)>np.roll(smd[init[i]:fin[i], chns[i]],-1),
                    np.roll(smd[init[i]:fin[i], chns[i]],-1)>smd[init[i]:fin[i],chns[i]]))))[0]
        sp=np.append(sp, Sp).astype(int)
        sp_p=np.append(sp_p, i*np.ones(len(Sp))).astype(int)
        n_subpeaks[i]+=len(Sp)

    return area, h, d, d_ind, init10, sp, sp_p, n_peaks, n_pmts, n_subpeaks




















def Find_Peaks(smd, bl, blw, dif, dif_bl, dif_blw):
    init=np.array([]).astype(int)
    init10=np.array([]).astype(int)
    initd=np.array([]).astype(int)
    fin=np.array([]).astype(int)
    h=np.array([])
    d=np.array([])
    wf_bl=smd

    Maxi=np.argmin(smd)
    if smd[Maxi]<bl-blw:
        if len(np.nonzero(np.logical_and(smd[:Maxi]>bl-blw, dif[:Maxi]<dif_bl+dif_blw))[0])>0:
            Init=np.amax(np.nonzero(np.logical_and(smd[:Maxi]>bl-blw, dif[:Maxi]<dif_bl+dif_blw))[0])
        else:
            Init=0
        if len(np.nonzero(np.logical_and(smd[Maxi:]>bl-blw, dif[Maxi:]>dif_bl-dif_blw))[0])>0:
            Fin=Maxi+np.amin(np.nonzero(np.logical_and(smd[Maxi:]>bl-blw, dif[Maxi:]>dif_bl-dif_blw))[0])
        else:
            Fin=len(smd)-1
        init=np.append(init, Init)
        fin=np.append(fin, Fin)

        if Init==0:
            wf_bl=smd
        else:
            wf_bl=smd[:Init]
        #marker=Init
        bl=np.median(wf_bl)
        blw=np.sqrt(np.mean((wf_bl-bl)**2))

        i=Init-1
        while i>0:
            if smd[i]<bl-blw and dif[i]>dif_bl+dif_blw:
                if len(np.nonzero(np.logical_and(smd[i:Init]>bl-blw, dif[i:Init]>dif_bl-dif_blw))[0])>0:
                    Fin=i+np.amin(np.nonzero(np.logical_and(smd[i:Init]>bl-blw, dif[i:Init]>dif_bl-dif_blw))[0])
                else:
                    Fin=Init-1
                if len(np.nonzero(np.logical_and(smd[:i]>bl-blw, dif[:i]<dif_bl+dif_blw))[0])>0:
                    Init=np.amax(np.nonzero(np.logical_and(smd[:i]>bl-blw, dif[:i]<dif_bl+dif_blw))[0])
                else:
                    Init=0
                i=Init-1
                init=np.append(init, Init)
                fin=np.append(fin, Fin)

                # marker-=len(range(Init,Fin+1))
                wf_bl=np.delete(wf_bl, range(Init,Fin+1))
                bl=np.median(wf_bl)
                blw=np.sqrt(np.mean((wf_bl-bl)**2))
            else:
                i-=1

        Fin=np.amax(fin)
        i=Fin+1
        while i<len(smd)-1:
            if smd[i]<bl-blw and dif[i]>dif_bl+dif_blw:
                if len(np.nonzero(np.logical_and(smd[Fin:i]>bl-blw, dif[Fin:i]<dif_bl+dif_blw))[0])>0:
                    Init=Fin+np.amax(np.nonzero(np.logical_and(smd[Fin:i]>bl-blw, dif[Fin:i]<dif_bl+dif_blw))[0])
                else:
                    Init=Fin+1
                if len(np.nonzero(np.logical_and(smd[i:]>bl-blw, dif[i:]>dif_bl-dif_blw))[0])>0:
                    Fin=i+np.amin(np.nonzero(np.logical_and(smd[i:]>bl-blw, dif[i:]>dif_bl-dif_blw))[0])
                else:
                    Fin=len(smd)-1
                i=Fin+1
                init=np.append(init, Init)
                fin=np.append(fin, Fin)
                # wf_bl=np.delete(wf_bl, np.arange(Init,Fin+1)-marker)
                # bl=np.median(wf_bl)
                # blw=np.sqrt(np.mean((wf_bl-bl)**2))
                # marker-=len(range(Init,Fin+1))
            else:
                i+=1

    return init, fin, bl, blw, wf_bl


def Fix_Peaks(init, fin, smd, bl, blw, dif, dif_bl, dif_blw, wf_bl):
    same=np.array([]).astype(int)
    maxi=np.zeros(len(init)).astype(int)
    for i in range(len(init)):
        while smd[init[i]]<bl-blw and init[i]>0:
            init[i]-=1
            wf_bl=np.delete(wf_bl, np.nonzero(wf_bl==smd[init[i]])[0])
            bl=np.median(wf_bl)
            blw=np.sqrt(np.mean((wf_bl-bl)**2))
        while smd[fin[i]]<bl-blw  and fin[i]<len(smd)-1:
            fin[i]+=1
            # wf_bl=np.delete(wf_bl, np.nonzero(wf_bl==smd[fin[i]])[0])
            # bl=np.median(wf_bl)
            # blw=np.sqrt(np.mean((wf_bl-bl)**2))
        maxi[i]=init[i]+np.argmin(smd[init[i]:fin[i]])
        if len(np.nonzero(maxi==maxi[i])[0])>1:
            same=np.append(same, np.nonzero(maxi==maxi[i])[0][1:])
    init=np.delete(init, same)
    maxi=np.delete(maxi, same)
    fin=np.delete(fin, same)

    return init, maxi, fin, bl, blw, wf_bl


def Analize_Peaks(init, maxi, fin, data, smd, bl, blw, dif):
    area=np.zeros(len(init))
    h=np.zeros(len(init))
    d=np.zeros(len(init))
    d_ind=np.zeros(len(init)).astype(int)
    init10=np.zeros(len(init)).astype(int)
    sp=np.array([]).astype(int)
    sp_p=np.array([]).astype(int)
    n_peaks=np.zeros(len(init)).astype(int)
    n_subpeaks=np.zeros(len(init)).astype(int)
    bl_area=np.sum(data)
    bl_len=len(data)
    for i in range(len(init)):
        n_peaks[i]+=1
        bl_area-=np.sum(data[init[i]:fin[i]])
        bl_len-=fin[i]-init[i]

    for i in range(len(init)):
        area[i]=(fin[i]-init[i])/(bl_len)*bl_area-np.sum(data[init[i]:fin[i]])
        h[i]=np.amax(bl-data[init[i]:fin[i]])
        d_ind[i]=init[i]+np.argmax(dif[init[i]:fin[i]])
        d[i]=np.amax(dif[init[i]:fin[i]])
        if len(np.nonzero((bl-data[init[i]:maxi[i]])<0.1*h[i])[0])>0:
            init10[i]=init[i]+np.amax(np.nonzero((bl-data[init[i]:maxi[i]])<0.1*h[i])[0])
        else:
            init10[i]=init[i]
        Sp=init[i]+np.nonzero(np.logical_and(
                    np.roll(smd[init[i]:fin[i]],2)>np.roll(smd[init[i]:fin[i]],1), np.logical_and(
                    np.roll(smd[init[i]:fin[i]],1)>smd[init[i]:fin[i]], np.logical_and(
                    np.roll(smd[init[i]:fin[i]],-2)>np.roll(smd[init[i]:fin[i]],-1),
                    np.roll(smd[init[i]:fin[i]],-1)>smd[init[i]:fin[i]]))))[0]
        sp=np.append(sp, Sp).astype(int)
        sp_p=np.append(sp_p, i*np.ones(len(Sp))).astype(int)
        n_subpeaks[i]+=len(Sp)

    return area, h, d, d_ind, init10, sp, sp_p, n_peaks, n_subpeaks


def Fit_Decay(SMD, Init, Maxi, Fin, func):
    tau=np.zeros(len(Init))
    const=np.zeros(len(Init))
    same=np.array([]).astype(int)
    change=0
    for i in range(len(Init)):
        p=(SMD[Maxi[i]],32*0.2)
        try:
            [const[i], tau[i]], cov=curve_fit(func, np.arange(Maxi[i],Fin[i]), SMD[Maxi[i]:Fin[i]], p0=p, method='lm')
            perr = np.sqrt(np.diag(cov))
        except:
            tau[i]=0
            const[i]=0
    if tau[np.argmin(SMD[Maxi])]>10*5:
        Fin[np.argmin(SMD[Maxi])]=len(SMD)-1
        change=1
    # if len(np.nonzero(Fin==len(SMD)-1)[0])>1:
    #     change=1
    #     same=np.append(same, np.nonzero(SMD[Maxi[Fin==len(SMD)-1]]!=np.amin(SMD[Maxi[Fin==len(SMD)-1]]))[0])
    # Init=np.delete(Init, same)
    # Maxi=np.delete(Maxi, same)
    # Fin=np.delete(Fin, same)
    if change:
        dlt=np.nonzero(Init>Init[np.argmin(SMD[Maxi])])[0]
        Init=np.delete(Init, dlt)
        Fin=np.delete(Fin, dlt)
        Maxi=np.delete(Maxi, dlt)
        tau=np.zeros(len(Init))
        const=np.zeros(len(Init))
        for i in range(len(Init)):
            p=(SMD[Maxi[i]],32*0.2)
            try:
                [const[i], tau[i]], cov=curve_fit(func, np.arange(Maxi[i],Fin[i]), SMD[Maxi[i]:Fin[i]], p0=p, method='lm')
                perr = np.sqrt(np.diag(cov))
            except:
                tau[i]=0
                const[i]=0

    return const, tau, Init, Maxi, Fin
