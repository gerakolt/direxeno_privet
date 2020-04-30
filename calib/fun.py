import os, sys
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
import sys
from classes import Peak



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



def find_bl(wf):
    i=90
    j=i
    blw=np.std(wf[i-90:i+90])
    while i+90<len(wf):
        i+=1
        std=np.std(wf[i-90:i+90])
        if std<blw:
            blw=std
            j=i
    return np.mean(wf[j-90:j+90]), blw, j


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



def do_dif(smd):
    if len(np.shape(smd))>1:
        return (np.roll(smd,1,axis=0)-np.roll(smd,-1,axis=0))/2
    else:
        return (np.roll(smd,1)-np.roll(smd,-1))/2

def find_peaks(wf, blw, pmt, l, r):
    wf_copy=np.array(wf)
    dif=do_dif(wf)
    dif=dif-np.median(dif[:200], axis=0)
    dif_blw=np.sqrt(np.mean((dif[:200])**2, axis=0))
    maxi=np.argmin(wf)
    counter=0
    while wf[maxi]<-3*blw and counter<1000:
        counter+=1
        peak=Peak(pmt, maxi, -wf[maxi], blw)
        if len(np.nonzero(np.logical_and(wf[:maxi]>-blw, dif[:maxi]<dif_blw))[0])>0:
            init=np.amax(np.nonzero(np.logical_and(wf[:maxi]>-blw, dif[:maxi]<dif_blw))[0])
        else:
            init=0
        if len(np.nonzero(np.logical_and(wf[maxi:]>-blw, dif[maxi:]>-dif_blw))[0])>0:
            fin=maxi+np.amin(np.nonzero(np.logical_and(wf[maxi:]>-blw, dif[maxi:]>-dif_blw))[0])
        else:
            fin=len(wf)-1
        peak.init=init
        peak.fin=fin
        peak.area=-np.sum(wf[init:fin])
        if len(np.nonzero(wf[peak.init:peak.maxi]>-0.1*peak.height)[0])>0:
            peak.init10=peak.init+np.amax(np.nonzero(wf[peak.init:peak.maxi]>-0.1*peak.height)[0])
        else:
            peak.init10=peak.init
        wf[init:fin+1]=0
        maxi=np.argmin(wf)
        yield (-np.sum(wf_copy[peak.init10+l-200:peak.init10+r-200]), peak)

def show_peaks(wfs, blw, pmts):
    dif=do_dif(wfs)
    dif=dif-np.median(dif[:150], axis=0)
    dif_blw=np.sqrt(np.mean((dif[:150])**2, axis=0))
    for i, wf in enumerate(wfs.T):
        WF=np.array(wf)
        maxi=np.argmin(wf)
        counter=0
        while wf[maxi]<-3*blw[i] and counter<1000:
            counter+=1
            peak=Peak(pmts[i], maxi, -wf[maxi], blw[i])
            if len(np.nonzero(np.logical_and(wf[:maxi]>-blw[i], dif[:maxi,i]<dif_blw[i]))[0])>0:
                init=np.amax(np.nonzero(np.logical_and(wf[:maxi]>-blw[i], dif[:maxi,i]<dif_blw[i]))[0])
            else:
                init=0
            if len(np.nonzero(np.logical_and(wf[maxi:]>-blw[i], dif[maxi:,i]>-dif_blw[i]))[0])>0:
                fin=maxi+np.amin(np.nonzero(np.logical_and(wf[maxi:]>-blw[i], dif[maxi:,i]>-dif_blw[i]))[0])
            else:
                fin=len(wf)-1
            peak.init=init
            peak.fin=fin
            peak.area=-np.sum(wf[init:fin])
            if len(np.nonzero(wf[peak.init:peak.maxi]>-0.1*peak.height)[0])>0:
                peak.init10=peak.init+np.amax(np.nonzero(wf[peak.init:peak.maxi]>-0.1*peak.height)[0])
            else:
                peak.init10=peak.init
            wf[init:fin+1]=0
            maxi=np.argmin(wf)

            plt.figure()
            x=np.arange(1000)
            plt.title('PMT{}'.format(i), fontsize=25)
            plt.plot(x, WF, '.-', alpha=1)
            plt.axhline(y=0, color='k')
            plt.fill_between(x=x[:150], y1=-blw[i], y2=0, color='y', alpha=1, label='BLW')
            # plt.fill_between(x[peak.init:peak.fin], y1=WF[peak.init:peak.fin], y2=0, alpha=0.5)
            plt.legend(fontsize=25)
            plt.show()
            yield peak



# def find_peaks(data, wf):
#     smd=np.array(data)
#     maxi=np.argmin(smd)
#     dif=do_dif(smd)
#     dif=dif-np.median(dif[:200])
#     dif_blw=np.sqrt(np.mean((dif[:200])**2))
#     counter=0
#     while smd[maxi]<-3*wf.blw and counter<1000:
#         counter+=1
#         peak=Peak(maxi,-smd[maxi])
#         if len(np.nonzero(np.logical_and(smd[:maxi]>-wf.blw, dif[:maxi]<dif_blw))[0])>0:
#             init=np.amax(np.nonzero(np.logical_and(smd[:maxi]>-wf.blw, dif[:maxi]<dif_blw))[0])
#         else:
#             init=0
#         if len(np.nonzero(np.logical_and(smd[maxi:]>-wf.blw, dif[maxi:]>-dif_blw))[0])>0:
#             fin=maxi+np.amin(np.nonzero(np.logical_and(smd[maxi:]>-wf.blw, dif[maxi:]>-dif_blw))[0])
#         else:
#             fin=len(smd)-1
#         peak.init=init
#         peak.fin=fin
#         wf.peaks.append(peak)
#         smd[init:fin+1]=0
#         maxi=np.argmin(smd)



def analize_peaks(smd, wf):
    for peak in wf.peaks:
        peak.area=-np.sum(smd[peak.init:peak.fin+1])
        if len(np.nonzero(smd[peak.init:peak.maxi]>-0.1*peak.height)[0])>0:
            peak.init10=peak.init+np.amax(np.nonzero(smd[peak.init:peak.maxi]>-0.1*peak.height)[0])
        else:
            peak.init10=peak.init
