import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from fun import find_bl, import_spe, Recon_WF, find_hits, find_init10
from classes import DataSet, Event, WF



PMT_num=20
time_samples=1024
pmts=np.array([0,1,2,3,4,5,6,7,8,9,10,11,17,14,15,16,18,19])
dn=12
up=6
pmt=0
first_ID=2134
spe=import_spe(pmts)
t=np.arange(1000)/5
source='Co57'
type=''
path='/home/gerak/Desktop/DireXeno/'+source+'_190803'+type+'/'
file=open(path+'out.DXD', 'rb')
np.fromfile(file, np.float32, (PMT_num+4)*(2+time_samples)*first_ID)
start_time = time.time()
spectrum=np.zeros((1000, 1000))
mean_WF=np.zeros(1000)
Recon_wf=np.zeros(1000)
Chi2=[]
id=[]
init10=[]
ID=first_ID-1
j=0
while j<len(spectrum[:,0]):
    if ID%10==0:
        print(source+type+' In PMT {}, Event number {} ({} sec for event).'.format(pmt, ID, (time.time()-start_time)/10))
        start_time = time.time()
    Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
    ID+=1
    if len(Data)<(PMT_num+4)*(time_samples+2):
        break
    Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
    data=np.array(Data[2:1002,2:20])
    wf=data[:,np.nonzero(pmt==pmts)[0][0]]
    bl, blw, J=find_bl(wf)
    if blw>10:
        continue
    wf=np.array(wf-bl)
    Wf=WF(pmt, blw)
    find_hits(Wf, wf)
    if len(Wf.hits)>0:
        find_init10(Wf, wf)
        if Wf.init10>len(wf)-250:
            continue
        else:
            wf=np.roll(wf, -Wf.init10+200)
    init10.append(Wf.init10)
    Wf.hits=[]
    find_hits(Wf,wf)
    i=np.nonzero(pmts==pmt)[0][0]
    recon_wf, chi2, T=Recon_WF(Wf, wf, spe[i], 12, 6)
    Chi2.append(chi2)
    id.append(ID)
    h, bins=np.histogram(T, bins=1000, range=[-0.5, 999.5])
    spectrum[j,:]=h
    mean_WF+=wf
    Recon_wf+=recon_wf
    j+=1
    if j%len(spectrum[:,0])==0:
        np.savez(path+'spectra{}/spectrum{}to{}'.format(pmt,first_ID,ID), spectrum=spectrum, Recon_wf=Recon_wf, mean_WF=mean_WF, Chi2=Chi2, ID=id, init10=init10)
        first_ID=ID+1
        spectrum=np.zeros((len(spectrum[:,0]), 1000))
        mean_WF=np.zeros(1000)
        Recon_wf=np.zeros(1000)
        Chi2=[]
        id=[]
        j=0
