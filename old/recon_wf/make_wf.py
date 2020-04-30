import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from fun import find_bl, import_spe, Recon_WF, find_hits, find_init10
from classes import DataSet, Event, WaveForm



PMT_num=20
time_samples=1024
pmts=np.array([0,1,2,3,4,5,6,7,8,9,10,11,17,14,15,16,18,19])
pmt=0
id0=10926
id=id0
source='Co57'
type=''
path='/home/gerak/Desktop/DireXeno/'+source+'_190803'+type+'/'
file=open(path+'out.DXD', 'rb')
np.fromfile(file, np.float32, (PMT_num+4)*(2+time_samples)*id0)

rec=np.recarray(5000, dtype=[
    ('id', 'i8'),
    ('blw', 'f8'),
    ('init', 'i8'),
    ('height', 'i8')])


j=0
start_time = time.time()
while j<len(rec):
    if id%10==0:
        print(source+type+' In PMT {}, Event number {} ({} sec for event). {} Events were saved'.format(pmt, id, (time.time()-start_time)/10, j))
        start_time = time.time()
    Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
    if len(Data)<(PMT_num+4)*(time_samples+2):
        break
    Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
    data=np.array(Data[2:1002,2:20])
    wf=data[:,np.nonzero(pmt==pmts)[0][0]]
    bl, blw=find_bl(wf)
    wf=np.array(wf-bl)
    WF=WaveForm(pmt, blw)
    find_hits(WF, wf)
    for hit in WF.hits:
        rec[j]=id, blw, hit.init, hit.height
        j+=1
        if j==len(rec):
            np.savez(path+'PMT{}/WFs{}to{}'.format(pmt,id0,id-1), rec=rec)
            id0=id
            j=0
    id+=1
np.savez(path+'PMT{}/WFs{}to{}'.format(pmt,id0,id), rec=rec[:j-1])
