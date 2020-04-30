import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from fun import find_bl, import_spe, Recon_WF, find_hits, find_init10
from classes import DataSet, Event, WaveForm



PMT_num=21
time_samples=1024
pmts=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
id0=0
id=id0
source='Co57'
type=''
path='/home/gerak/Desktop/DireXeno/'+source+'_190803'+type+'/'
file=open(path+'out.DXD', 'rb')
np.fromfile(file, np.float32, (PMT_num+4)*(2+time_samples)*id0)

rec=np.recarray(15000, dtype=[
    ('id', 'i8'),
    ('pmt', 'i8'),
    ('init', 'i8'),
    ('blw', 'f8'),
    ('height', 'f8')])


def hits_from_data():
    stop=0
    while stop==0:
        Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
        if len(Data)<(PMT_num+4)*(time_samples+2):
            stop=1
        Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
        data=np.array(Data[2:1002,1:21])
        bl=np.median(data[:150], axis=0)
        data=data-bl
        blw=np.sqrt(np.mean(data[:150]**2, axis=0))
        for i, hit in enumerate(hit_out_of_data(data, blw)):
            yield hit


def read_data():
    stop=0
    while stop==0:
        Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
        if len(Data)<(PMT_num+4)*(time_samples+2):
            stop=1
        Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
        data=np.array(Data[2:1002,1:21])
        bl=np.median(data[:150], axis=0)
        yield data-bl

j=0
i=0
for id, data in enumerate(read_data()):
    for hit in hits_from_data():
        rec[j]=id, hit.pmt, hit.init, hit.height
        j+=1
        if j==len(rec):
            np.savez(path+'Allhits/subAllhits{}'.format(i), rec=rec)
            j=0
            i+=1
np.savez(path+'Allhits/subAllhits{}'.format(i), rec=rec[:j])

rec=np.recarray(0, dtype=[
    ('id', 'i8'),
    ('pmt', 'i8'),
    ('init', 'i8'),
    ('blw', 'f8'),
    ('height', 'f8')])
for filename in os.listdir(path+'Allhits/'):
    if filename.startswith('subAllhits'):
        rec=np.append(rec, np.load(path+filename)['rec'])
        os.remove(path+filename)
        print(filename)
np.savez(path+'Allhits', rec=rec)








j=0
start_time = time.time()
while j<len(rec_hits):
    if id%10==0:
        print(source+type+' In PMT {}, Event number {} ({} sec for event). {} Events were saved'.format(pmt, id, (time.time()-start_time)/10, j))
        start_time = time.time()
    Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
    if len(Data)<(PMT_num+4)*(time_samples+2):
        break
    Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
    data=np.array(Data[2:1002,1:21])

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
