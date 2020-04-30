import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from fun import find_hits
from classes import WaveForm
import os



PMT_num=20
time_samples=1024
pmts=np.array([0,1,2,3,4,5,6,7,8,9,10,11,17,14,15,16,18,19])
id0=0
id=id0
source='Co57'
type=''
path='/home/gerak/Desktop/DireXeno/190803/'+source+type+'/'
file=open(path+'out.DXD', 'rb')
np.fromfile(file, np.float32, (PMT_num+4)*(2+time_samples)*id0)

rec=np.recarray(5000, dtype=[
    ('id', 'i8'),
    ('pmt', 'i8'),
    ('init_first_hit', 'i8'),
    ('init10', 'f8'),
    ('blw', 'f8'),
    ('height_first_hit', 'f8')])


def hits_from_data(data):
    blw=np.sqrt(np.mean(data[:40]**2, axis=0))
    for i, wf in enumerate(data.T):
        WF=WaveForm(pmts[i], blw[i])
        find_hits(WF, wf)
        if len(WF.hits):
            yield WF.hits[0], pmts[i], blw[i]



def read_data():
    stop=0
    while stop==0:
        Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
        if len(Data)<(PMT_num+4)*(time_samples+2):
            break
        Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
        data=np.array(Data[2:1002,2:20])
        bl=np.median(data[:40], axis=0)
        yield data-bl

j=0
i=0
for id, data in enumerate(read_data()):
    if id%10==0:
        print('In '+source+type+', ID=', id, '({} events were made.)'.format(j))
    for hit, pmt, blw in hits_from_data(data):
        rec[j]=id, pmt, hit.init, hit.init10, blw, hit.height
        j+=1
        if j==len(rec):
            np.savez(path+'subAllhits{}'.format(i), rec=rec)
            j=0
            i+=1
np.savez(path+'subAllhits{}'.format(i), rec=rec[:j])

rec=np.recarray(0, dtype=[
    ('id', 'i8'),
    ('pmt', 'i8'),
    ('init_first_hit', 'i8'),
    ('init10', 'f8'),
    ('blw', 'f8'),
    ('height_first_hit', 'f8')])

for filename in os.listdir(path):
    if filename.startswith('subAllhits'):
        rec=np.append(rec, np.load(path+filename)['rec'])
        os.remove(path+filename)
        print(filename)
np.savez(path+'Allhits', rec=rec)
