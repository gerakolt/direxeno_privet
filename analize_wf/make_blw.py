import numpy as np
import time
import sys
import matplotlib.pyplot as plt

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

BLW=np.zeros(len(pmts))
for id, data in enumerate(read_data()):
    if id%100==0:
        print(id, source, type)
    BLW=np.vstack((BLW, np.sqrt(np.mean(data[:40]**2, axis=0))))
    # if np.sqrt(np.mean(data[:150]**2, axis=0))[0]<20:
    #     plt.plot(data[:,0], 'k.')
    #     plt.show()
np.savez(path+'BLW_table', BLW=BLW[1:])
