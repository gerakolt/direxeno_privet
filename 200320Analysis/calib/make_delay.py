import numpy as np
import time
import matplotlib.pyplot as plt

pmts=[0,8]
for pmt1 in pmts:
    for pmt2 in pmts:
        if pmt2>pmt1:
            dt=[]
            path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt1)
            data=np.load(path+'spe.npz')
            rec=data['rec']
            area=rec['area']
            t=rec['t']
            rise_time=rec['rise_time']
            id=rec['id']
            rise_time_cut=data['rise_time_cut']

            t1=t[np.nonzero(rise_time>rise_time_cut)[0]]
            id1=id[np.nonzero(rise_time>rise_time_cut)[0]]

            path='/home/gerak/Desktop/DireXeno/190803/pulser/PMT{}/'.format(pmt2)
            data=np.load(path+'spe.npz')
            rec=data['rec']
            area=rec['area']
            t=rec['t']
            rise_time=rec['rise_time']
            id=rec['id']
            rise_time_cut=data['rise_time_cut']

            t2=t[np.nonzero(rise_time>rise_time_cut)[0]]
            id2=id[np.nonzero(rise_time>rise_time_cut)[0]]

            for i, id in enumerate(id1):
                print(pmt1, pmt2, i)
                if np.isin(id, id2):
                    dt.append(t1[i]-t2[np.nonzero(id==id2)][0])
            dt=np.array(dt)/5

            plt.hist(dt, bins=50, range=[-10,10])
            np.savez('/home/gerak/Desktop/DireXeno/190803/pulser/delays/pmts_{}_{}'.format(pmt1, pmt2), dt=dt)
            plt.show()
