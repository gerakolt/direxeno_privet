import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
def func(x, a, m,s):
    return a*np.exp(-0.5*(x-m)**2/s**2)

pmts=[0,7,8]
for i, pmt1 in enumerate(pmts):
    path1='/home/gerak/Desktop/DireXeno/190803/pulser/NEWPMT{}/'.format(pmt1)
    data=np.load(path1+'cuts.npz')
    blw_cut=data['blw_cut']
    height_cut=data['height_cut']
    left=data['left']
    right=data['right']
    rise_time_cut=data['rise_time_cut']

    data=np.load(path1+'spe.npz')
    rec=data['rec']

    rng=np.nonzero(np.logical_and(rec['blw']<blw_cut, np.logical_and(rec['height']>height_cut, np.logical_and(rec['maxi']>left, np.logical_and(rec['maxi']<right, rec['rise_time']>rise_time_cut)))))
    rec1=rec[rng]

    for j, pmt2 in enumerate(pmts):
        if j>i:
            print(pmt1, pmt2)
            path2='/home/gerak/Desktop/DireXeno/190803/pulser/NEWPMT{}/'.format(pmt2)
            data=np.load(path2+'cuts.npz')
            blw_cut=data['blw_cut']
            height_cut=data['height_cut']
            left=data['left']
            right=data['right']
            rise_time_cut=data['rise_time_cut']

            data=np.load(path2+'spe.npz')
            rec=data['rec']

            rng=np.nonzero(np.logical_and(rec['blw']<blw_cut, np.logical_and(rec['height']>height_cut, np.logical_and(rec['maxi']>left, np.logical_and(rec['maxi']<right, rec['rise_time']>rise_time_cut)))))
            rec2=rec[rng]

            ids=rec1['id'][np.isin(rec1['id'], rec2['id'])]
            t=[]
            for id in ids:
                t1=rec1[rec1['id']==id]['init10']
                t2=rec2[rec2['id']==id]['init10']
                if len(t1)==1 and len(t2)==1:
                    t.append(t2[0]-t1[0])
                    # if t2[0]-t1[0]==0:
                    #     print(id)
                    #     sys.exit()

            np.savez('/home/gerak/Desktop/DireXeno/190803/pulser/delays/ts_{}_{}'.format(pmt1, pmt2), ts=np.array(t)/5)
