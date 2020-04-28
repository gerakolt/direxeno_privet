import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy.optimize import curve_fit

path='/home/gerak/Desktop/DireXeno/190803/pulser/EventRecon/'
pmts=[7,8]
Rec=np.recarray(100000, dtype=[
    ('blw', 'f8', len(pmts)),
    ('id', 'i8'),
    ('h', 'i8', (1000, len(pmts))),
    ])
j=0
id=0
for filename in os.listdir(path):
    if filename.endswith(".npz") and filename.startswith("recon"):
        print(filename)
        data=np.load(path+filename)
        rec=data['rec']
        for r in rec:
            Rec[j]['blw']=r['blw']
            Rec[j]['id']=r['id']
            Rec[j]['h']=r['h']
            if r['id']>id:
                id=r['id']
            j+=1
        os.remove(path+filename)
np.savez(path+'recon{}'.format(id), rec=Rec[:j-1])
rec=Rec[:j-1]

def func(x, a, q0):
    y=np.zeros(len(x))
    y[0]=a*(1-q0/(1-q0))
    y[1:]=a*q0**(np.arange(1, len(y)))
    return y


fig, (ax1, ax2)=plt.subplots(2,1)
h1, bins, pa=ax1.hist(np.ravel(rec['h'][:,400:,0]), bins=np.arange(5)-0.5, label='PMT7', histtype='step', linewidth=5)
h2, bins, pa=ax2.hist(np.ravel(rec['h'][:,400:,1]), bins=np.arange(5)-0.5, label='PMT8', histtype='step', linewidth=5)
x=0.5*(bins[1:]+bins[:-1])
rng=np.nonzero(x<6)[0]
p0=[50000, 0.001]
p1, cov=curve_fit(func, x[rng], h1[rng], p0=p0)
p2, cov=curve_fit(func, x[rng], h2[rng], p0=p0)

ax1.plot(x[rng], func(x[rng], *p1), 'ro', label=r'$f_{dc}=$'+'{}'.format(np.round(p1[1], 5)))
ax2.plot(x[rng], func(x[rng], *p2), 'ro', label=r'$f_{dc}=$'+'{}'.format(np.round(p2[1], 5)))

ax1.legend(fontsize=25)
ax2.legend(fontsize=25)
ax2.set_yscale('log')
ax1.set_yscale('log')
fig.text(0.04, 0.5, 'Number of dark PEs in all\nevents all digi points', va='center', ha='center', rotation='vertical', fontsize=35)

plt.show()
